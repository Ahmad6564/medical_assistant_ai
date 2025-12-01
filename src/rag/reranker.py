"""
Cross-encoder re-ranker for improving retrieval quality.
Re-ranks retrieved documents using a more sophisticated model.
"""

import torch
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import numpy as np

from .retriever import RetrievalResult
from .vector_store import Document
from ..utils import get_logger, get_device

logger = get_logger(__name__)


class Reranker:
    """Base class for re-rankers."""
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """
        Initialize re-ranker.
        
        Args:
            model_name: Name of the model (optional)
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        # For backward compatibility, delegate to CrossEncoderReranker if model_name provided
        if model_name:
            self._impl = CrossEncoderReranker(model_name=model_name, **kwargs)
        else:
            self._impl = None
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank retrieval results.
        
        Args:
            query: Query string
            results: Initial retrieval results
            top_k: Number of results to return after re-ranking
            
        Returns:
            Re-ranked results
        """
        if self._impl:
            return self._impl.rerank(query, results, top_k)
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Cross-encoder based re-ranker for precise relevance scoring."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[torch.device] = None,
        batch_size: int = 32
    ):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.batch_size = batch_size
        
        # Load cross-encoder model
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name, device=str(self.device))
        
        logger.info(f"Initialized CrossEncoderReranker on {self.device}")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Query string
            results: Initial retrieval results
            top_k: Number of results to return (default: all)
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        if top_k is None:
            top_k = len(results)
        
        # Prepare query-document pairs
        pairs = [[query, result.document.content] for result in results]
        
        # Get cross-encoder scores
        logger.debug(f"Re-ranking {len(pairs)} documents")
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Update scores and sort
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_result = RetrievalResult(
                document=result.document,
                score=float(score),
                retrieval_method=f"{result.retrieval_method}_reranked",
                rank=result.rank
            )
            reranked_results.append(reranked_result)
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i
        
        return reranked_results[:top_k]


class EnsembleReranker(Reranker):
    """
    Ensemble re-ranker combining multiple scoring methods.
    """
    
    def __init__(
        self,
        rerankers: List[Reranker],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble re-ranker.
        
        Args:
            rerankers: List of re-ranker instances
            weights: Weights for each re-ranker (default: equal weights)
        """
        self.rerankers = rerankers
        
        if weights is None:
            self.weights = [1.0 / len(rerankers)] * len(rerankers)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"Initialized EnsembleReranker with {len(rerankers)} re-rankers")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank using ensemble of re-rankers.
        
        Args:
            query: Query string
            results: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        if top_k is None:
            top_k = len(results)
        
        # Get scores from each re-ranker
        all_scores = []
        for reranker in self.rerankers:
            reranked = reranker.rerank(query, results, top_k=None)
            scores = [r.score for r in reranked]
            all_scores.append(scores)
        
        # Combine scores with weights
        combined_scores = np.zeros(len(results))
        for scores, weight in zip(all_scores, self.weights):
            # Normalize scores to 0-1 range
            scores_array = np.array(scores)
            if scores_array.max() > scores_array.min():
                normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
            else:
                normalized = scores_array
            
            combined_scores += weight * normalized
        
        # Create re-ranked results
        reranked_results = []
        for result, score in zip(results, combined_scores):
            reranked_result = RetrievalResult(
                document=result.document,
                score=float(score),
                retrieval_method="ensemble_reranked",
                rank=result.rank
            )
            reranked_results.append(reranked_result)
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i
        
        return reranked_results[:top_k]


class MMRReranker(Reranker):
    """
    Maximal Marginal Relevance (MMR) re-ranker for diversity.
    Balances relevance with diversity to avoid redundant results.
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR re-ranker.
        
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param
        logger.info(f"Initialized MMRReranker (lambda={lambda_param})")
    
    def _compute_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Compute similarity between two documents.
        Uses simple word overlap for efficiency.
        """
        if doc1.embedding is not None and doc2.embedding is not None:
            # Use cosine similarity of embeddings
            dot_product = np.dot(doc1.embedding, doc2.embedding)
            norm1 = np.linalg.norm(doc1.embedding)
            norm2 = np.linalg.norm(doc2.embedding)
            return dot_product / (norm1 * norm2)
        else:
            # Fall back to word overlap
            words1 = set(doc1.content.lower().split())
            words2 = set(doc2.content.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank using MMR for diversity.
        
        Args:
            query: Query string
            results: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Re-ranked results with diversity
        """
        if not results:
            return []
        
        if top_k is None:
            top_k = len(results)
        
        # Start with empty selected set
        selected: List[RetrievalResult] = []
        remaining = list(results)
        
        # Iteratively select documents
        while len(selected) < top_k and remaining:
            if not selected:
                # First document: highest relevance
                best_idx = 0
                best_score = remaining[0].score
                for i, result in enumerate(remaining):
                    if result.score > best_score:
                        best_score = result.score
                        best_idx = i
            else:
                # Subsequent documents: balance relevance and diversity
                best_idx = 0
                best_mmr_score = float('-inf')
                
                for i, result in enumerate(remaining):
                    # Relevance component
                    relevance = result.score
                    
                    # Diversity component (max similarity to selected documents)
                    max_similarity = max(
                        self._compute_similarity(result.document, sel.document)
                        for sel in selected
                    )
                    
                    # MMR score
                    mmr_score = (
                        self.lambda_param * relevance -
                        (1 - self.lambda_param) * max_similarity
                    )
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = i
            
            # Move best document from remaining to selected
            selected.append(remaining.pop(best_idx))
        
        # Update ranks
        for i, result in enumerate(selected):
            result.rank = i
            result.retrieval_method = f"{result.retrieval_method}_mmr"
        
        return selected


def create_reranker(
    reranker_type: str = "cross_encoder",
    **kwargs
) -> Reranker:
    """
    Factory function to create re-ranker.
    
    Args:
        reranker_type: Type of re-ranker ("cross_encoder", "mmr", "ensemble")
        **kwargs: Arguments for the re-ranker
        
    Returns:
        Reranker instance
    """
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == "mmr":
        return MMRReranker(**kwargs)
    elif reranker_type == "ensemble":
        return EnsembleReranker(**kwargs)
    else:
        raise ValueError(f"Unknown re-ranker type: {reranker_type}")
