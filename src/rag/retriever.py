"""
Hybrid retrieval system combining dense and sparse search.
Implements BM25 sparse retrieval and dense vector search.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import re

from .vector_store import VectorStore, Document
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    document: Document
    score: float
    retrieval_method: str  # "dense", "sparse", or "hybrid"
    rank: int


class SparseRetriever:
    """BM25-based sparse retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents: List[Document] = []
        self.tokenized_corpus = []
        
        logger.info(f"Initialized BM25 retriever (k1={k1}, b={b})")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to BM25 index.
        
        Args:
            documents: List of documents
        """
        if not documents:
            return
        
        self.documents.extend(documents)
        
        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc.content)
            for doc in self.documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Added {len(documents)} documents to BM25 index (total: {len(self.documents)})")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search using BM25.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25 is None or len(self.documents) == 0:
            logger.warning("No documents in BM25 index")
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return results
        results = [
            (self.documents[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return results


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        use_sparse: bool = True,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Dense vector store
            use_sparse: Whether to use sparse retrieval
            sparse_weight: Weight for sparse scores
            dense_weight: Weight for dense scores
        """
        self.vector_store = vector_store
        self.use_sparse = use_sparse
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        
        if use_sparse:
            self.sparse_retriever = SparseRetriever()
        else:
            self.sparse_retriever = None
        
        # Normalize weights
        total_weight = sparse_weight + dense_weight
        self.sparse_weight = sparse_weight / total_weight
        self.dense_weight = dense_weight / total_weight
        
        logger.info(
            f"Initialized hybrid retriever (sparse: {self.sparse_weight:.2f}, "
            f"dense: {self.dense_weight:.2f})"
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to both retrievers.
        
        Args:
            documents: List of documents
        """
        # Add to dense vector store
        self.vector_store.add_documents(documents)
        
        # Add to sparse retriever
        if self.use_sparse and self.sparse_retriever:
            self.sparse_retriever.add_documents(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        retrieval_mode: str = "hybrid"  # "hybrid", "dense", or "sparse"
    ) -> List[RetrievalResult]:
        """
        Search using hybrid retrieval.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Metadata filters
            retrieval_mode: Retrieval mode
            
        Returns:
            List of RetrievalResult objects
        """
        if retrieval_mode == "dense":
            return self._dense_search(query, top_k, filter_dict)
        elif retrieval_mode == "sparse":
            if not self.use_sparse:
                logger.warning("Sparse retrieval not enabled, falling back to dense")
                return self._dense_search(query, top_k, filter_dict)
            return self._sparse_search(query, top_k)
        else:  # hybrid
            return self._hybrid_search(query, top_k, filter_dict)
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Dense vector search."""
        results = self.vector_store.search(query, top_k, filter_dict)
        
        return [
            RetrievalResult(
                document=doc,
                score=score,
                retrieval_method="dense",
                rank=i
            )
            for i, (doc, score) in enumerate(results)
        ]
    
    def _sparse_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Sparse BM25 search."""
        if self.sparse_retriever is None:
            return []
        
        results = self.sparse_retriever.search(query, top_k)
        
        return [
            RetrievalResult(
                document=doc,
                score=score,
                retrieval_method="sparse",
                rank=i
            )
            for i, (doc, score) in enumerate(results)
        ]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining dense and sparse retrieval.
        
        Uses Reciprocal Rank Fusion (RRF) for score combination.
        """
        # Get results from both retrievers
        dense_results = self.vector_store.search(query, top_k * 2, filter_dict)
        
        if self.use_sparse and self.sparse_retriever:
            sparse_results = self.sparse_retriever.search(query, top_k * 2)
        else:
            sparse_results = []
        
        # Create document score map
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # Add dense scores
        for rank, (doc, score) in enumerate(dense_results):
            doc_key = doc.content  # Use content as key
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "document": doc,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "dense_rank": float('inf'),
                    "sparse_rank": float('inf')
                }
            
            doc_scores[doc_key]["dense_score"] = score
            doc_scores[doc_key]["dense_rank"] = rank
        
        # Add sparse scores
        for rank, (doc, score) in enumerate(sparse_results):
            doc_key = doc.content
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "document": doc,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "dense_rank": float('inf'),
                    "sparse_rank": float('inf')
                }
            
            doc_scores[doc_key]["sparse_score"] = score
            doc_scores[doc_key]["sparse_rank"] = rank
        
        # Calculate hybrid scores using RRF
        k = 60  # RRF constant
        for doc_key, scores in doc_scores.items():
            # Reciprocal Rank Fusion
            rrf_score = (
                self.dense_weight / (k + scores["dense_rank"] + 1) +
                self.sparse_weight / (k + scores["sparse_rank"] + 1)
            )
            scores["hybrid_score"] = rrf_score
        
        # Sort by hybrid score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        # Return top-k results
        results = [
            RetrievalResult(
                document=item["document"],
                score=item["hybrid_score"],
                retrieval_method="hybrid",
                rank=i
            )
            for i, item in enumerate(sorted_docs[:top_k])
        ]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "use_sparse": self.use_sparse,
            "sparse_weight": self.sparse_weight,
            "dense_weight": self.dense_weight
        }
        
        if self.sparse_retriever:
            stats["num_documents_sparse"] = len(self.sparse_retriever.documents)
        
        return stats


class ContextualRetriever(HybridRetriever):
    """
    Enhanced retriever with contextual understanding.
    Adds query context and document context to improve retrieval.
    """
    
    def __init__(self, vector_store: VectorStore, **kwargs):
        """Initialize contextual retriever."""
        super().__init__(vector_store, **kwargs)
        self.query_history: List[str] = []
    
    def search_with_context(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search with conversation context.
        
        Args:
            query: Current query
            conversation_history: Previous queries/responses
            top_k: Number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of RetrievalResult objects
        """
        # Enhance query with context
        enhanced_query = query
        
        if conversation_history:
            # Add recent context (last 2 turns)
            context = " ".join(conversation_history[-2:])
            enhanced_query = f"{context} {query}"
        
        # Perform search
        results = self.search(enhanced_query, top_k, **kwargs)
        
        # Store query in history
        self.query_history.append(query)
        
        return results
