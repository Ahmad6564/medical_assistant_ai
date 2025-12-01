"""
Medical RAG (Retrieval-Augmented Generation) System.
Main interface for question answering with medical literature.
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

from .vector_store import VectorStore, Document, create_vector_store
from .retriever import HybridRetriever, ContextualRetriever
from .reranker import Reranker, create_reranker
from .query_processing import MedicalQueryEnhancer, ProcessedQuery
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_info: Dict[str, Any]
    retrieval_info: Dict[str, Any]


class DocumentChunker:
    """Chunk documents for better retrieval."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separator: Separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        logger.info(f"Initialized DocumentChunker (size={chunk_size}, overlap={chunk_overlap})")
    
    def chunk_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of Document chunks
        """
        if metadata is None:
            metadata = {}
        
        # Split by separator first
        sections = content.split(self.separator)
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is small enough, add to current chunk
            if len(current_chunk) + len(section) <= self.chunk_size:
                current_chunk += section + self.separator
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_idx
                    chunks.append(Document(
                        content=current_chunk.strip(),
                        metadata=chunk_metadata,
                        doc_id=f"{metadata.get('doc_id', 'doc')}_{chunk_idx}"
                    ))
                    chunk_idx += 1
                
                # Start new chunk with overlap
                if len(section) > self.chunk_size:
                    # Split large section
                    words = section.split()
                    current_words = []
                    
                    for word in words:
                        current_words.append(word)
                        current_text = " ".join(current_words)
                        
                        if len(current_text) >= self.chunk_size:
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_id"] = chunk_idx
                            chunks.append(Document(
                                content=current_text,
                                metadata=chunk_metadata,
                                doc_id=f"{metadata.get('doc_id', 'doc')}_{chunk_idx}"
                            ))
                            chunk_idx += 1
                            
                            # Keep overlap
                            overlap_words = int(len(current_words) * self.chunk_overlap / self.chunk_size)
                            current_words = current_words[-overlap_words:]
                    
                    # Remaining words
                    if current_words:
                        current_chunk = " ".join(current_words) + self.separator
                else:
                    current_chunk = section + self.separator
        
        # Add last chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_idx
            chunks.append(Document(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                doc_id=f"{metadata.get('doc_id', 'doc')}_{chunk_idx}"
            ))
        
        logger.debug(f"Chunked document into {len(chunks)} pieces")
        return chunks
    
    def chunk_documents(self, documents: List[Tuple[str, Dict]]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of (content, metadata) tuples
            
        Returns:
            List of Document chunks
        """
        all_chunks = []
        
        for content, metadata in documents:
            chunks = self.chunk_document(content, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks


class MedicalRAG:
    """
    Complete Medical RAG system.
    Combines retrieval, re-ranking, and generation for medical Q&A.
    """
    
    def __init__(
        self,
        vector_store_type: str = "faiss",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_hybrid_search: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        enable_query_enhancement: bool = True,
        **kwargs
    ):
        """
        Initialize Medical RAG system.
        
        Args:
            vector_store_type: Type of vector store ("faiss" or "chromadb")
            embedding_model: Embedding model name
            use_reranker: Whether to use re-ranker
            reranker_model: Re-ranker model name
            use_hybrid_search: Whether to use hybrid retrieval
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap
            top_k: Number of documents to retrieve
            enable_query_enhancement: Enable query processing
            **kwargs: Additional arguments
        """
        logger.info("Initializing Medical RAG system")
        
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.enable_query_enhancement = enable_query_enhancement
        
        # Initialize vector store
        self.vector_store = create_vector_store(
            store_type=vector_store_type,
            embedding_model=embedding_model,
            **kwargs
        )
        
        # Initialize retriever
        if use_hybrid_search:
            self.retriever = ContextualRetriever(
                vector_store=self.vector_store,
                use_sparse=True
            )
        else:
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                use_sparse=False
            )
        
        # Initialize re-ranker
        if use_reranker:
            self.reranker = create_reranker(
                reranker_type="cross_encoder",
                model_name=reranker_model
            )
        else:
            self.reranker = None
        
        # Initialize query enhancer
        if enable_query_enhancement:
            self.query_enhancer = MedicalQueryEnhancer()
        else:
            self.query_enhancer = None
        
        # Initialize document chunker
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info("Medical RAG system initialized successfully")
    
    def add_documents(
        self,
        documents: List[Tuple[str, Dict]],
        chunk: bool = True
    ) -> None:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of (content, metadata) tuples
            chunk: Whether to chunk documents
        """
        if chunk:
            doc_objects = self.chunker.chunk_documents(documents)
        else:
            doc_objects = [
                Document(content=content, metadata=metadata)
                for content, metadata in documents
            ]
        
        # Add to retriever
        self.retriever.add_documents(doc_objects)
        
        logger.info(f"Added {len(documents)} documents ({len(doc_objects)} chunks) to RAG system")
    
    def load_documents_from_directory(
        self,
        directory: Path,
        file_pattern: str = "*.txt"
    ) -> None:
        """
        Load documents from a directory.
        
        Args:
            directory: Directory path
            file_pattern: File pattern to match
        """
        directory = Path(directory)
        files = list(directory.glob(file_pattern))
        
        documents = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "source": str(file_path),
                "filename": file_path.name
            }
            
            documents.append((content, metadata))
        
        self.add_documents(documents)
        logger.info(f"Loaded {len(files)} documents from {directory}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        conversation_history: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            use_reranking: Whether to use re-ranking
            conversation_history: Previous conversation turns
            
        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        if use_reranking is None:
            use_reranking = self.use_reranker
        
        # Enhance query
        enhanced_query = query
        if self.query_enhancer:
            enhanced = self.query_enhancer.enhance(query)
            enhanced_query = enhanced.get("rewritten", query) or query
            logger.debug(f"Enhanced query: {query} -> {enhanced_query}")
        
        # Retrieve documents
        if conversation_history and isinstance(self.retriever, ContextualRetriever):
            results = self.retriever.search_with_context(
                query=enhanced_query,
                conversation_history=conversation_history,
                top_k=top_k * 2 if use_reranking else top_k
            )
        else:
            results = self.retriever.search(
                query=enhanced_query,
                top_k=top_k * 2 if use_reranking else top_k
            )
        
        # Re-rank if enabled
        if use_reranking and self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        
        # Convert to dictionaries
        retrieved_docs = [
            {
                "content": result.document.content,
                "metadata": result.document.metadata,
                "score": result.score,
                "rank": result.rank,
                "retrieval_method": result.retrieval_method
            }
            for result in results[:top_k]
        ]
        
        return retrieved_docs
    
    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        conversation_history: Optional[List[str]] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question to the RAG system.
        Note: This returns retrieved context. For answer generation,
        integrate with LLM module.
        
        Args:
            query: Question to ask
            top_k: Number of source documents
            conversation_history: Previous conversation
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(
            query=query,
            top_k=top_k,
            conversation_history=conversation_history
        )
        
        # Prepare response
        response = {
            "query": query,
            "num_sources": len(retrieved_docs),
            "context": "\n\n".join([doc["content"] for doc in retrieved_docs]),
            "retrieval_info": {
                "top_k": len(retrieved_docs),
                "methods": list(set([doc["retrieval_method"] for doc in retrieved_docs]))
            }
        }
        
        if return_sources:
            response["sources"] = retrieved_docs
        
        # Add query enhancement info
        if self.query_enhancer:
            enhanced = self.query_enhancer.enhance(query)
            response["query_info"] = {
                "original": enhanced["original"],
                "enhanced": enhanced.get("rewritten") or enhanced.get("processed"),
                "keywords": enhanced["keywords"],
                "medical_terms": enhanced["medical_terms"]
            }
        
        return response
    
    def save(self, save_path: Path) -> None:
        """
        Save RAG system state.
        
        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(save_path / "vector_store")
        
        # Save configuration
        config = {
            "top_k": self.top_k,
            "use_reranker": self.use_reranker,
            "enable_query_enhancement": self.enable_query_enhancement
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"RAG system saved to {save_path}")
    
    def load(self, load_path: Path) -> None:
        """
        Load RAG system state.
        
        Args:
            load_path: Directory to load from
        """
        load_path = Path(load_path)
        
        # Load vector store
        self.vector_store.load(load_path / "vector_store")
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            config = json.load(f)
        
        self.top_k = config.get("top_k", self.top_k)
        self.use_reranker = config.get("use_reranker", self.use_reranker)
        
        logger.info(f"RAG system loaded from {load_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        stats = {
            "retriever": self.retriever.get_statistics(),
            "use_reranker": self.use_reranker,
            "top_k": self.top_k
        }
        
        return stats
