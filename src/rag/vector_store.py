"""
Vector store implementation for medical document embeddings.
Supports FAISS and ChromaDB backends.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Document with metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """Base class for vector stores."""
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        raise NotImplementedError
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        raise NotImplementedError
    
    def save(self, path: Path) -> None:
        """Save the vector store."""
        raise NotImplementedError
    
    def load(self, path: Path) -> None:
        """Load the vector store."""
        raise NotImplementedError


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        dimension: int = 768,
        index_type: str = "flat"  # "flat", "ivf", "hnsw"
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model
            dimension: Embedding dimension
            index_type: Type of FAISS index
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.documents: List[Document] = []
        self.is_trained = False
        
        logger.info(f"Initialized FAISS vector store with {index_type} index")
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using sentence transformer.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype('float32')
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        # Extract texts for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self._embed_texts(texts)
        
        # Store embeddings in documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.is_trained:
            logger.info("Training IVF index")
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to vector store (total: {len(self.documents)})")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Metadata filters (applied post-retrieval)
            
        Returns:
            List of (document, score) tuples
        """
        if len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        # Embed query
        query_embedding = self._embed_texts([query])[0:1]
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k * 2, len(self.documents)))
        
        # Convert to documents with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply metadata filtering
                if filter_dict:
                    if not all(doc.metadata.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                # Convert L2 distance to similarity score (0-1 range)
                score = 1.0 / (1.0 + float(dist))
                results.append((doc, score))
        
        # Return top_k results
        return results[:top_k]
    
    def save(self, path: Path) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        
        # Restore attributes
        self.embedding_model_name = metadata["embedding_model"]
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.is_trained = metadata["is_trained"]
        
        logger.info(f"Vector store loaded from {path} ({len(self.documents)} documents)")


class ChromaDBVectorStore(VectorStore):
    """ChromaDB-based vector store with metadata filtering."""
    
    def __init__(
        self,
        collection_name: str = "medical_documents",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        persist_directory: Optional[Path] = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to persist data
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedding_model}
        )
        
        logger.info(f"Initialized ChromaDB vector store: {collection_name}")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence transformer."""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to ChromaDB.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        # Prepare data
        ids = [doc.doc_id or f"doc_{i}" for i, doc in enumerate(documents)]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self._embed_texts(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        # Embed query
        query_embedding = self._embed_texts([query])[0]
        
        # Prepare where clause for filtering
        where_clause = filter_dict if filter_dict else None
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # Convert to Document objects
        documents = []
        if results["documents"] and results["documents"][0]:
            for doc_text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                doc = Document(
                    content=doc_text,
                    metadata=metadata
                )
                # Convert distance to similarity score
                score = 1.0 / (1.0 + distance)
                documents.append((doc, score))
        
        return documents
    
    def save(self, path: Path) -> None:
        """ChromaDB auto-persists if persist_directory is set."""
        logger.info("ChromaDB auto-persists data")
    
    def load(self, path: Path) -> None:
        """ChromaDB auto-loads from persist_directory."""
        logger.info("ChromaDB auto-loads data from persist directory")


def create_vector_store(
    store_type: str = "faiss",
    **kwargs
) -> VectorStore:
    """
    Factory function to create vector store.
    
    Args:
        store_type: Type of vector store ("faiss" or "chromadb")
        **kwargs: Arguments for the vector store
        
    Returns:
        VectorStore instance
    """
    if store_type.lower() == "faiss":
        return FAISSVectorStore(**kwargs)
    elif store_type.lower() == "chromadb":
        return ChromaDBVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
