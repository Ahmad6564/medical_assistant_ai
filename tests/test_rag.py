"""
Unit tests for RAG (Retrieval Augmented Generation) components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.mark.unit
class TestVectorStores:
    """Tests for vector store implementations."""
    
    @pytest.mark.skipif(not pytest.importorskip("faiss"), reason="FAISS not installed")
    def test_faiss_store_initialization(self):
        """Test FAISS vector store can be initialized."""
        from src.rag import FAISSVectorStore
        
        dimension = 768
        store = FAISSVectorStore(dimension=dimension)
        
        assert store is not None
        assert store.dimension == dimension
    
    @pytest.mark.skipif(not pytest.importorskip("faiss"), reason="FAISS not installed")
    def test_faiss_add_documents(self, mock_embedding_model):
        """Test adding documents to FAISS store."""
        from src.rag import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=768)
        
        documents = ["Document 1", "Document 2", "Document 3"]
        embeddings = np.random.randn(3, 768).astype(np.float32)
        
        store.add_documents(documents, embeddings)
        
        assert store.count() == 3
    
    @pytest.mark.skipif(not pytest.importorskip("faiss"), reason="FAISS not installed")
    def test_faiss_search(self, mock_embedding_model):
        """Test searching in FAISS store."""
        from src.rag import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=768)
        
        documents = ["Document 1", "Document 2", "Document 3"]
        embeddings = np.random.randn(3, 768).astype(np.float32)
        store.add_documents(documents, embeddings)
        
        query_embedding = np.random.randn(1, 768).astype(np.float32)
        results, distances = store.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert len(distances) == 2
    
    @pytest.mark.skipif(not pytest.importorskip("chromadb"), reason="ChromaDB not installed")
    def test_chroma_store_initialization(self):
        """Test ChromaDB vector store can be initialized."""
        from src.rag import ChromaVectorStore
        
        store = ChromaVectorStore(collection_name="test_collection")
        
        assert store is not None
    
    @pytest.mark.skipif(not pytest.importorskip("chromadb"), reason="ChromaDB not installed")
    def test_chroma_add_documents(self):
        """Test adding documents to ChromaDB store."""
        from src.rag import ChromaVectorStore
        
        store = ChromaVectorStore(collection_name="test_collection")
        
        documents = ["Document 1", "Document 2", "Document 3"]
        embeddings = [[0.1] * 768 for _ in range(3)]
        
        store.add_documents(documents, embeddings)
        
        assert store.count() >= 3


@pytest.mark.unit
class TestRetriever:
    """Tests for document retriever."""
    
    def test_retriever_initialization(self, mock_embedding_model):
        """Test retriever can be initialized."""
        from src.rag import DocumentRetriever
        
        retriever = DocumentRetriever(
            embedding_model=mock_embedding_model,
            vector_store_type="faiss"
        )
        
        assert retriever is not None
    
    def test_retrieve_documents(self, mock_embedding_model, sample_documents):
        """Test document retrieval."""
        from src.rag import DocumentRetriever
        
        # Skip if FAISS not available
        pytest.importorskip("faiss")
        
        retriever = DocumentRetriever(
            embedding_model=mock_embedding_model,
            vector_store_type="faiss"
        )
        
        # Add documents
        retriever.add_documents(sample_documents)
        
        # Retrieve
        query = "What is diabetes?"
        results = retriever.retrieve(query, k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, str) for doc in results)


@pytest.mark.unit
class TestReranker:
    """Tests for result reranking."""
    
    def test_reranker_initialization(self):
        """Test reranker can be initialized."""
        from src.rag import Reranker
        
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        assert reranker is not None
    
    def test_rerank_documents(self, sample_documents, sample_query):
        """Test document reranking."""
        from src.rag import Reranker
        
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Get subset of documents
        documents = sample_documents[:3]
        
        # Rerank
        reranked = reranker.rerank(sample_query, documents, top_k=2)
        
        assert len(reranked) <= 2
        assert all(isinstance(doc, str) for doc in reranked)


@pytest.mark.unit
class TestHybridSearch:
    """Tests for hybrid search combining dense and sparse retrieval."""
    
    def test_hybrid_search_initialization(self, mock_embedding_model):
        """Test hybrid search can be initialized."""
        from src.rag import HybridSearch
        
        search = HybridSearch(
            embedding_model=mock_embedding_model,
            use_bm25=True
        )
        
        assert search is not None
    
    def test_hybrid_search_retrieval(self, mock_embedding_model, sample_documents, sample_query):
        """Test hybrid search retrieval."""
        from src.rag import HybridSearch
        
        # Skip if required dependencies not available
        pytest.importorskip("faiss")
        pytest.importorskip("rank_bm25")
        
        search = HybridSearch(
            embedding_model=mock_embedding_model,
            use_bm25=True
        )
        
        # Add documents
        search.add_documents(sample_documents)
        
        # Search
        results = search.search(sample_query, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(doc, str) for doc in results)


@pytest.mark.unit
class TestDocumentProcessor:
    """Tests for document processing utilities."""
    
    def test_chunk_text(self):
        """Test text chunking."""
        from src.rag import chunk_text
        
        text = "This is a long document. " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        from src.rag import extract_metadata
        
        text = "# Title\nAuthor: John Doe\nDate: 2024-01-01\n\nContent here."
        metadata = extract_metadata(text)
        
        assert isinstance(metadata, dict)
    
    def test_clean_text(self):
        """Test text cleaning."""
        from src.rag import clean_text
        
        text = "This  has   extra   spaces.\n\n\nAnd newlines."
        cleaned = clean_text(text)
        
        assert "  " not in cleaned
        assert "\n\n\n" not in cleaned


@pytest.mark.integration
@pytest.mark.slow
def test_rag_pipeline(mock_embedding_model, sample_documents, sample_query):
    """Test complete RAG pipeline."""
    from src.rag import RAGPipeline
    
    # Skip if dependencies not available
    pytest.importorskip("faiss")
    
    pipeline = RAGPipeline(
        embedding_model=mock_embedding_model,
        vector_store_type="faiss",
        use_reranking=False
    )
    
    # Add documents
    pipeline.add_documents(sample_documents)
    
    # Query
    results = pipeline.query(sample_query, k=3)
    
    assert "documents" in results
    assert len(results["documents"]) <= 3


@pytest.mark.unit
def test_load_medical_literature(test_data_dir):
    """Test loading medical literature."""
    from src.rag import load_medical_literature
    
    # Create test file
    test_file = test_data_dir / "test_literature.txt"
    test_file.write_text("Test medical document content.")
    
    documents = load_medical_literature(test_data_dir)
    
    assert len(documents) > 0
    assert all(isinstance(doc, str) for doc in documents)


@pytest.mark.unit
def test_citation_extraction():
    """Test extracting citations from documents."""
    from src.rag import extract_citations
    
    text = "According to [1], diabetes is common. See also [2, 3]."
    citations = extract_citations(text)
    
    assert len(citations) > 0
    assert all(isinstance(c, str) for c in citations)


@pytest.mark.unit  
def test_document_similarity():
    """Test computing document similarity."""
    from src.rag import compute_similarity
    
    doc1 = "Diabetes is a metabolic disorder."
    doc2 = "Diabetes affects blood sugar levels."
    doc3 = "The weather is sunny today."
    
    sim_12 = compute_similarity(doc1, doc2)
    sim_13 = compute_similarity(doc1, doc3)
    
    assert sim_12 > sim_13  # More similar documents have higher score
