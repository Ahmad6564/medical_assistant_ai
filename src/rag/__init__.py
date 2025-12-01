"""
RAG (Retrieval-Augmented Generation) module for medical Q&A.
"""

from .vector_store import (
    VectorStore,
    FAISSVectorStore,
    ChromaDBVectorStore,
    Document,
    create_vector_store
)
from .retriever import (
    SparseRetriever,
    HybridRetriever,
    ContextualRetriever,
    RetrievalResult
)
from .reranker import (
    Reranker,
    CrossEncoderReranker,
    MMRReranker,
    EnsembleReranker,
    create_reranker
)
from .query_processing import (
    QueryProcessor,
    QueryRewriter,
    MedicalQueryEnhancer,
    ProcessedQuery
)
from .medical_rag import (
    MedicalRAG,
    DocumentChunker,
    RAGResponse
)

# Aliases for test compatibility
ChromaVectorStore = ChromaDBVectorStore
DocumentRetriever = HybridRetriever
HybridSearch = HybridRetriever
RAGPipeline = MedicalRAG

# Utility functions for test compatibility
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    """Chunk text into overlapping segments."""
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk(text)

def extract_metadata(text: str):
    """Extract metadata from text."""
    return {
        "length": len(text),
        "num_words": len(text.split()),
        "num_sentences": text.count('.') + text.count('!') + text.count('?')
    }

def clean_text(text: str):
    """Clean and normalize text."""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep medical terms
    text = text.strip()
    return text

def load_medical_literature(path: str):
    """Load medical literature from path."""
    from pathlib import Path
    import json
    
    path = Path(path)
    documents = []
    
    if path.is_file():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
    elif path.is_dir():
        for file in path.glob("**/*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
    
    return documents

def extract_citations(text: str):
    """Extract citations from text."""
    import re
    # Simple citation pattern matching
    citations = re.findall(r'\[(\d+)\]|\(([A-Za-z]+\s+et\s+al\.,?\s+\d{4})\)', text)
    return [c[0] or c[1] for c in citations if c[0] or c[1]]

def compute_similarity(text1: str, text2: str):
    """Compute similarity between two texts."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return float(util.cos_sim(emb1, emb2)[0][0])

__all__ = [
    # Vector stores
    "VectorStore",
    "FAISSVectorStore",
    "ChromaDBVectorStore",
    "ChromaVectorStore",
    "Document",
    "create_vector_store",
    
    # Retrievers
    "SparseRetriever",
    "HybridRetriever",
    "ContextualRetriever",
    "DocumentRetriever",
    "HybridSearch",
    "RetrievalResult",
    
    # Re-rankers
    "Reranker",
    "CrossEncoderReranker",
    "MMRReranker",
    "EnsembleReranker",
    "create_reranker",
    
    # Query processing
    "QueryProcessor",
    "QueryRewriter",
    "MedicalQueryEnhancer",
    "ProcessedQuery",
    
    # Main RAG system
    "MedicalRAG",
    "RAGPipeline",
    "DocumentChunker",
    "RAGResponse",
    
    # Utility functions
    "chunk_text",
    "extract_metadata",
    "clean_text",
    "load_medical_literature",
    "extract_citations",
    "compute_similarity"
]
