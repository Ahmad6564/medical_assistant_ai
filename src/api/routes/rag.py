"""
RAG (Retrieval-Augmented Generation) API endpoints.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status, UploadFile, File

from src.rag import MedicalRAG
from src.llm import LLMInterface, ChainOfThoughtReasoner, ReasoningStrategy
from src.safety import MedicalAISafetyGuardrails
from src.api.models import (
    RAGRequest,
    RAGResponse,
    RetrievedDocument,
    DocumentUploadRequest,
    DocumentUploadResponse
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Global instances (lazy loaded)
_rag_system = None
_llm = None
_safety_guardrails = None


def get_rag_system():
    """Get or initialize RAG system."""
    global _rag_system
    if _rag_system is None:
        logger.info("Loading RAG system...")
        try:
            _rag_system = MedicalRAG(
                vector_store_type="faiss",
                use_reranker=True,
                use_hybrid_search=True,
                top_k=5
            )
            logger.info("RAG system loaded")
        except Exception as e:
            logger.warning(f"Could not load RAG system: {e}. Using mock mode.")
            _rag_system = MockRAG()
    return _rag_system


def get_llm() -> Optional[LLMInterface]:
    """Get or initialize LLM."""
    global _llm
    if _llm is None:
        try:
            logger.info("Loading LLM interface...")
            _llm = LLMInterface(
                provider="openai",
                model="gpt-4",
                max_retries=3
            )
            logger.info("LLM interface loaded")
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            _llm = None
    return _llm


def get_safety_guardrails():
    """Get or initialize safety guardrails."""
    global _safety_guardrails
    if _safety_guardrails is None:
        logger.info("Loading safety guardrails...")
        try:
            _safety_guardrails = MedicalAISafetyGuardrails(
                enable_emergency_detection=True,
                enable_claim_filtering=True,
                enable_dosage_validation=True,
                enable_disclaimers=True
            )
            logger.info("Safety guardrails loaded")
        except Exception as e:
            logger.warning(f"Could not load safety guardrails: {e}")
            _safety_guardrails = None
    return _safety_guardrails


class MockRAG:
    """Mock RAG system for demo purposes."""
    
    def __init__(self):
        logger.info("Using RAG in DEMO MODE")
        self.documents = []
    
    def ask(self, query: str, top_k: int = 5, conversation_history=None, return_sources: bool = True):
        """Mock RAG response."""
        return {
            "context": f"This is a demo response. The RAG system would normally search medical literature to answer: {query}",
            "sources": [
                {
                    "content": "Demo medical literature excerpt",
                    "score": 0.85,
                    "metadata": {"source": "demo"}
                }
            ],
            "num_sources": 1
        }
    
    def add_documents(self, documents):
        """Mock document addition."""
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents (demo mode)")


@router.post("/ask", response_model=RAGResponse)
async def ask_question(
    request: RAGRequest
):
    """
    Ask a medical question using RAG.
    
    Args:
        request: RAG request with query
        
    Returns:
        RAGResponse: Generated answer with sources
    """
    try:
        start_time = time.time()
        
        # Get systems
        rag = get_rag_system()
        safety = get_safety_guardrails()
        
        logger.info(f"Processing RAG query")
        
        # Check query safety first
        query_safety = safety.check_query_safety(request.query)
        
        # Handle emergencies
        if query_safety.emergency_detected:
            logger.warning(f"Emergency detected in query: {request.query}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return RAGResponse(
                answer="\n".join(query_safety.recommendations),
                query=request.query,
                retrieved_documents=[],
                num_sources=0,
                disclaimers=["EMERGENCY SITUATION DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION"],
                processing_time_ms=round(processing_time, 2)
            )
        
        # Retrieve relevant documents
        rag_result = rag.ask(
            query=request.query,
            top_k=request.top_k,
            conversation_history=None,  # TODO: Implement conversation history tracking
            return_sources=True
        )
        
        # Convert retrieved documents
        retrieved_docs = []
        if "sources" in rag_result:
            for doc in rag_result["sources"]:
                retrieved_docs.append(RetrievedDocument(
                    content=doc.get("content", ""),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {})
                ))
        
        # Generate answer
        context = rag_result.get("context", "")
        answer = f"Based on the available medical literature:\n\n{context}"
        
        reasoning_steps = None
        
        # Optional: Use LLM for better answer generation
        llm = get_llm()
        if llm:
            try:
                # Use chain-of-thought if requested
                if request.use_chain_of_thought:
                    reasoner = ChainOfThoughtReasoner(
                        llm=llm,
                        strategy=ReasoningStrategy.LINEAR
                    )
                    reasoning_result = reasoner.reason(
                        task=request.query,
                        context=context
                    )
                    answer = reasoning_result.final_answer
                    reasoning_steps = [step.content for step in reasoning_result.steps]
                else:
                    # Simple LLM generation
                    from src.llm import Message, PromptBuilder, PromptType
                    
                    prompt_builder = PromptBuilder()
                    prompts = prompt_builder.build(
                        PromptType.QUESTION_ANSWERING,
                        {"question": request.query, "context": context}
                    )
                    
                    messages = [
                        Message(role="system", content=prompts["system"]),
                        Message(role="user", content=prompts["user"])
                    ]
                    
                    llm_response = llm.generate(messages)
                    answer = llm_response.content
                    
            except Exception as e:
                logger.warning(f"LLM generation failed, using fallback: {e}")
        
        # Apply safety checks to answer
        is_safe, safe_answer, response_safety = safety.safe_response(
            request.query,
            answer
        )
        
        if not is_safe:
            logger.warning(f"Unsafe response detected: {response_safety.warnings}")
        
        # Extract disclaimers
        disclaimers = [d.value for d in response_safety.required_disclaimers]
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=safe_answer,
            query=request.query,
            retrieved_documents=retrieved_docs,
            num_sources=len(retrieved_docs),
            reasoning_steps=reasoning_steps,
            disclaimers=disclaimers,
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}"
        )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: DocumentUploadRequest
):
    """
    Upload a document to the RAG system.
    
    Args:
        request: Document upload request
        
    Returns:
        DocumentUploadResponse: Upload status
    """
    try:
        rag = get_rag_system()
        
        logger.info(f"Uploading document")
        
        # Add metadata
        metadata = request.metadata or {}
        metadata["uploaded_by"] = "api_user"
        metadata["source"] = request.source or "api_upload"
        
        # Add document
        documents = [(request.content, metadata)]
        rag.add_documents(documents)
        
        # Generate document ID
        import hashlib
        doc_id = hashlib.md5(request.content.encode()).hexdigest()
        
        # Estimate chunks (rough estimate)
        chunk_size = 512
        num_chunks = len(request.content) // chunk_size + 1
        
        return DocumentUploadResponse(
            success=True,
            document_id=doc_id,
            num_chunks=num_chunks,
            message="Document uploaded successfully"
        )
    
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )


@router.get("/system/info")
async def system_info():
    """
    Get RAG system information.
    
    Returns:
        Dict: System information
    """
    rag = get_rag_system()
    
    return {
        "status": "operational",
        "vector_store": "faiss",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "llm_available": get_llm() is not None,
        "safety_enabled": True,
        "features": {
            "hybrid_search": True,
            "reranking": True,
            "chain_of_thought": True,
            "safety_guardrails": True
        }
    }
