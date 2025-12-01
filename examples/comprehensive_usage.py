"""
Example usage of the Medical AI Assistant RAG, LLM, and Safety systems.
Demonstrates integrated usage of all components.
"""

from src.rag import MedicalRAG
from src.llm import LLMInterface, PromptBuilder, PromptType, ChainOfThoughtReasoner, ReasoningStrategy
from src.safety import MedicalAISafetyGuardrails
from pathlib import Path


def example_rag_basic():
    """Basic RAG system usage."""
    print("=" * 80)
    print("EXAMPLE 1: Basic RAG System")
    print("=" * 80)
    
    # Initialize RAG system
    rag = MedicalRAG(
        vector_store_type="faiss",
        use_reranker=True,
        use_hybrid_search=True,
        top_k=5
    )
    
    # Add some example documents
    documents = [
        ("Hypertension, also known as high blood pressure, is a chronic medical condition. "
         "First-line treatments include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers.",
         {"source": "hypertension_guide.pdf", "topic": "cardiology"}),
        
        ("Diabetes mellitus type 2 is managed through lifestyle modifications and medications. "
         "Metformin is the first-line medication. Regular monitoring of HbA1c is essential.",
         {"source": "diabetes_management.pdf", "topic": "endocrinology"}),
        
        ("Acute coronary syndrome requires immediate medical attention. Symptoms include "
         "chest pain, shortness of breath, and arm pain. Call 911 immediately.",
         {"source": "emergency_guide.pdf", "topic": "emergency_medicine"})
    ]
    
    rag.add_documents(documents)
    
    # Ask a question
    query = "What are the first-line treatments for hypertension?"
    response = rag.ask(query)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved Context:\n{response['context'][:200]}...")
    print(f"\nNumber of Sources: {response['num_sources']}")
    print(f"\nQuery Enhancement: {response.get('query_info', {})}")


def example_llm_basic():
    """Basic LLM usage with prompts."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: LLM with Medical Prompts")
    print("=" * 80)
    
    # Initialize LLM (requires API key in environment)
    try:
        llm = LLMInterface(
            provider="openai",
            model="gpt-4",
            max_retries=3
        )
        
        # Use prompt builder
        prompt_builder = PromptBuilder()
        
        # Build a Q&A prompt
        prompts = prompt_builder.build(
            prompt_type=PromptType.QUESTION_ANSWERING,
            variables={
                "question": "What causes hypertension?",
                "context": "Hypertension can be caused by various factors including genetics, diet, stress, and kidney disease."
            }
        )
        
        # Generate response
        from src.llm import Message
        messages = [
            Message(role="system", content=prompts["system"]),
            Message(role="user", content=prompts["user"])
        ]
        
        print("\nSystem Prompt (truncated):")
        print(prompts["system"][:200] + "...")
        
        print("\nUser Prompt:")
        print(prompts["user"])
        
        print("\n[LLM Response would be generated here - requires API key]")
        
    except Exception as e:
        print(f"\nLLM initialization skipped (API key needed): {e}")


def example_chain_of_thought():
    """Chain-of-thought reasoning example."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Chain-of-Thought Clinical Reasoning")
    print("=" * 80)
    
    try:
        llm = LLMInterface(provider="openai", model="gpt-4")
        
        # Initialize reasoner
        reasoner = ChainOfThoughtReasoner(
            llm=llm,
            strategy=ReasoningStrategy.LINEAR,
            temperature=0.3
        )
        
        # Clinical reasoning task
        task = "Patient presents with chest pain, shortness of breath, and arm pain. Determine differential diagnosis."
        context = "Patient is 55-year-old male with history of smoking and hypertension."
        
        print(f"\nTask: {task}")
        print(f"Context: {context}")
        print("\n[Chain-of-thought reasoning would be performed here - requires API key]")
        
    except Exception as e:
        print(f"\nChain-of-thought example skipped (API key needed): {e}")


def example_safety_guardrails():
    """Safety guardrails demonstration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: AI Safety Guardrails")
    print("=" * 80)
    
    # Initialize safety system
    guardrails = MedicalAISafetyGuardrails(
        enable_emergency_detection=True,
        enable_claim_filtering=True,
        enable_dosage_validation=True,
        enable_disclaimers=True,
        strict_mode=True
    )
    
    # Example 1: Emergency detection
    emergency_query = "I'm having severe chest pain and can't breathe"
    query_safety = guardrails.check_query_safety(emergency_query)
    
    print("\n--- Emergency Detection ---")
    print(f"Query: '{emergency_query}'")
    print(f"Safety Level: {query_safety.safety_level.value}")
    print(f"Emergency Detected: {query_safety.emergency_detected}")
    print(f"Warnings: {query_safety.warnings}")
    
    # Example 2: Prohibited claims filtering
    dangerous_response = "You can cure cancer by taking essential oils and skipping chemotherapy."
    response_safety = guardrails.check_response_safety(dangerous_response)
    
    print("\n--- Claim Filtering ---")
    print(f"Response: '{dangerous_response}'")
    print(f"Safety Level: {response_safety.safety_level.value}")
    print(f"Is Safe: {response_safety.is_safe}")
    print(f"Prohibited Claims: {response_safety.prohibited_claims}")
    
    # Example 3: Safe response with disclaimers
    safe_query = "What are treatments for high blood pressure?"
    safe_response = "Treatments for hypertension include lifestyle modifications and medications like ACE inhibitors."
    
    is_safe, modified_response, safety_result = guardrails.safe_response(safe_query, safe_response)
    
    print("\n--- Complete Safety Pipeline ---")
    print(f"Query: '{safe_query}'")
    print(f"Original Response: '{safe_response}'")
    print(f"\nIs Safe: {is_safe}")
    print(f"Required Disclaimers: {[d.value for d in safety_result.required_disclaimers]}")
    print(f"\nModified Response (with disclaimers):\n{modified_response[:500]}...")


def example_integrated_system():
    """Complete integrated system example."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Integrated RAG + LLM + Safety System")
    print("=" * 80)
    
    # Initialize all components
    rag = MedicalRAG(use_reranker=True, top_k=3)
    guardrails = MedicalAISafetyGuardrails()
    
    # Add medical literature
    documents = [
        ("Metformin is the first-line medication for type 2 diabetes. "
         "Starting dose is typically 500mg twice daily with meals. "
         "Maximum dose is 2550mg per day divided into 2-3 doses.",
         {"source": "diabetes_guidelines.pdf"})
    ]
    rag.add_documents(documents)
    
    # User query
    user_query = "What is the typical starting dose for metformin?"
    
    print(f"\nUser Query: '{user_query}'")
    
    # Step 1: Check query safety
    query_safety = guardrails.check_query_safety(user_query)
    print(f"\nQuery Safety Check: {query_safety.safety_level.value}")
    
    if query_safety.emergency_detected:
        print("EMERGENCY DETECTED - Bypassing normal flow")
        print("\n".join(query_safety.recommendations))
        return
    
    # Step 2: Retrieve relevant context
    rag_response = rag.ask(user_query)
    context = rag_response['context']
    
    print(f"\nRetrieved Context: {context[:150]}...")
    
    # Step 3: Generate LLM response (simulated)
    simulated_llm_response = (
        "Based on the guidelines, metformin for type 2 diabetes typically starts at "
        "500mg twice daily with meals. The dose can be gradually increased based on "
        "blood glucose response and tolerability, up to a maximum of 2550mg daily."
    )
    
    print(f"\nLLM Response: {simulated_llm_response}")
    
    # Step 4: Apply safety guardrails
    is_safe, final_response, response_safety = guardrails.safe_response(
        user_query,
        simulated_llm_response
    )
    
    print(f"\nResponse Safety Check: {response_safety.safety_level.value}")
    print(f"Is Safe: {is_safe}")
    print(f"Required Disclaimers: {[d.value for d in response_safety.required_disclaimers]}")
    
    print(f"\nFinal Response to User:\n{final_response[:600]}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MEDICAL AI ASSISTANT - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("RAG Basic Usage", example_rag_basic),
        ("LLM with Prompts", example_llm_basic),
        ("Chain-of-Thought Reasoning", example_chain_of_thought),
        ("Safety Guardrails", example_safety_guardrails),
        ("Integrated System", example_integrated_system)
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nNote: Some examples require API keys to be set in environment variables:")
    print("  - OPENAI_API_KEY for OpenAI")
    print("  - ANTHROPIC_API_KEY for Anthropic")
    print("\nThe system is fully implemented and ready for use!")


if __name__ == "__main__":
    main()
