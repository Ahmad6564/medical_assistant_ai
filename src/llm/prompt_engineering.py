"""
Prompt engineering templates and utilities for medical AI.
Includes specialized prompts for different medical tasks.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of medical prompts."""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    EXPLANATION = "explanation"
    SUMMARY = "summary"
    QUESTION_ANSWERING = "qa"
    CLINICAL_REASONING = "clinical_reasoning"
    ADVERSE_EVENTS = "adverse_events"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    system_prompt: str
    user_template: str
    variables: List[str]
    examples: Optional[List[Dict[str, str]]] = None


class MedicalPromptTemplates:
    """Collection of medical prompt templates."""
    
    # Base system prompt for all medical tasks
    BASE_SYSTEM = """You are an expert medical AI assistant with comprehensive knowledge of medicine, healthcare, and clinical practice. Your role is to provide accurate, evidence-based information while maintaining the highest standards of medical ethics and patient safety.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety and well-being
2. Provide evidence-based information from reputable medical sources
3. Acknowledge limitations and uncertainty when appropriate
4. Never provide definitive diagnoses - only suggest possibilities
5. Recommend consulting qualified healthcare professionals for medical advice
6. Use clear, precise medical terminology while remaining accessible
7. Consider differential diagnoses and multiple perspectives
8. Be mindful of contraindications and potential adverse effects

Remember: You are an educational tool, not a replacement for professional medical judgment."""
    
    TEMPLATES = {
        PromptType.QUESTION_ANSWERING: PromptTemplate(
            name="Medical Q&A",
            system_prompt=BASE_SYSTEM + """\n
For this task, you will answer medical questions using the provided context from medical literature. 
- Base your answers primarily on the provided context
- Cite sources when possible
- If the context doesn't contain sufficient information, clearly state this
- Provide comprehensive but concise answers
- Include relevant warnings or precautions""",
            user_template="""Question: {question}

Context from medical literature:
{context}

Please provide a comprehensive answer based on the context above. If the context is insufficient, indicate what additional information would be needed.""",
            variables=["question", "context"]
        ),
        
        PromptType.CLINICAL_REASONING: PromptTemplate(
            name="Clinical Reasoning",
            system_prompt=BASE_SYSTEM + """\n
For this task, apply systematic clinical reasoning to analyze medical scenarios.
- Use a structured approach (e.g., problem representation, differential diagnosis, testing strategy)
- Consider patient context, risk factors, and clinical presentation
- Think through the diagnostic process step-by-step
- Weigh probabilities and prioritize based on severity and likelihood
- Recommend appropriate next steps in evaluation or management""",
            user_template="""Clinical Scenario:
{scenario}

Additional Context:
{context}

Please provide a structured clinical analysis including:
1. Problem representation
2. Differential diagnosis with reasoning
3. Recommended diagnostic approach
4. Initial management considerations
5. Red flags or urgent concerns""",
            variables=["scenario", "context"]
        ),
        
        PromptType.DIAGNOSIS: PromptTemplate(
            name="Differential Diagnosis",
            system_prompt=BASE_SYSTEM + """\n
For this task, generate differential diagnoses based on clinical presentations.
- Consider both common and serious/life-threatening conditions
- Organize by likelihood and severity
- Explain reasoning for each diagnosis
- Note key distinguishing features
- Suggest relevant diagnostic tests""",
            user_template="""Patient Presentation:
{presentation}

Medical Context:
{context}

Please provide:
1. Top differential diagnoses (most to least likely)
2. Reasoning for each diagnosis
3. Key clinical features supporting or refuting each diagnosis
4. Recommended diagnostic workup
5. Urgent/emergent considerations""",
            variables=["presentation", "context"]
        ),
        
        PromptType.TREATMENT: PromptTemplate(
            name="Treatment Recommendations",
            system_prompt=BASE_SYSTEM + """\n
For this task, discuss treatment options for medical conditions.
- Present evidence-based treatment approaches
- Include first-line, second-line, and alternative therapies
- Discuss mechanisms of action, dosing, and administration
- Note contraindications, side effects, and monitoring requirements
- Consider patient-specific factors""",
            user_template="""Medical Condition: {condition}

Patient Context: {patient_context}

Reference Information:
{context}

Please discuss:
1. Evidence-based treatment options (first-line to alternative)
2. Mechanism of action and expected outcomes
3. Dosing and administration guidelines
4. Contraindications and precautions
5. Side effects and monitoring requirements
6. Patient education points""",
            variables=["condition", "patient_context", "context"]
        ),
        
        PromptType.SUMMARY: PromptTemplate(
            name="Clinical Summary",
            system_prompt=BASE_SYSTEM + """\n
For this task, create concise, accurate summaries of clinical information.
- Capture key points and essential information
- Use standard medical documentation format
- Maintain clinical accuracy while being concise
- Highlight critical findings and action items""",
            user_template="""Clinical Information:
{clinical_text}

Please provide a structured summary including:
1. Chief complaint/reason for encounter
2. Key history and physical findings
3. Relevant test results
4. Assessment/diagnoses
5. Plan and recommendations
6. Follow-up requirements""",
            variables=["clinical_text"]
        ),
        
        PromptType.EXPLANATION: PromptTemplate(
            name="Medical Explanation",
            system_prompt=BASE_SYSTEM + """\n
For this task, explain medical concepts in clear, accessible language.
- Use plain language while maintaining accuracy
- Provide context and relevant background
- Use analogies when helpful
- Address common misconceptions
- Tailor complexity to the audience""",
            user_template="""Medical Concept: {concept}

Audience Level: {audience_level}

Context:
{context}

Please provide a clear explanation that:
1. Defines the concept in accessible terms
2. Explains why it matters
3. Provides relevant examples or analogies
4. Addresses common questions or misconceptions
5. Includes any important caveats or limitations""",
            variables=["concept", "audience_level", "context"]
        ),
        
        PromptType.ADVERSE_EVENTS: PromptTemplate(
            name="Adverse Events Analysis",
            system_prompt=BASE_SYSTEM + """\n
For this task, analyze potential adverse events and medication reactions.
- Consider drug-drug interactions
- Evaluate severity and likelihood
- Recommend monitoring and management
- Note when to seek urgent medical attention""",
            user_template="""Medications/Interventions: {medications}

Patient Factors: {patient_factors}

Context:
{context}

Please analyze:
1. Potential adverse effects and their likelihood
2. Drug-drug or drug-condition interactions
3. Monitoring parameters and frequency
4. Warning signs requiring immediate attention
5. Risk mitigation strategies""",
            variables=["medications", "patient_factors", "context"]
        ),
        
        PromptType.DIFFERENTIAL_DIAGNOSIS: PromptTemplate(
            name="Differential Diagnosis with Reasoning",
            system_prompt=BASE_SYSTEM + """\n
For this task, develop a comprehensive differential diagnosis with explicit reasoning.
- Use clinical reasoning frameworks
- Consider epidemiology and patient context
- Apply Bayesian reasoning (pre-test probability)
- Explain the diagnostic approach""",
            user_template="""Chief Complaint: {chief_complaint}

Clinical Details:
{clinical_details}

Reference Context:
{context}

Please provide:
1. Initial problem representation
2. Broad differential categories (anatomic, pathophysiologic)
3. Specific differential diagnoses with likelihood ratings
4. Reasoning using supporting and refuting evidence
5. Diagnostic strategy (what tests to order and why)
6. Clinical pearls or key distinguishing features""",
            variables=["chief_complaint", "clinical_details", "context"]
        )
    }
    
    # Few-shot examples for different tasks
    EXAMPLES = {
        PromptType.QUESTION_ANSWERING: [
            {
                "question": "What are the first-line treatments for hypertension?",
                "context": "Guidelines recommend thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers as first-line agents...",
                "answer": "According to current guidelines, first-line treatments for hypertension include: 1) Thiazide diuretics..."
            }
        ]
    }


class PromptBuilder:
    """Build prompts from templates."""
    
    def __init__(self):
        """Initialize prompt builder."""
        self.templates = MedicalPromptTemplates.TEMPLATES
        logger.info("Initialized PromptBuilder")
    
    def build(
        self,
        prompt_type: PromptType,
        variables: Dict[str, str],
        include_examples: bool = False,
        custom_system: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build prompt from template.
        
        Args:
            prompt_type: Type of prompt to build
            variables: Variables to fill in template
            include_examples: Whether to include few-shot examples
            custom_system: Custom system prompt (overrides template)
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.templates[prompt_type]
        
        # Check all required variables are provided
        missing = set(template.variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Build user prompt
        user_prompt = template.user_template.format(**variables)
        
        # Add examples if requested
        if include_examples and prompt_type in MedicalPromptTemplates.EXAMPLES:
            examples_text = "\n\nExamples:\n"
            for i, example in enumerate(MedicalPromptTemplates.EXAMPLES[prompt_type], 1):
                examples_text += f"\nExample {i}:\n"
                for key, value in example.items():
                    examples_text += f"{key.title()}: {value}\n"
            user_prompt = examples_text + "\n" + user_prompt
        
        # Use custom or template system prompt
        system_prompt = custom_system or template.system_prompt
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def build_rag_prompt(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        include_sources: bool = True
    ) -> Dict[str, str]:
        """
        Build RAG-specific prompt with context.
        
        Args:
            question: User question
            context: Retrieved context from documents
            conversation_history: Previous conversation turns
            include_sources: Whether to ask for source citations
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Build base prompt
        variables = {
            "question": question,
            "context": context
        }
        
        prompts = self.build(PromptType.QUESTION_ANSWERING, variables)
        
        # Add conversation history if provided
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_text += f"{turn['role']}: {turn['content']}\n"
            prompts["user"] = history_text + "\n" + prompts["user"]
        
        # Add source citation instruction
        if include_sources:
            prompts["user"] += "\n\nPlease cite relevant sources from the context in your answer."
        
        return prompts
    
    def build_chain_of_thought_prompt(
        self,
        task: str,
        context: str,
        reasoning_steps: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Build chain-of-thought prompt for complex reasoning.
        
        Args:
            task: Task description
            context: Relevant context
            reasoning_steps: Specific reasoning steps to include
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = MedicalPromptTemplates.BASE_SYSTEM + """\n
For complex medical reasoning, use a step-by-step approach:
1. Break down the problem into components
2. Consider each component systematically
3. Synthesize information to reach conclusions
4. Verify reasoning and check for errors
5. Provide clear, justified recommendations

Think through each step explicitly before providing your final answer."""
        
        user_prompt = f"Task: {task}\n\nContext: {context}\n\n"
        
        if reasoning_steps:
            user_prompt += "Please address the following steps in your reasoning:\n"
            for i, step in enumerate(reasoning_steps, 1):
                user_prompt += f"{i}. {step}\n"
        else:
            user_prompt += "Please think through this step-by-step and show your reasoning."
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }


class SafetyPrompts:
    """Safety-specific prompts and disclaimers."""
    
    MEDICAL_DISCLAIMER = """
IMPORTANT MEDICAL DISCLAIMER:
This information is for educational purposes only and should not be considered medical advice. 
Always consult with qualified healthcare professionals for medical diagnosis, treatment, and care.
In case of medical emergency, call emergency services immediately."""
    
    EMERGENCY_WARNING = """
⚠️ EMERGENCY WARNING ⚠️
Based on the information provided, this may be a medical emergency requiring immediate attention.
Please call emergency services (911 in the US) or go to the nearest emergency room immediately.
Do not rely on this system for emergency medical guidance."""
    
    LIMITATION_NOTICE = """
Note: This response is based on general medical knowledge and the provided context. 
Individual cases may vary significantly, and personalized medical evaluation is essential.
Always consider patient-specific factors, comorbidities, and current clinical guidelines."""
    
    @staticmethod
    def add_disclaimer(text: str, include_limitations: bool = True) -> str:
        """
        Add medical disclaimer to text.
        
        Args:
            text: Original text
            include_limitations: Whether to include limitation notice
            
        Returns:
            Text with disclaimer
        """
        disclaimer_text = SafetyPrompts.MEDICAL_DISCLAIMER
        if include_limitations:
            disclaimer_text += "\n\n" + SafetyPrompts.LIMITATION_NOTICE
        
        return text + "\n\n---\n" + disclaimer_text
    
    @staticmethod
    def wrap_emergency_warning(text: str) -> str:
        """
        Wrap text with emergency warning.
        
        Args:
            text: Original text
            
        Returns:
            Text with emergency warning
        """
        return SafetyPrompts.EMERGENCY_WARNING + "\n\n" + text
