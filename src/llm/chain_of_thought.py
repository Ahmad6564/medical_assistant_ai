"""
Chain-of-thought reasoning for complex medical tasks.
Implements structured reasoning approaches for clinical decision-making.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .llm_interface import LLMInterface, Message
from .prompt_engineering import PromptBuilder, PromptType
from ..utils import get_logger

logger = get_logger(__name__)


class ReasoningStrategy(Enum):
    """Types of reasoning strategies."""
    LINEAR = "linear"  # Step-by-step linear reasoning
    TREE = "tree"  # Tree-based exploration of possibilities
    CRITIQUE = "critique"  # Self-critique and refinement
    SOCRATIC = "socratic"  # Question-driven exploration


@dataclass
class ReasoningStep:
    """Single step in reasoning chain."""
    step_number: int
    description: str
    reasoning: str
    conclusion: str
    confidence: float
    
metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningResult:
    """Result from chain-of-thought reasoning."""
    final_answer: str
    reasoning_steps: List[ReasoningStep]
    confidence: float
    alternative_conclusions: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class ChainOfThoughtReasoner:
    """
    Implement chain-of-thought reasoning for medical AI.
    Uses structured prompting to elicit step-by-step reasoning.
    """
    
    def __init__(
        self,
        llm: LLMInterface,
        strategy: ReasoningStrategy = ReasoningStrategy.LINEAR,
        max_steps: int = 5,
        temperature: float = 0.3
    ):
        """
        Initialize chain-of-thought reasoner.
        
        Args:
            llm: LLM interface
            strategy: Reasoning strategy to use
            max_steps: Maximum reasoning steps
            temperature: LLM temperature (lower for more focused reasoning)
        """
        self.llm = llm
        self.strategy = strategy
        self.max_steps = max_steps
        self.temperature = temperature
        self.prompt_builder = PromptBuilder()
        
        logger.info(f"Initialized ChainOfThoughtReasoner with {strategy.value} strategy")
    
    def reason(
        self,
        task: str,
        context: str,
        reasoning_framework: Optional[List[str]] = None
    ) -> ReasoningResult:
        """
        Perform chain-of-thought reasoning.
        
        Args:
            task: Task or question to reason about
            context: Relevant context information
            reasoning_framework: Specific steps to follow
            
        Returns:
            ReasoningResult with steps and conclusion
        """
        if self.strategy == ReasoningStrategy.LINEAR:
            return self._linear_reasoning(task, context, reasoning_framework)
        elif self.strategy == ReasoningStrategy.TREE:
            return self._tree_reasoning(task, context)
        elif self.strategy == ReasoningStrategy.CRITIQUE:
            return self._critique_reasoning(task, context)
        elif self.strategy == ReasoningStrategy.SOCRATIC:
            return self._socratic_reasoning(task, context)
        else:
            raise ValueError(f"Unknown reasoning strategy: {self.strategy}")
    
    def _linear_reasoning(
        self,
        task: str,
        context: str,
        reasoning_framework: Optional[List[str]] = None
    ) -> ReasoningResult:
        """
        Linear step-by-step reasoning.
        
        Args:
            task: Task description
            context: Context information
            reasoning_framework: Specific steps to follow
            
        Returns:
            ReasoningResult
        """
        # Build chain-of-thought prompt
        prompts = self.prompt_builder.build_chain_of_thought_prompt(
            task=task,
            context=context,
            reasoning_steps=reasoning_framework
        )
        
        messages = [
            Message(role="system", content=prompts["system"]),
            Message(role="user", content=prompts["user"])
        ]
        
        # Generate reasoning
        response = self.llm.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=2000
        )
        
        # Parse response into steps
        reasoning_steps = self._parse_reasoning_steps(response.content)
        
        # Extract final conclusion
        final_answer = self._extract_conclusion(response.content)
        
        # Calculate overall confidence
        avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.5
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=avg_confidence
        )
    
    def _tree_reasoning(self, task: str, context: str) -> ReasoningResult:
        """
        Tree-based reasoning exploring multiple paths.
        
        Args:
            task: Task description
            context: Context information
            
        Returns:
            ReasoningResult
        """
        # Generate initial hypotheses
        hypotheses_prompt = f"""Given the following task, generate 3-5 possible approaches or hypotheses:

Task: {task}

Context: {context}

For each hypothesis, briefly explain the reasoning."""
        
        messages = [Message(role="user", content=hypotheses_prompt)]
        response = self.llm.generate(messages, temperature=0.7, max_tokens=1000)
        
        # Evaluate each hypothesis
        evaluation_prompt = f"""Now, evaluate each hypothesis and determine which is most likely correct:

Original task: {task}
Context: {context}

Hypotheses:
{response.content}

Provide a detailed evaluation and select the best hypothesis with justification."""
        
        messages.append(Message(role="assistant", content=response.content))
        messages.append(Message(role="user", content=evaluation_prompt))
        
        final_response = self.llm.generate(messages, temperature=self.temperature, max_tokens=1500)
        
        # Parse into reasoning result
        reasoning_steps = self._parse_reasoning_steps(final_response.content)
        final_answer = self._extract_conclusion(final_response.content)
        
        # Extract alternative conclusions from hypotheses
        alternative_conclusions = self._extract_alternatives(response.content)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=0.7,
            alternative_conclusions=alternative_conclusions
        )
    
    def _critique_reasoning(self, task: str, context: str) -> ReasoningResult:
        """
        Self-critique reasoning with refinement.
        
        Args:
            task: Task description
            context: Context information
            
        Returns:
            ReasoningResult
        """
        # Initial reasoning
        initial_prompt = f"""Task: {task}

Context: {context}

Please provide your initial reasoning and conclusion."""
        
        messages = [Message(role="user", content=initial_prompt)]
        initial_response = self.llm.generate(messages, temperature=0.5, max_tokens=1000)
        
        # Self-critique
        critique_prompt = f"""Now, critically evaluate your previous reasoning:

1. What assumptions did you make?
2. What evidence supports or contradicts your conclusion?
3. What are potential weaknesses in the reasoning?
4. What alternative explanations exist?
5. How confident should we be in this conclusion?

Provide a refined answer based on this critique."""
        
        messages.append(Message(role="assistant", content=initial_response.content))
        messages.append(Message(role="user", content=critique_prompt))
        
        refined_response = self.llm.generate(messages, temperature=self.temperature, max_tokens=1500)
        
        # Parse results
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                description="Initial reasoning",
                reasoning=initial_response.content,
                conclusion=self._extract_conclusion(initial_response.content),
                confidence=0.6
            ),
            ReasoningStep(
                step_number=2,
                description="Critical evaluation and refinement",
                reasoning=refined_response.content,
                conclusion=self._extract_conclusion(refined_response.content),
                confidence=0.8
            )
        ]
        
        final_answer = self._extract_conclusion(refined_response.content)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=0.8
        )
    
    def _socratic_reasoning(self, task: str, context: str) -> ReasoningResult:
        """
        Socratic method - reasoning through questions.
        
        Args:
            task: Task description
            context: Context information
            
        Returns:
            ReasoningResult
        """
        # Generate key questions
        questions_prompt = f"""Using the Socratic method, generate 5 key questions that would help reason through this task:

Task: {task}
Context: {context}

Format: List each question numbered 1-5."""
        
        messages = [Message(role="user", content=questions_prompt)]
        questions_response = self.llm.generate(messages, temperature=0.5, max_tokens=500)
        
        # Answer each question
        answers_prompt = f"""Now, answer each of these questions systematically:

{questions_response.content}

Context: {context}

Provide detailed answers that build toward a final conclusion about: {task}"""
        
        messages.append(Message(role="assistant", content=questions_response.content))
        messages.append(Message(role="user", content=answers_prompt))
        
        answers_response = self.llm.generate(messages, temperature=self.temperature, max_tokens=2000)
        
        # Parse into steps
        reasoning_steps = self._parse_reasoning_steps(answers_response.content)
        final_answer = self._extract_conclusion(answers_response.content)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=0.75
        )
    
    def _parse_reasoning_steps(self, text: str) -> List[ReasoningStep]:
        """
        Parse text into reasoning steps.
        
        Args:
            text: Text to parse
            
        Returns:
            List of ReasoningStep objects
        """
        steps = []
        
        # Simple parsing - look for numbered steps or paragraphs
        lines = text.split('\n')
        current_step = None
        step_num = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number (step indicator)
            if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                if current_step:
                    steps.append(current_step)
                
                step_num += 1
                current_step = ReasoningStep(
                    step_number=step_num,
                    description=f"Step {step_num}",
                    reasoning=line,
                    conclusion="",
                    confidence=0.7
                )
            elif current_step:
                current_step.reasoning += " " + line
        
        # Add last step
        if current_step:
            steps.append(current_step)
        
        # If no numbered steps found, create one step with all content
        if not steps:
            steps = [ReasoningStep(
                step_number=1,
                description="Reasoning",
                reasoning=text,
                conclusion="",
                confidence=0.7
            )]
        
        return steps
    
    def _extract_conclusion(self, text: str) -> str:
        """
        Extract final conclusion from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Conclusion string
        """
        # Look for conclusion indicators
        conclusion_markers = [
            "in conclusion",
            "therefore",
            "thus",
            "final answer",
            "conclusion:",
            "summary:"
        ]
        
        text_lower = text.lower()
        
        for marker in conclusion_markers:
            if marker in text_lower:
                idx = text_lower.rfind(marker)
                conclusion = text[idx:].strip()
                # Take first 500 characters after marker
                return conclusion[:500]
        
        # If no marker found, take last paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        # Fallback: truncate text
        return text[-500:] if len(text) > 500 else text
    
    def _extract_alternatives(self, text: str) -> List[str]:
        """
        Extract alternative conclusions from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of alternative conclusions
        """
        alternatives = []
        
        # Look for numbered hypotheses or alternatives
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                alternatives.append(line)
                if len(alternatives) >= 5:
                    break
        
        return alternatives


class ClinicalReasoningFramework:
    """
    Structured framework for clinical reasoning.
    Implements common clinical reasoning patterns.
    """
    
    DIAGNOSTIC_REASONING_STEPS = [
        "Problem representation: Synthesize key clinical features",
        "Generate differential diagnosis (broad categories and specific conditions)",
        "Evaluate likelihood using clinical evidence and epidemiology",
        "Identify distinguishing features and diagnostic criteria",
        "Recommend diagnostic testing strategy",
        "Formulate clinical impression and next steps"
    ]
    
    TREATMENT_REASONING_STEPS = [
        "Define treatment goals and desired outcomes",
        "Review evidence-based treatment options",
        "Consider patient-specific factors (contraindications, preferences, comorbidities)",
        "Weigh benefits vs. risks for each option",
        "Develop treatment plan with monitoring strategy",
        "Identify potential complications and mitigation strategies"
    ]
    
    MEDICATION_REASONING_STEPS = [
        "Verify indication and appropriateness",
        "Check for contraindications and interactions",
        "Determine appropriate dosing and route",
        "Identify monitoring parameters",
        "Assess cost-effectiveness and patient acceptance",
        "Plan for follow-up and reassessment"
    ]
    
    @staticmethod
    def get_framework(task_type: str) -> List[str]:
        """
        Get reasoning framework for task type.
        
        Args:
            task_type: Type of clinical task
            
        Returns:
            List of reasoning steps
        """
        frameworks = {
            "diagnosis": ClinicalReasoningFramework.DIAGNOSTIC_REASONING_STEPS,
            "treatment": ClinicalReasoningFramework.TREATMENT_REASONING_STEPS,
            "medication": ClinicalReasoningFramework.MEDICATION_REASONING_STEPS
        }
        
        return frameworks.get(task_type.lower(), ClinicalReasoningFramework.DIAGNOSTIC_REASONING_STEPS)
