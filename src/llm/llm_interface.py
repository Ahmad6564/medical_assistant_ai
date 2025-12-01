"""
LLM interface for integrating with various language model providers.
Supports OpenAI, Anthropic, and other providers.
"""

import os
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]


@dataclass
class Message:
    """Message in conversation."""
    role: str  # "system", "user", "assistant"
    content: str


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate streaming response from LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        organization: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name
            organization: Organization ID
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            organization=organization
        )
        
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to OpenAI format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using OpenAI API.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
            
        Returns:
            LLMResponse object
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason,
                metadata={"response_id": response.id}
            )
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Generate streaming response using OpenAI API.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
            
        Yields:
            Content chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) API provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229"
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.model = model
        self.client = Anthropic(api_key=self.api_key)
        
        logger.info(f"Initialized Anthropic provider with model: {model}")
    
    def _convert_messages(self, messages: List[Message]) -> Tuple[str, List[Dict]]:
        """Convert Message objects to Anthropic format."""
        system_message = ""
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, conversation
    
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Anthropic API.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
            
        Returns:
            LLMResponse object
        """
        try:
            system_message, conversation = self._convert_messages(messages)
            
            response = self.client.messages.create(
                model=self.model,
                system=system_message if system_message else None,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                finish_reason=response.stop_reason,
                metadata={"response_id": response.id}
            )
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Generate streaming response using Anthropic API.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
            
        Yields:
            Content chunks
        """
        try:
            system_message, conversation = self._convert_messages(messages)
            
            with self.client.messages.stream(
                model=self.model,
                system=system_message if system_message else None,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise


class LLMInterface:
    """
    Unified interface for multiple LLM providers.
    Handles provider selection, fallbacks, and retry logic.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize LLM interface.
        
        Args:
            provider: Provider name ("openai" or "anthropic")
            model: Model name (provider-specific)
            api_key: API key
            fallback_provider: Fallback provider if primary fails
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional provider arguments
        """
        self.provider_name = provider.lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize primary provider
        self.provider = self._create_provider(
            provider=self.provider_name,
            model=model,
            api_key=api_key,
            **kwargs
        )
        
        # Initialize fallback provider if specified
        self.fallback = None
        if fallback_provider:
            try:
                self.fallback = self._create_provider(
                    provider=fallback_provider,
                    model=None,
                    api_key=None
                )
                logger.info(f"Fallback provider initialized: {fallback_provider}")
            except Exception as e:
                logger.warning(f"Could not initialize fallback provider: {e}")
    
    def _create_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """Create provider instance."""
        if provider == "openai":
            if model is None:
                model = "gpt-4"
            return OpenAIProvider(api_key=api_key, model=model, **kwargs)
        
        elif provider == "anthropic":
            if model is None:
                model = "claude-3-sonnet-20240229"
            return AnthropicProvider(api_key=api_key, model=model)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_fallback: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with retry logic.
        
        Args:
            messages: Messages (can be list of Message, list of dicts, or single string)
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            use_fallback: Whether to use fallback provider on failure
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        # Convert messages to Message objects
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        elif isinstance(messages, list) and messages and isinstance(messages[0], dict):
            messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        
        # Try primary provider with retries
        for attempt in range(self.max_retries):
            try:
                return self.provider.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Try fallback on final failure
                    if use_fallback and self.fallback:
                        logger.info("Trying fallback provider")
                        try:
                            return self.fallback.generate(
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                **kwargs
                            )
                        except Exception as fallback_error:
                            logger.error(f"Fallback provider also failed: {fallback_error}")
                    
                    raise
    
    def generate_stream(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Generate streaming response.
        
        Args:
            messages: Messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Yields:
            Content chunks
        """
        # Convert messages
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        elif isinstance(messages, list) and messages and isinstance(messages[0], dict):
            messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        
        yield from self.provider.generate_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (approximate).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4
