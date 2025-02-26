"""
LLM Provider interfaces for the BMW Agents framework.
This module provides a uniform interface to interact with various LLM APIs.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

import openai
from anthropic import Anthropic
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Provides a standard interface for interacting with different LLM APIs.
    """
    
    @abstractmethod
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.7, 
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM based on the provided messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness of the output
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with response content and metadata
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The input text
            
        Returns:
            Number of tokens
        """
        pass


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_name: The name of the OpenAI model to use
            api_key: OpenAI API key (if None, will try to use from environment)
        """
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.7, 
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate a response using the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": self.model_name,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using OpenAI's tiktoken."""
        import tiktoken
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))


class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic models."""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_name: The name of the Anthropic model to use
            api_key: Anthropic API key (if None, will try to use from environment)
        """
        self.model_name = model_name
        if api_key:
            self.api_key = api_key
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = Anthropic(api_key=self.api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.7, 
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate a response using the Anthropic API."""
        try:
            # Convert messages to Anthropic format if needed
            anthropic_messages = messages
            
            response = self.client.messages.create(
                model=self.model_name,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens or 1024
            )
            
            return {
                "content": response.content[0].text,
                "model": self.model_name,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            logging.error(f"Error calling Anthropic API: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Anthropic models.
        This is a rough approximation as Anthropic doesn't provide a public tokenizer.
        """
        # Crude approximation: ~4 characters per token
        return len(text) // 4


class OllamaProvider(LLMProvider):
    """LLM provider for Ollama models."""
    
    def __init__(self, model_name: str = "deepseek-r1:14b", host: Optional[str] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            model_name: The name of the Ollama model to use
            host: Ollama host address (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize client with host if provided
        if self.host and self.host != "http://localhost:11434":
            ollama.set_host(self.host)
            
        logging.info(f"Initialized Ollama provider with model: {model_name}, host: {self.host}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.7, 
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate a response using Ollama."""
        try:
            # Log the input messages (list only first few for brevity)
            msg_preview = []
            for m in messages[:3]:  # Only show up to 3 messages
                msg_preview.append({"role": m["role"], "content": m["content"][:100] + "..."})
            logging.debug(f"Sending messages to Ollama: {msg_preview}")
            
            # Set options for generation
            options = {
                "temperature": temperature,
            }
            
            # Add max tokens if provided
            if max_tokens:
                options["num_predict"] = max_tokens
                
            # Make the API call
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options
            )
            
            # Log the raw response for debugging (in a safer way)
            logging.debug(f"Raw Ollama response keys: {response.keys() if hasattr(response, 'keys') else 'Not a dict'}")
            
            # Extract the content from the response
            content = response["message"]["content"]
            
            # Log the extracted content
            logging.debug(f"Extracted content: {content[:200]}...")
            
            # Structure the response in a consistent format
            return {
                "content": content,
                "model": self.model_name,
                "finish_reason": "stop",  # Ollama doesn't provide finish reason
                "usage": {
                    # Ollama doesn't provide token counts directly
                    "prompt_tokens": self._estimate_token_count(" ".join([m["content"] for m in messages])),
                    "completion_tokens": self._estimate_token_count(content),
                    "total_tokens": 0  # Will be calculated below
                }
            }
        except Exception as e:
            logging.error(f"Error calling Ollama API: {str(e)}")
            raise
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for Ollama models."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.
        This is a very rough estimation as Ollama doesn't expose token counting.
        """
        return self._estimate_token_count(text)


def get_llm_provider(provider_name: str, model_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        provider_name: Name of the provider ('openai', 'anthropic', 'ollama')
        model_name: Specific model to use (optional)
        
    Returns:
        An instance of the requested LLM provider
    """
    if provider_name.lower() == "openai":
        return OpenAIProvider(model_name or "gpt-4")
    elif provider_name.lower() == "anthropic":
        return AnthropicProvider(model_name or "claude-3-opus-20240229")
    elif provider_name.lower() == "ollama":
        return OllamaProvider(model_name or "deepseek-r1:14b")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}") 