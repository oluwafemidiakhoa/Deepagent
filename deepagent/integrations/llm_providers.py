"""
LLM Provider Integrations

Unified interface for multiple LLM providers:
- OpenAI (GPT-4, GPT-4-turbo, GPT-3.5)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- Ollama (Local models)

Author: Oluwafemi Idiakhoa
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    tokens_used: int
    cost: float
    latency: float
    metadata: Dict[str, Any]


class LLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, api_key: Optional[str] = None, model: str = "default"):
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.total_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key from environment"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Stream response from LLM"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model": self.model
        }


class OpenAIProvider(LLMProvider):
    """OpenAI API Provider"""

    def _get_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            import openai
            import time

            client = openai.OpenAI(api_key=self.api_key)

            start_time = time.time()

            response = client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )

            latency = time.time() - start_time

            # Calculate cost (approximate rates as of 2025)
            cost_per_1k_tokens = {
                "gpt-4": 0.06,  # Input: $0.03, Output: $0.06 average
                "gpt-4-turbo": 0.02,
                "gpt-3.5-turbo": 0.002,
            }

            model_name = response.model
            tokens = response.usage.total_tokens
            cost = (tokens / 1000) * cost_per_1k_tokens.get(model_name.split("-")[0] + "-" + model_name.split("-")[1], 0.02)

            self.total_tokens += tokens
            self.total_cost += cost

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=tokens,
                cost=cost,
                latency=latency,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def stream(self, prompt: str, **kwargs):
        """Stream response from OpenAI"""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            stream = client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {str(e)}")


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) API Provider"""

    def _get_api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return api_key

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API"""
        try:
            import anthropic
            import time

            client = anthropic.Anthropic(api_key=self.api_key)

            start_time = time.time()

            response = client.messages.create(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )

            latency = time.time() - start_time

            # Calculate cost (approximate rates as of 2025)
            cost_per_1k_tokens = {
                "claude-3-5-sonnet": 0.015,  # Average input/output
                "claude-3-opus": 0.075,
                "claude-3-sonnet": 0.015,
                "claude-3-haiku": 0.0025,
            }

            model_name = response.model
            tokens = response.usage.input_tokens + response.usage.output_tokens
            cost = (tokens / 1000) * cost_per_1k_tokens.get(model_name, 0.015)

            self.total_tokens += tokens
            self.total_cost += cost

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                tokens_used=tokens,
                cost=cost,
                latency=latency,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def stream(self, prompt: str, **kwargs):
        """Stream response from Anthropic"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            with client.messages.stream(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", 2000),
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise RuntimeError(f"Anthropic streaming error: {str(e)}")


class OllamaProvider(LLMProvider):
    """Ollama (Local Models) Provider"""

    def _get_api_key(self) -> str:
        # Ollama doesn't require API key
        return ""

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        try:
            import requests
            import time

            base_url = kwargs.get("base_url", "http://localhost:11434")
            model = kwargs.get("model", self.model or "llama2")

            start_time = time.time()

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 1.0),
                    }
                }
            )

            response.raise_for_status()
            data = response.json()

            latency = time.time() - start_time

            # Ollama is free (local), no cost
            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
            self.total_tokens += tokens

            return LLMResponse(
                content=data["response"],
                model=model,
                tokens_used=tokens,
                cost=0.0,  # Local model, no cost
                latency=latency,
                metadata={
                    "eval_count": data.get("eval_count", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "total_duration": data.get("total_duration", 0),
                }
            )

        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")

    def stream(self, prompt: str, **kwargs):
        """Stream response from Ollama"""
        try:
            import requests
            import json

            base_url = kwargs.get("base_url", "http://localhost:11434")
            model = kwargs.get("model", self.model or "llama2")

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                },
                stream=True
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

        except Exception as e:
            raise RuntimeError(f"Ollama streaming error: {str(e)}")


def get_llm_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    model: str = "default"
) -> LLMProvider:
    """
    Factory function to get LLM provider

    Args:
        provider_name: "openai", "anthropic", or "ollama"
        api_key: Optional API key (uses env var if not provided)
        model: Model name

    Returns:
        Initialized LLM provider

    Example:
        >>> provider = get_llm_provider("openai", model="gpt-4")
        >>> response = provider.generate("Hello, world!")
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    if provider_name.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(providers.keys())}")

    provider_class = providers[provider_name.lower()]
    return provider_class(api_key=api_key, model=model)
