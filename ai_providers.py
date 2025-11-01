"""
ai_providers.py

Purpose
- Unified provider abstraction for Azure OpenAI and Azure AI Foundry (Models as a Service),
    supporting OpenAI-compatible chat.completions across first- and third-party models (e.g., Grok,
    Mistral, Llama, GPT families).

How it fits
- Consumed by `AIAnalyzer` and the Flask API to issue model-agnostic chat completions. Environment
    variables determine which backend to use and how to authenticate.

Main role
- Provide a single `BaseAIProvider` interface with concrete `AzureOpenAIProvider` and
    `AzureFoundryProvider` implementations. Normalize parameters (temperature, max tokens, response
    format) and return message content strings.

Notes
- Uses the OpenAI-style messages schema for both backends:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
- Foundry requests go to an OpenAI-compatible endpoint (global or regional) with `api-key` auth.
- Factory `get_provider_from_env()` selects the provider based on env vars.
"""
from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, List, Optional

from logger_config import log

try:
    # OpenAI 1.x SDK for Azure OpenAI
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AzureOpenAI = None  # type: ignore


class BaseAIProvider:
    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError


class AzureOpenAIProvider(BaseAIProvider):
    def __init__(self, endpoint: str, api_key: str, api_version: str = "2024-02-01") -> None:
        if AzureOpenAI is None:
            raise RuntimeError("openai package is not available for Azure OpenAI provider")
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
        }
        if response_format:
            params["response_format"] = response_format
        if temperature is not None:
            params["temperature"] = temperature
        # Azure OpenAI uses max_completion_tokens in 2024 APIs
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens

        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content  # type: ignore


class AzureFoundryProvider(BaseAIProvider):
    """
    Azure AI Foundry Inference provider using OpenAI-compatible REST API.

    Endpoint examples:
      - Global models endpoint: https://models.inference.ai.azure.com
      - Regional: https://{region}.models.ai.azure.com

    Required headers:
      - Authorization via 'api-key' header
    """

    def __init__(self, endpoint: str, api_key: str, api_version: Optional[str] = None) -> None:
        if not endpoint.startswith("http"):
            raise ValueError("AZURE_INFERENCE_ENDPOINT must be a valid https URL")
        self.base_url = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version  # currently optional; keep for forward compatibility
        # Use OpenAI-compatible path
        self.chat_path = "/v1/chat/completions"

    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}{self.chat_path}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        # Build OpenAI-compatible payload
        payload: Dict[str, Any] = {
            "model": model,  # e.g., "grok-2", "mistral-large", "gpt-4o-mini", etc.
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            # OpenAI-style name is max_tokens; support that for Foundry
            payload["max_tokens"] = max_tokens
        if self.api_version:
            # Some preview endpoints accept an api-version query param
            url = f"{url}?api-version={self.api_version}"

        try:
            res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            # Bubble up detailed error text if present
            detail = getattr(e, "response", None)
            if detail is not None:
                try:
                    log.error(f"Foundry error: {detail.text}")
                except Exception:
                    pass
            raise


def get_provider_from_env() -> BaseAIProvider:
    """Factory: decide provider based on environment variables.

    Environment variables consumed:
      - AI_PROVIDER: "azure-openai" (default) | "azure-foundry"
      - AZURE_AI_ENDPOINT / AZURE_AI_API_KEY / AZURE_OPENAI_API_VERSION (for azure-openai)
      - AZURE_INFERENCE_ENDPOINT / AZURE_INFERENCE_API_KEY / AZURE_INFERENCE_API_VERSION (for azure-foundry)
    """
    provider = os.getenv("AI_PROVIDER", "azure-openai").lower()

    if provider in ("azure-openai", "azure_openai"):
        endpoint = os.getenv("AZURE_AI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_AI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not endpoint or not api_key:
            raise ValueError("Missing AZURE_AI_ENDPOINT/AZURE_OPENAI_ENDPOINT or AZURE_AI_API_KEY/AZURE_OPENAI_API_KEY")
        log.info("Using Azure OpenAI provider")
        return AzureOpenAIProvider(endpoint, api_key, api_version)

    if provider in ("azure-foundry", "azure_foundry", "foundry"):
        endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com")
        api_key = os.getenv("AZURE_INFERENCE_API_KEY")
        api_version = os.getenv("AZURE_INFERENCE_API_VERSION", None)
        if not api_key:
            raise ValueError("Missing AZURE_INFERENCE_API_KEY for Azure AI Foundry provider")
        log.info(f"Using Azure AI Foundry provider (endpoint={endpoint})")
        return AzureFoundryProvider(endpoint, api_key, api_version)

    raise ValueError(f"Unsupported AI_PROVIDER value: {provider}")
