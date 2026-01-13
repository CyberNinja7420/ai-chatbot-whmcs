"""
Multi-Model Provider Service
Supports Ollama (local), Open-WebUI, and OpenRouter with automatic fallback.
Optimized for 7x RTX 2080 Ti GPU deployment.
"""
import asyncio
import httpx
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import time
import json

from config import (
    get_settings, 
    ModelProvider, 
    ModelConfig, 
    MODEL_CONFIGS, 
    FALLBACK_CHAIN,
    OllamaModel
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from an AI model"""
    content: str
    model: str
    provider: ModelProvider
    tokens_used: int
    latency_ms: float
    cached: bool = False


@dataclass 
class ModelHealth:
    """Health status of a model provider"""
    provider: ModelProvider
    available: bool
    latency_ms: float
    models: List[str]
    error: Optional[str] = None


class MultiModelProvider:
    """
    Multi-model AI provider with automatic fallback chain.
    Supports Ollama, Open-WebUI, and OpenRouter.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._health_cache: Dict[ModelProvider, ModelHealth] = {}
        self._health_cache_ttl = 60  # seconds
        self._last_health_check: Dict[ModelProvider, float] = {}
    
    async def _check_ollama_health(self) -> ModelHealth:
        """Check Ollama server health and available models"""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return ModelHealth(
                        provider=ModelProvider.OLLAMA,
                        available=True,
                        latency_ms=(time.time() - start) * 1000,
                        models=models
                    )
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
        
        return ModelHealth(
            provider=ModelProvider.OLLAMA,
            available=False,
            latency_ms=(time.time() - start) * 1000,
            models=[],
            error=str(e) if 'e' in locals() else "Connection failed"
        )
    
    async def _check_openwebui_health(self) -> ModelHealth:
        """Check Open-WebUI server health"""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {}
                if self.settings.openwebui_api_key:
                    headers["Authorization"] = f"Bearer {self.settings.openwebui_api_key}"
                
                response = await client.get(
                    f"{self.settings.openwebui_base_url}/api/models",
                    headers=headers
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("id", m.get("name", "unknown")) for m in data.get("data", data) if isinstance(m, dict)]
                    return ModelHealth(
                        provider=ModelProvider.OPENWEBUI,
                        available=True,
                        latency_ms=(time.time() - start) * 1000,
                        models=models
                    )
        except Exception as e:
            logger.warning(f"Open-WebUI health check failed: {e}")
        
        return ModelHealth(
            provider=ModelProvider.OPENWEBUI,
            available=False,
            latency_ms=(time.time() - start) * 1000,
            models=[],
            error=str(e) if 'e' in locals() else "Connection failed"
        )
    
    async def _check_openrouter_health(self) -> ModelHealth:
        """Check OpenRouter API health"""
        start = time.time()
        if not self.settings.openrouter_api_key:
            return ModelHealth(
                provider=ModelProvider.OPENROUTER,
                available=False,
                latency_ms=0,
                models=[],
                error="No API key configured"
            )
        
        # OpenRouter is cloud-based, assume available if key exists
        return ModelHealth(
            provider=ModelProvider.OPENROUTER,
            available=True,
            latency_ms=(time.time() - start) * 1000,
            models=[self.settings.openrouter_model]
        )
    
    async def check_health(self, provider: Optional[ModelProvider] = None) -> Dict[ModelProvider, ModelHealth]:
        """Check health of one or all providers"""
        if provider:
            providers = [provider]
        else:
            providers = list(ModelProvider)
        
        results = {}
        tasks = []
        
        for p in providers:
            # Check cache
            if p in self._last_health_check:
                age = time.time() - self._last_health_check[p]
                if age < self._health_cache_ttl and p in self._health_cache:
                    results[p] = self._health_cache[p]
                    continue
            
            if p == ModelProvider.OLLAMA:
                tasks.append((p, self._check_ollama_health()))
            elif p == ModelProvider.OPENWEBUI:
                tasks.append((p, self._check_openwebui_health()))
            elif p == ModelProvider.OPENROUTER:
                tasks.append((p, self._check_openrouter_health()))
        
        if tasks:
            health_results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            for (p, _), result in zip(tasks, health_results):
                if isinstance(result, Exception):
                    result = ModelHealth(
                        provider=p,
                        available=False,
                        latency_ms=0,
                        models=[],
                        error=str(result)
                    )
                results[p] = result
                self._health_cache[p] = result
                self._last_health_check[p] = time.time()
        
        return results
    
    async def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """Call Ollama API"""
        start = time.time()
        
        # Prepare messages with system prompt
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        
        payload = {
            "model": model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        async with httpx.AsyncClient(timeout=self.settings.ollama_timeout) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        
        return ModelResponse(
            content=content,
            model=model,
            provider=ModelProvider.OLLAMA,
            tokens_used=tokens,
            latency_ms=(time.time() - start) * 1000
        )
    
    async def _call_openwebui(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """Call Open-WebUI API (OpenAI compatible)"""
        start = time.time()
        
        # Prepare messages with system prompt
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        
        headers = {"Content-Type": "application/json"}
        if self.settings.openwebui_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openwebui_api_key}"
        
        payload = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient(timeout=self.settings.ollama_timeout) as client:
            response = await client.post(
                f"{self.settings.openwebui_base_url}/api/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        return ModelResponse(
            content=content,
            model=model,
            provider=ModelProvider.OPENWEBUI,
            tokens_used=tokens,
            latency_ms=(time.time() - start) * 1000
        )
    
    async def _call_openrouter(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """Call OpenRouter API"""
        start = time.time()
        
        if not self.settings.openrouter_api_key:
            raise ValueError("OpenRouter API key not configured")
        
        # Prepare messages with system prompt
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model or self.settings.openrouter_model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.settings.openrouter_api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        return ModelResponse(
            content=content,
            model=model or self.settings.openrouter_model,
            provider=ModelProvider.OPENROUTER,
            tokens_used=tokens,
            latency_ms=(time.time() - start) * 1000
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[ModelProvider] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        use_fallback: bool = True
    ) -> ModelResponse:
        """
        Generate a response using the specified or default model.
        Uses fallback chain if primary provider fails.
        """
        # Get model config
        model_name = model or self.settings.ollama_default_model
        model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS.get("llama3.2"))
        
        # Override with explicit values
        final_max_tokens = max_tokens or model_config.max_tokens
        final_temperature = temperature if temperature is not None else model_config.temperature
        
        # Determine provider order
        if provider:
            providers = [provider]
            if use_fallback:
                providers += [p for p in FALLBACK_CHAIN if p != provider]
        else:
            providers = FALLBACK_CHAIN.copy()
        
        errors = []
        
        for p in providers:
            try:
                logger.info(f"Attempting generation with {p.value}:{model_name}")
                
                if p == ModelProvider.OLLAMA:
                    return await self._call_ollama(
                        messages, model_name, final_max_tokens, final_temperature, system_prompt
                    )
                elif p == ModelProvider.OPENWEBUI:
                    return await self._call_openwebui(
                        messages, model_name, final_max_tokens, final_temperature, system_prompt
                    )
                elif p == ModelProvider.OPENROUTER:
                    return await self._call_openrouter(
                        messages, self.settings.openrouter_model, final_max_tokens, final_temperature, system_prompt
                    )
                    
            except Exception as e:
                logger.warning(f"Provider {p.value} failed: {e}")
                errors.append(f"{p.value}: {str(e)}")
                
                if not use_fallback:
                    raise
                continue
        
        raise RuntimeError(f"All providers failed: {'; '.join(errors)}")
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from Ollama (streaming only supported for local models).
        """
        model_name = model or self.settings.ollama_default_model
        model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS.get("llama3.2"))
        
        final_max_tokens = max_tokens or model_config.max_tokens
        final_temperature = temperature if temperature is not None else model_config.temperature
        
        # Prepare messages with system prompt
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        
        payload = {
            "model": model_name,
            "messages": full_messages,
            "stream": True,
            "options": {
                "num_predict": final_max_tokens,
                "temperature": final_temperature
            }
        }
        
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embeddings using Ollama's embedding model.
        """
        embed_model = model or self.settings.embedding_model
        
        payload = {
            "model": embed_model,
            "prompt": text
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/embeddings",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        return data.get("embedding", [])
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self.get_embedding(text, model) for text in batch]
            )
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_available_models(self) -> List[str]:
        """Get list of configured model names"""
        return list(MODEL_CONFIGS.keys())
    
    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return MODEL_CONFIGS.get(model)


# Singleton instance
_provider_instance: Optional[MultiModelProvider] = None


def get_model_provider() -> MultiModelProvider:
    """Get or create the model provider singleton"""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = MultiModelProvider()
    return _provider_instance
