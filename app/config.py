"""
Configuration management for the AI Chatbot system.
Handles multi-model settings, RAG configuration, and service endpoints.
"""
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from enum import Enum
from functools import lru_cache


class ModelProvider(str, Enum):
    """Supported AI model providers"""
    OLLAMA = "ollama"
    OPENWEBUI = "openwebui"
    OPENROUTER = "openrouter"


class OllamaModel(str, Enum):
    """Available Ollama models on local GPU server"""
    LLAMA3_2 = "llama3.2"
    MISTRAL = "mistral"
    DOLPHIN_MIXTRAL = "dolphin-mixtral"
    QWEN2_5_CODER = "qwen2.5-coder"


class ModelConfig(BaseModel):
    """Configuration for a specific model"""
    name: str
    provider: ModelProvider
    max_tokens: int = 2048
    temperature: float = 0.7
    context_window: int = 4096
    use_for_code: bool = False
    use_for_support: bool = True


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database settings
    postgres_user: str = Field(default="chatbot_user")
    postgres_password: str = Field(default="secret")
    postgres_db: str = Field(default="chatbot_db")
    postgres_host: str = Field(default="ai-chatbot-db")
    postgres_port: int = Field(default=5432)
    
    # Ollama settings (local GPU server)
    ollama_host: str = Field(default="96.31.83.171")
    ollama_port: int = Field(default=11434)
    ollama_default_model: str = Field(default="llama3.2")
    ollama_timeout: int = Field(default=120)
    
    # Open-WebUI settings
    openwebui_host: str = Field(default="96.31.83.171")
    openwebui_port: int = Field(default=8080)
    openwebui_api_key: Optional[str] = Field(default=None)
    
    # OpenRouter (cloud fallback)
    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_model: str = Field(default="google/gemma-3-27b-it")
    openrouter_api_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")
    
    # Qdrant Vector DB settings
    qdrant_host: str = Field(default="qdrant")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_kb: str = Field(default="knowledge_base")
    qdrant_collection_tickets: str = Field(default="ticket_history")
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_dimension: int = Field(default=768)
    
    # WHMCS API settings
    whmcs_api_url: Optional[str] = Field(default=None)
    whmcs_api_key: Optional[str] = Field(default=None)
    whmcs_api_identifier: Optional[str] = Field(default=None)
    whmcs_username: Optional[str] = Field(default=None)
    whmcs_password: Optional[str] = Field(default=None)
    whmcs_webhook_secret: Optional[str] = Field(default=None)
    
    # n8n Workflow settings
    n8n_host: str = Field(default="n8n")
    n8n_port: int = Field(default=5678)
    n8n_api_key: Optional[str] = Field(default=None)
    n8n_webhook_url: Optional[str] = Field(default=None)
    
    # Redis cache settings
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_password: Optional[str] = Field(default=None)
    redis_db: int = Field(default=0)
    cache_ttl: int = Field(default=3600)  # 1 hour default
    
    # Rate limiting
    rate_limit_requests: int = Field(default=60)
    rate_limit_window: int = Field(default=60)  # seconds
    
    # Security
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration: int = Field(default=3600)
    
    # PII Detection
    pii_detection_enabled: bool = Field(default=True)
    pii_redaction_enabled: bool = Field(default=True)
    
    # Logging
    log_level: str = Field(default="INFO")
    audit_log_enabled: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL connection URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def ollama_base_url(self) -> str:
        """Construct Ollama API base URL"""
        return f"http://{self.ollama_host}:{self.ollama_port}"
    
    @property
    def openwebui_base_url(self) -> str:
        """Construct Open-WebUI API base URL"""
        return f"http://{self.openwebui_host}:{self.openwebui_port}"
    
    @property
    def qdrant_url(self) -> str:
        """Construct Qdrant URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    @property
    def n8n_base_url(self) -> str:
        """Construct n8n base URL"""
        return f"http://{self.n8n_host}:{self.n8n_port}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Model configurations for different use cases
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "llama3.2": ModelConfig(
        name="llama3.2",
        provider=ModelProvider.OLLAMA,
        max_tokens=4096,
        temperature=0.7,
        context_window=8192,
        use_for_support=True,
        use_for_code=False
    ),
    "mistral": ModelConfig(
        name="mistral",
        provider=ModelProvider.OLLAMA,
        max_tokens=4096,
        temperature=0.7,
        context_window=8192,
        use_for_support=True,
        use_for_code=False
    ),
    "dolphin-mixtral": ModelConfig(
        name="dolphin-mixtral",
        provider=ModelProvider.OLLAMA,
        max_tokens=4096,
        temperature=0.8,
        context_window=32768,
        use_for_support=True,
        use_for_code=True
    ),
    "qwen2.5-coder": ModelConfig(
        name="qwen2.5-coder",
        provider=ModelProvider.OLLAMA,
        max_tokens=8192,
        temperature=0.3,
        context_window=32768,
        use_for_support=False,
        use_for_code=True
    ),
}

# Fallback chain order
FALLBACK_CHAIN = [
    ModelProvider.OLLAMA,
    ModelProvider.OPENWEBUI,
    ModelProvider.OPENROUTER
]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
