"""
Services module for WHMCS AI Chatbot.

Available services:
- model_provider: Multi-model AI provider with fallback chain
- rag_service: RAG integration with Qdrant
- whmcs_service: WHMCS API integration
- n8n_service: n8n workflow automation
- security: Rate limiting, caching, audit logging, PII detection
- analytics: Chat analytics and metrics
"""

from .model_provider import get_model_provider, MultiModelProvider
from .rag_service import get_rag_service, RAGService
from .whmcs_service import get_whmcs_service
from .n8n_service import get_n8n_service
from .security import (
    get_rate_limiter,
    get_response_cache,
    get_audit_logger,
    get_pii_detector
)
from .analytics import get_analytics_service

__all__ = [
    "get_model_provider",
    "MultiModelProvider",
    "get_rag_service",
    "RAGService",
    "get_whmcs_service",
    "get_n8n_service",
    "get_rate_limiter",
    "get_response_cache",
    "get_audit_logger",
    "get_pii_detector",
    "get_analytics_service"
]
