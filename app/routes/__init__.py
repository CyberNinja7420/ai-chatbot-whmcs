"""
API Routes module for WHMCS AI Chatbot.

Routes:
- chatbot: Legacy chat endpoint
- tickets: WHMCS tickets endpoint
- chat_enhanced: Enhanced chat with multi-model and RAG
- admin: Admin dashboard endpoints
- webhooks: WHMCS webhook handlers
"""

from . import chatbot, tickets

__all__ = ["chatbot", "tickets"]
