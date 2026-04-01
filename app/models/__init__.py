"""
Database Models module for WHMCS AI Chatbot.

Models:
- Ticket: Support ticket model
- ChatSession: Chat session tracking
- ChatMessage: Individual chat messages
- KnowledgeArticle: Knowledge base articles
- AuditLog: Audit logging
- ModelMetrics: Model performance metrics
"""

from .ticket import Ticket
from .chat import ChatSession, ChatMessage, KnowledgeArticle, AuditLog, ModelMetrics

__all__ = [
    "Ticket",
    "ChatSession",
    "ChatMessage",
    "KnowledgeArticle",
    "AuditLog",
    "ModelMetrics"
]
