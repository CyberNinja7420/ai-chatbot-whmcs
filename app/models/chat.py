"""
Database models for chat sessions and messages.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from utils.database import Base


class ChatSession(Base):
    """Chat session model"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    client_id = Column(Integer, index=True, nullable=True)
    client_email = Column(String(255), nullable=True)
    
    # Model info
    model_used = Column(String(100), nullable=True)
    provider = Column(String(50), nullable=True)
    
    # Timing
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    
    # Metrics
    messages_count = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    total_latency_ms = Column(Float, default=0)
    
    # Outcome
    resolved = Column(Boolean, default=False)
    satisfaction_score = Column(Integer, nullable=True)
    feedback_text = Column(Text, nullable=True)
    
    # Context
    initial_query = Column(Text, nullable=True)
    intent_classification = Column(String(100), nullable=True)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat message model"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    
    # Timing
    created_at = Column(DateTime, server_default=func.now())
    latency_ms = Column(Float, nullable=True)
    
    # Model info
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cached = Column(Boolean, default=False)
    
    # RAG context
    context_used = Column(JSON, nullable=True)  # Sources used for this response
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class KnowledgeArticle(Base):
    """Knowledge base article model"""
    __tablename__ = "knowledge_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String(64), unique=True, index=True)
    
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Vector indexing
    indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime, nullable=True)
    chunks_count = Column(Integer, default=0)
    
    # Metadata
    source = Column(String(50), default="whmcs")  # 'whmcs', 'manual', 'import'
    source_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class AuditLog(Base):
    """Audit log model for compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    
    action = Column(String(50), nullable=False, index=True)
    client_id = Column(Integer, nullable=True, index=True)
    user_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    success = Column(Boolean, default=True)
    error = Column(Text, nullable=True)
    
    details = Column(JSON, nullable=True)


class ModelMetrics(Base):
    """Daily model performance metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(10), index=True)  # YYYY-MM-DD
    model = Column(String(100), index=True)
    provider = Column(String(50))
    
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    total_tokens = Column(Integer, default=0)
    total_latency_ms = Column(Float, default=0)
    
    avg_latency_ms = Column(Float, nullable=True)
    p50_latency_ms = Column(Float, nullable=True)
    p95_latency_ms = Column(Float, nullable=True)
    p99_latency_ms = Column(Float, nullable=True)
    
    cache_hits = Column(Integer, default=0)
    
    class Config:
        # Unique constraint on date + model
        pass
