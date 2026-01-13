"""
Enhanced Chat API Routes
Provides multi-model support, RAG integration, and advanced features.
"""
import uuid
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from config import get_settings, ModelProvider, MODEL_CONFIGS
from services.model_provider import get_model_provider, ModelResponse
from services.rag_service import get_rag_service, RAGContext
from services.whmcs_service import get_whmcs_service
from services.security import (
    get_rate_limiter, 
    get_response_cache, 
    get_audit_logger,
    get_pii_detector,
    AuditAction,
    RateLimitResult
)
from services.analytics import get_analytics_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


# ========== Request/Response Models ==========

class ChatRequest(BaseModel):
    """Enhanced chat request"""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    client_id: Optional[int] = None
    
    # Model settings
    model: Optional[str] = Field(default=None, description="Model to use (e.g., llama3.2, mistral)")
    provider: Optional[str] = Field(default=None, description="Provider to use (ollama, openwebui, openrouter)")
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    
    # RAG settings
    use_rag: bool = Field(default=True, description="Use RAG for context")
    include_knowledge_base: bool = True
    include_tickets: bool = True
    
    # Conversation
    conversation_history: Optional[List[dict]] = None
    system_prompt: Optional[str] = None
    
    # Options
    stream: bool = False
    use_cache: bool = True


class ChatResponse(BaseModel):
    """Enhanced chat response"""
    response: str
    session_id: str
    model: str
    provider: str
    
    # Metrics
    tokens_used: int
    latency_ms: float
    cached: bool = False
    
    # RAG info
    sources_used: List[str] = []
    context_documents: int = 0
    
    # Intent (if classified)
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None


class FeedbackRequest(BaseModel):
    """Feedback submission request"""
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    provider: str
    available: bool
    max_tokens: int
    use_for_code: bool
    use_for_support: bool


# ========== Dependencies ==========

async def check_rate_limit(request: Request) -> RateLimitResult:
    """Rate limiting dependency"""
    rate_limiter = get_rate_limiter()
    
    # Get client identifier (IP or client_id)
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        result = await rate_limiter.check_rate_limit(client_ip)
        
        if not result.allowed:
            # Log rate limit hit
            audit = get_audit_logger()
            await audit.log(
                AuditAction.RATE_LIMIT_HIT,
                {"identifier": client_ip},
                user_ip=client_ip
            )
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": result.retry_after,
                    "reset_at": result.reset_at.isoformat()
                },
                headers={"Retry-After": str(result.retry_after)}
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")
        # Allow request if rate limiting fails
        return RateLimitResult(
            allowed=True,
            remaining=999,
            reset_at=None
        )


# ========== Endpoints ==========

@router.post("/chat/enhanced", response_model=ChatResponse)
async def enhanced_chat(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    rate_limit: RateLimitResult = Depends(check_rate_limit)
):
    """
    Enhanced chat endpoint with multi-model support and RAG.
    
    Features:
    - Multiple AI model backends (Ollama, Open-WebUI, OpenRouter)
    - Automatic fallback chain
    - RAG with Qdrant vector search
    - Response caching
    - PII detection and redaction
    - Rate limiting
    """
    settings = get_settings()
    model_provider = get_model_provider()
    cache = get_response_cache()
    pii_detector = get_pii_detector()
    audit = get_audit_logger()
    analytics = get_analytics_service()
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())
    client_ip = http_request.client.host if http_request.client else None
    
    # Start chat session if new
    if not request.session_id:
        await analytics.record_chat_start(
            session_id=session_id,
            client_id=request.client_id,
            model=request.model,
            provider=request.provider
        )
    
    # PII detection on input
    if settings.pii_detection_enabled:
        pii_matches = pii_detector.detect(request.message)
        if pii_matches:
            # Log PII detection
            await audit.log(
                AuditAction.PII_DETECTED,
                {
                    "pii_types": list(set(m.type for m in pii_matches)),
                    "count": len(pii_matches)
                },
                client_id=request.client_id,
                user_ip=client_ip
            )
            
            # Optionally redact PII in stored messages
            if settings.pii_redaction_enabled:
                request.message, _ = pii_detector.redact(request.message)
    
    # Check cache if enabled
    if request.use_cache:
        cached_response = await cache.get(
            query=request.message,
            model=request.model or settings.ollama_default_model
        )
        
        if cached_response:
            # Record cached response
            await analytics.record_message(
                session_id=session_id,
                model=cached_response.get("model", settings.ollama_default_model),
                provider=ModelProvider.OLLAMA.value,
                tokens_used=0,
                latency_ms=0,
                cached=True
            )
            
            return ChatResponse(
                response=cached_response["response"],
                session_id=session_id,
                model=cached_response.get("model", settings.ollama_default_model),
                provider="cache",
                tokens_used=0,
                latency_ms=0,
                cached=True,
                sources_used=[],
                context_documents=0
            )
    
    # Build RAG context if enabled
    rag_context: Optional[RAGContext] = None
    context_prompt = None
    
    if request.use_rag:
        try:
            rag_service = get_rag_service()
            rag_context = await rag_service.build_context(
                query=request.message,
                include_knowledge_base=request.include_knowledge_base,
                include_tickets=request.include_tickets,
                client_id=str(request.client_id) if request.client_id else None
            )
            
            if rag_context.relevant_docs:
                context_prompt = "\n\n---\nRelevant Context:\n"
                for i, doc in enumerate(rag_context.relevant_docs, 1):
                    source_info = f"[{doc.source_type}"
                    if doc.metadata.get("title"):
                        source_info += f": {doc.metadata['title']}"
                    source_info += f"]"
                    context_prompt += f"\n{i}. {source_info}\n{doc.content[:500]}...\n"
                    
        except Exception as e:
            logger.warning(f"RAG context building failed: {e}")
    
    # Prepare system prompt
    system_prompt = request.system_prompt or """You are a helpful AI support assistant for a web hosting company.
Be concise, professional, and accurate in your responses.
If you're unsure about something, say so rather than guessing."""
    
    if context_prompt:
        system_prompt += context_prompt
    
    # Inject WHMCS client context if available
    if request.client_id:
        try:
            whmcs = get_whmcs_service()
            client_context = await whmcs.build_client_context(request.client_id)
            if client_context.get("summary"):
                system_prompt += f"\n\nClient Context:\n{client_context['summary']}"
        except Exception as e:
            logger.warning(f"Failed to get client context: {e}")
    
    # Prepare messages
    messages = request.conversation_history or []
    messages.append({"role": "user", "content": request.message})
    
    # Determine provider
    provider = None
    if request.provider:
        try:
            provider = ModelProvider(request.provider)
        except ValueError:
            pass
    
    # Generate response
    try:
        model_response: ModelResponse = await model_provider.generate(
            messages=messages,
            model=request.model,
            provider=provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=system_prompt,
            use_fallback=True
        )
        
        # PII redaction on output
        response_text = model_response.content
        if settings.pii_redaction_enabled:
            response_text, redacted = pii_detector.redact(response_text)
            if redacted:
                await audit.log(
                    AuditAction.PII_REDACTED,
                    {"pii_types": list(set(m.type for m in redacted))},
                    client_id=request.client_id
                )
        
        # Cache the response
        if request.use_cache:
            background_tasks.add_task(
                cache.set,
                query=request.message,
                model=model_response.model,
                response=response_text
            )
        
        # Record analytics
        await analytics.record_message(
            session_id=session_id,
            model=model_response.model,
            provider=model_response.provider.value,
            tokens_used=model_response.tokens_used,
            latency_ms=model_response.latency_ms,
            cached=False
        )
        
        # Audit log
        await audit.log(
            AuditAction.CHAT_RESPONSE,
            {
                "model": model_response.model,
                "provider": model_response.provider.value,
                "tokens": model_response.tokens_used,
                "latency_ms": model_response.latency_ms
            },
            client_id=request.client_id,
            user_ip=client_ip
        )
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            model=model_response.model,
            provider=model_response.provider.value,
            tokens_used=model_response.tokens_used,
            latency_ms=model_response.latency_ms,
            cached=False,
            sources_used=rag_context.sources_used if rag_context else [],
            context_documents=len(rag_context.relevant_docs) if rag_context else 0
        )
        
    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        
        await audit.log(
            AuditAction.CHAT_RESPONSE,
            {"error": str(e)},
            client_id=request.client_id,
            user_ip=client_ip,
            success=False,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/chat/enhanced/stream")
async def enhanced_chat_stream(
    request: ChatRequest,
    http_request: Request,
    rate_limit: RateLimitResult = Depends(check_rate_limit)
):
    """
    Streaming chat endpoint for real-time responses.
    Only available with Ollama provider.
    """
    model_provider = get_model_provider()
    analytics = get_analytics_service()
    
    session_id = request.session_id or str(uuid.uuid4())
    
    # Start session
    if not request.session_id:
        await analytics.record_chat_start(
            session_id=session_id,
            client_id=request.client_id,
            model=request.model
        )
    
    async def generate():
        messages = request.conversation_history or []
        messages.append({"role": "user", "content": request.message})
        
        full_response = ""
        
        async for chunk in model_provider.generate_stream(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        ):
            full_response += chunk
            yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
        
        # Send completion event
        yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'full_response': full_response})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.post("/chat/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    http_request: Request
):
    """Submit feedback for a chat session"""
    analytics = get_analytics_service()
    audit = get_audit_logger()
    
    await analytics.record_feedback(
        session_id=request.session_id,
        rating=request.rating,
        feedback_text=request.feedback_text
    )
    
    await audit.log(
        AuditAction.CUSTOMER_FEEDBACK,
        {
            "session_id": request.session_id,
            "rating": request.rating,
            "has_text": bool(request.feedback_text)
        },
        user_ip=http_request.client.host if http_request.client else None
    )
    
    return {"status": "success", "message": "Feedback recorded"}


@router.post("/chat/end")
async def end_chat_session(
    session_id: str,
    resolved: bool = False,
    http_request: Request = None
):
    """End a chat session"""
    analytics = get_analytics_service()
    
    await analytics.record_chat_end(
        session_id=session_id,
        resolved=resolved
    )
    
    return {"status": "success", "session_id": session_id}


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available AI models"""
    model_provider = get_model_provider()
    health = await model_provider.check_health()
    
    models = []
    
    for name, config in MODEL_CONFIGS.items():
        provider_health = health.get(config.provider, None)
        available = provider_health.available if provider_health else False
        
        # Check if specific model is available
        if provider_health and provider_health.available:
            if config.provider == ModelProvider.OLLAMA:
                available = name in provider_health.models or any(
                    name in m for m in provider_health.models
                )
        
        models.append(ModelInfo(
            name=name,
            provider=config.provider.value,
            available=available,
            max_tokens=config.max_tokens,
            use_for_code=config.use_for_code,
            use_for_support=config.use_for_support
        ))
    
    return models


@router.get("/models/health")
async def check_model_health():
    """Check health of all model providers"""
    model_provider = get_model_provider()
    health = await model_provider.check_health()
    
    return {
        provider.value: {
            "available": h.available,
            "latency_ms": h.latency_ms,
            "models": h.models,
            "error": h.error
        }
        for provider, h in health.items()
    }
