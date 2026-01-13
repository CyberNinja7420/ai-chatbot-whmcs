"""
Admin Dashboard API Routes
Provides analytics, metrics, and management endpoints.
"""
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from pydantic import BaseModel, Field

from config import get_settings
from services.analytics import get_analytics_service
from services.security import get_audit_logger, get_rate_limiter, get_response_cache, AuditAction
from services.rag_service import get_rag_service
from services.model_provider import get_model_provider
from services.whmcs_service import get_whmcs_service
from services.n8n_service import get_n8n_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


# ========== Request/Response Models ==========

class DashboardSummary(BaseModel):
    """Dashboard summary data"""
    today: dict
    trends: dict
    weekly: dict
    model_performance: dict
    recent_feedback: list
    peak_hour: int
    model_usage_today: dict


class ModelPerformanceResponse(BaseModel):
    """Model performance metrics"""
    model: str
    total_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class SystemHealthResponse(BaseModel):
    """System health status"""
    status: str
    components: dict
    timestamp: str


class IndexingRequest(BaseModel):
    """Request to index content"""
    content: str
    title: str
    source_type: str = "knowledge_base"
    category: Optional[str] = None
    tags: Optional[List[str]] = None


# ========== Dashboard Endpoints ==========

@router.get("/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary():
    """
    Get comprehensive dashboard summary.
    
    Returns:
    - Today's metrics (chats, messages, satisfaction)
    - Trends compared to yesterday
    - Weekly aggregates
    - Model performance comparison
    - Recent feedback
    """
    analytics = get_analytics_service()
    
    try:
        summary = await analytics.get_dashboard_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/daily-stats")
async def get_daily_stats(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Get detailed statistics for a specific day"""
    analytics = get_analytics_service()
    
    try:
        stats = await analytics.get_daily_stats(date)
        return stats
    except Exception as e:
        logger.error(f"Failed to get daily stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/response-time-trend")
async def get_response_time_trend(
    days: int = Query(7, ge=1, le=30)
):
    """Get response time trend over specified days"""
    analytics = get_analytics_service()
    
    try:
        trend = await analytics.get_response_time_trend(days)
        return {"trend": trend}
    except Exception as e:
        logger.error(f"Failed to get response time trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/satisfaction-distribution")
async def get_satisfaction_distribution(
    days: int = Query(30, ge=1, le=90)
):
    """Get distribution of satisfaction scores"""
    analytics = get_analytics_service()
    
    try:
        distribution = await analytics.get_satisfaction_distribution(days)
        return {"distribution": distribution, "days": days}
    except Exception as e:
        logger.error(f"Failed to get satisfaction distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Model Performance Endpoints ==========

@router.get("/models/performance")
async def get_models_performance(
    days: int = Query(7, ge=1, le=30)
):
    """Get performance metrics for all models"""
    analytics = get_analytics_service()
    
    models = ["llama3.2", "mistral", "dolphin-mixtral", "qwen2.5-coder"]
    performance = {}
    
    for model in models:
        try:
            perf = await analytics.get_model_performance(model, days)
            if perf.get("total_requests", 0) > 0:
                performance[model] = perf
        except Exception as e:
            logger.warning(f"Failed to get performance for {model}: {e}")
    
    return {"performance": performance, "days": days}


@router.get("/models/comparison")
async def compare_models():
    """
    Compare performance across all models.
    Useful for model selection decisions.
    """
    model_provider = get_model_provider()
    analytics = get_analytics_service()
    
    # Get health status
    health = await model_provider.check_health()
    
    comparison = []
    models = ["llama3.2", "mistral", "dolphin-mixtral", "qwen2.5-coder"]
    
    for model in models:
        perf = await analytics.get_model_performance(model, days=7)
        
        comparison.append({
            "model": model,
            "available": any(
                h.available and (model in h.models or any(model in m for m in h.models))
                for h in health.values()
            ),
            "requests_7d": perf.get("total_requests", 0),
            "avg_latency_ms": perf.get("avg_latency_ms", 0),
            "p95_latency_ms": perf.get("p95_latency_ms", 0),
            "recommended_for": "code" if model == "qwen2.5-coder" else "support"
        })
    
    return {"comparison": comparison}


# ========== RAG & Knowledge Base Endpoints ==========

@router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG/Vector DB statistics"""
    rag_service = get_rag_service()
    
    try:
        stats = await rag_service.get_collection_stats()
        return {"collections": stats}
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/index")
async def index_content(request: IndexingRequest):
    """Index new content into the RAG system"""
    rag_service = get_rag_service()
    audit = get_audit_logger()
    
    try:
        if request.source_type == "knowledge_base":
            doc_ids = await rag_service.index_knowledge_article(
                article_id=f"manual_{datetime.utcnow().timestamp()}",
                title=request.title,
                content=request.content,
                category=request.category,
                tags=request.tags
            )
        else:
            doc_ids = await rag_service.index_document(
                content=request.content,
                source_type=request.source_type,
                metadata={
                    "title": request.title,
                    "category": request.category,
                    "tags": request.tags
                }
            )
        
        await audit.log(
            AuditAction.ADMIN_ACTION,
            {
                "action": "index_content",
                "title": request.title,
                "source_type": request.source_type,
                "chunks_indexed": len(doc_ids)
            }
        )
        
        return {
            "status": "success",
            "indexed_chunks": len(doc_ids),
            "document_ids": doc_ids
        }
        
    except Exception as e:
        logger.error(f"Failed to index content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/sync-whmcs")
async def sync_whmcs_knowledge_base():
    """
    Sync WHMCS knowledge base articles to RAG.
    This would typically be called periodically.
    """
    # This would integrate with WHMCS KnowledgeBase API
    # For now, return a placeholder
    return {
        "status": "queued",
        "message": "Knowledge base sync has been queued"
    }


# ========== Audit & Security Endpoints ==========

@router.get("/audit/recent")
async def get_recent_audit_events(
    action: Optional[str] = None,
    client_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=500)
):
    """Get recent audit events"""
    audit = get_audit_logger()
    
    try:
        action_enum = AuditAction(action) if action else None
    except ValueError:
        action_enum = None
    
    events = await audit.get_recent_events(
        action=action_enum,
        client_id=client_id,
        limit=limit
    )
    
    return {"events": events, "count": len(events)}


@router.get("/rate-limits/usage")
async def get_rate_limit_usage(identifier: str):
    """Get rate limit usage for a specific identifier"""
    rate_limiter = get_rate_limiter()
    
    try:
        usage = await rate_limiter.get_usage(identifier)
        return usage
    except Exception as e:
        logger.error(f"Failed to get rate limit usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """Get response cache statistics"""
    cache = get_response_cache()
    
    try:
        stats = await cache.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/invalidate")
async def invalidate_cache(
    pattern: Optional[str] = None,
    model: Optional[str] = None
):
    """Invalidate cached responses"""
    cache = get_response_cache()
    audit = get_audit_logger()
    
    try:
        deleted = await cache.invalidate(pattern=pattern, model=model)
        
        await audit.log(
            AuditAction.ADMIN_ACTION,
            {
                "action": "cache_invalidate",
                "pattern": pattern,
                "model": model,
                "deleted_count": deleted
            }
        )
        
        return {"status": "success", "invalidated_count": deleted}
        
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== System Health Endpoints ==========

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """
    Get comprehensive system health status.
    Checks all integrated services.
    """
    settings = get_settings()
    model_provider = get_model_provider()
    
    components = {}
    overall_status = "healthy"
    
    # Check model providers
    try:
        health = await model_provider.check_health()
        for provider, h in health.items():
            components[f"model_{provider.value}"] = {
                "status": "healthy" if h.available else "unhealthy",
                "latency_ms": h.latency_ms,
                "details": {"models": h.models, "error": h.error}
            }
            if not h.available and provider.value == "ollama":
                overall_status = "degraded"
    except Exception as e:
        components["model_providers"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"
    
    # Check Qdrant
    try:
        rag_service = get_rag_service()
        stats = await rag_service.get_collection_stats()
        qdrant_healthy = all(
            "error" not in v for v in stats.values()
        )
        components["qdrant"] = {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "collections": stats
        }
        if not qdrant_healthy:
            overall_status = "degraded"
    except Exception as e:
        components["qdrant"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"
    
    # Check Redis (via rate limiter)
    try:
        rate_limiter = get_rate_limiter()
        await rate_limiter.get_usage("health_check")
        components["redis"] = {"status": "healthy"}
    except Exception as e:
        components["redis"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"
    
    return SystemHealthResponse(
        status=overall_status,
        components=components,
        timestamp=datetime.utcnow().isoformat()
    )


# ========== n8n Workflow Endpoints ==========

@router.get("/workflows")
async def list_workflows():
    """List available n8n workflows"""
    n8n = get_n8n_service()
    
    try:
        workflows = await n8n.get_workflows()
        return {"workflows": workflows}
    except Exception as e:
        logger.error(f"Failed to get workflows: {e}")
        return {"workflows": [], "error": str(e)}


@router.get("/workflows/executions")
async def get_workflow_executions(
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
):
    """Get recent workflow executions"""
    n8n = get_n8n_service()
    
    try:
        executions = await n8n.get_workflow_executions(
            workflow_id=workflow_id,
            status=status,
            limit=limit
        )
        return {"executions": executions}
    except Exception as e:
        logger.error(f"Failed to get executions: {e}")
        return {"executions": [], "error": str(e)}


# ========== Configuration Endpoints ==========

@router.get("/config/sla")
async def get_sla_config():
    """Get SLA configuration"""
    n8n = get_n8n_service()
    
    sla_config = {}
    for priority, config in n8n.DEFAULT_SLAS.items():
        sla_config[priority] = {
            "first_response_hours": config.first_response_hours,
            "resolution_hours": config.resolution_hours,
            "escalation_levels": config.escalation_levels
        }
    
    return {"sla_config": sla_config}


@router.get("/config/models")
async def get_model_config():
    """Get model configuration"""
    from config import MODEL_CONFIGS
    
    return {
        "models": {
            name: {
                "provider": config.provider.value,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "context_window": config.context_window,
                "use_for_code": config.use_for_code,
                "use_for_support": config.use_for_support
            }
            for name, config in MODEL_CONFIGS.items()
        }
    }
