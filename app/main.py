"""
WHMCS AI Chatbot - Main Application Entry Point

Features:
- Multi-Model Support (Ollama, Open-WebUI, OpenRouter)
- RAG Integration with Qdrant
- WHMCS Deep Integration
- n8n Workflow Automation
- Admin Dashboard with Analytics
- Security (Rate Limiting, Caching, Audit Logging, PII Detection)
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routes
from routes import tickets, chatbot
from routes.chat_enhanced import router as chat_enhanced_router
from routes.admin import router as admin_router
from routes.webhooks import router as webhooks_router

# Import config
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events"""
    # Startup
    logger.info("Starting WHMCS AI Chatbot...")
    settings = get_settings()
    logger.info(f"Ollama endpoint: {settings.ollama_base_url}")
    logger.info(f"Qdrant endpoint: {settings.qdrant_url}")
    
    # Initialize RAG collections
    try:
        from services.rag_service import get_rag_service
        rag = get_rag_service()
        await rag.initialize_collections()
        logger.info("RAG collections initialized")
    except Exception as e:
        logger.warning(f"RAG initialization failed (will retry on first use): {e}")
    
    # Check model provider health
    try:
        from services.model_provider import get_model_provider
        provider = get_model_provider()
        health = await provider.check_health()
        for name, status in health.items():
            logger.info(f"Model provider {name.value}: {'available' if status.available else 'unavailable'}")
    except Exception as e:
        logger.warning(f"Model health check failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down WHMCS AI Chatbot...")


# Create FastAPI application
app = FastAPI(
    title="WHMCS AI Chatbot",
    description="""
    AI-powered chatbot for WHMCS support with multi-model support and RAG integration.
    
    ## Features
    
    - **Multi-Model Support**: Ollama (local GPU), Open-WebUI, OpenRouter with automatic fallback
    - **RAG Integration**: Semantic search over knowledge base and ticket history
    - **WHMCS Integration**: Deep integration with tickets, clients, invoices, and services
    - **Workflow Automation**: n8n integration for escalation and notifications
    - **Analytics Dashboard**: Performance metrics and customer satisfaction tracking
    - **Security**: Rate limiting, response caching, audit logging, PII detection
    """,
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred", "type": type(exc).__name__}
    )


# Include API Routes
# Legacy routes
app.include_router(tickets.router, prefix="/api", tags=["WHMCS Tickets (Legacy)"])
app.include_router(chatbot.router, prefix="/api", tags=["Chat (Legacy)"])

# Enhanced routes
app.include_router(chat_enhanced_router, prefix="/api/v2", tags=["Chat (Enhanced)"])
app.include_router(admin_router, prefix="/api", tags=["Admin Dashboard"])
app.include_router(webhooks_router, prefix="/api", tags=["Webhooks"])


@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "name": "WHMCS AI Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "multi-model-support",
            "rag-integration",
            "whmcs-integration",
            "n8n-workflows",
            "admin-dashboard",
            "security-features"
        ]
    }


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/v2/info")
async def api_info():
    """Get API information and available endpoints"""
    settings = get_settings()
    
    return {
        "version": "2.0.0",
        "endpoints": {
            "chat": {
                "enhanced": "/api/v2/chat/enhanced",
                "stream": "/api/v2/chat/enhanced/stream",
                "feedback": "/api/v2/chat/feedback",
                "models": "/api/v2/models",
                "health": "/api/v2/models/health"
            },
            "admin": {
                "dashboard": "/api/admin/dashboard/summary",
                "daily_stats": "/api/admin/dashboard/daily-stats",
                "model_performance": "/api/admin/models/performance",
                "rag_stats": "/api/admin/rag/stats",
                "audit": "/api/admin/audit/recent",
                "system_health": "/api/admin/health"
            },
            "webhooks": {
                "ticket": "/api/webhooks/whmcs/ticket",
                "client": "/api/webhooks/whmcs/client",
                "invoice": "/api/webhooks/whmcs/invoice",
                "service": "/api/webhooks/whmcs/service"
            },
            "legacy": {
                "chat": "/api/chat",
                "tickets": "/api/tickets"
            }
        },
        "models": {
            "default": settings.ollama_default_model,
            "available": ["llama3.2", "mistral", "dolphin-mixtral", "qwen2.5-coder"]
        },
        "providers": {
            "primary": "ollama",
            "fallback": ["openwebui", "openrouter"]
        }
    }
