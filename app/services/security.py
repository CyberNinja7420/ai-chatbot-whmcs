"""
Security Service
Handles rate limiting, request caching, audit logging, and PII detection/redaction.
"""
import asyncio
import hashlib
import json
import logging
import re
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
import redis.asyncio as redis
from enum import Enum

from config import get_settings

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Types of audit actions"""
    CHAT_REQUEST = "chat_request"
    CHAT_RESPONSE = "chat_response"
    TICKET_CREATE = "ticket_create"
    TICKET_UPDATE = "ticket_update"
    TICKET_REPLY = "ticket_reply"
    CLIENT_LOOKUP = "client_lookup"
    INVOICE_QUERY = "invoice_query"
    SERVICE_QUERY = "service_query"
    WEBHOOK_RECEIVED = "webhook_received"
    WORKFLOW_TRIGGERED = "workflow_triggered"
    RATE_LIMIT_HIT = "rate_limit_hit"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    PII_DETECTED = "pii_detected"
    PII_REDACTED = "pii_redacted"
    ADMIN_ACTION = "admin_action"


@dataclass
class AuditEntry:
    """An audit log entry"""
    timestamp: datetime
    action: AuditAction
    client_id: Optional[int]
    user_ip: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None


@dataclass
class PIIMatch:
    """A detected PII match"""
    type: str
    value: str
    start: int
    end: int
    confidence: float


class PIIDetector:
    """
    PII (Personally Identifiable Information) detection and redaction.
    Detects and optionally redacts sensitive information.
    """
    
    # PII patterns with named groups
    PATTERNS = {
        "email": (
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            0.95
        ),
        "phone_us": (
            r'\b(?:\+1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b',
            0.85
        ),
        "phone_intl": (
            r'\b\+[1-9]\d{1,14}\b',
            0.80
        ),
        "ssn": (
            r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b',
            0.90
        ),
        "credit_card": (
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            0.95
        ),
        "ip_address": (
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            0.70  # Lower confidence as IPs may be intentional
        ),
        "password_mention": (
            r'(?i)(?:password|passwd|pwd)[\s:=]+[^\s]{4,}',
            0.80
        ),
        "api_key": (
            r'(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token)[\s:=]+[a-zA-Z0-9_-]{16,}',
            0.85
        ),
        "date_of_birth": (
            r'\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
            0.75
        ),
    }
    
    # Redaction templates
    REDACTION_TEMPLATES = {
        "email": "[EMAIL REDACTED]",
        "phone_us": "[PHONE REDACTED]",
        "phone_intl": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
        "ip_address": "[IP REDACTED]",
        "password_mention": "[PASSWORD REDACTED]",
        "api_key": "[API KEY REDACTED]",
        "date_of_birth": "[DOB REDACTED]",
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compiled_patterns = {
            name: re.compile(pattern)
            for name, (pattern, _) in self.PATTERNS.items()
        }
    
    def detect(self, text: str, min_confidence: float = 0.7) -> List[PIIMatch]:
        """
        Detect PII in text.
        Returns list of matches sorted by position.
        """
        if not self.enabled:
            return []
        
        matches = []
        
        for pii_type, compiled in self._compiled_patterns.items():
            _, confidence = self.PATTERNS[pii_type]
            
            if confidence < min_confidence:
                continue
            
            for match in compiled.finditer(text):
                matches.append(PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence
                ))
        
        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches
    
    def redact(
        self, 
        text: str, 
        pii_types: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        mask_partial: bool = False
    ) -> Tuple[str, List[PIIMatch]]:
        """
        Redact PII from text.
        Returns redacted text and list of what was redacted.
        """
        if not self.enabled:
            return text, []
        
        matches = self.detect(text, min_confidence)
        
        if pii_types:
            matches = [m for m in matches if m.type in pii_types]
        
        if not matches:
            return text, []
        
        # Build redacted text (work backwards to preserve positions)
        redacted = text
        for match in reversed(matches):
            if mask_partial and match.type in ["email", "credit_card"]:
                # Partial mask: show first/last few chars
                value = match.value
                if match.type == "email":
                    parts = value.split("@")
                    masked = f"{parts[0][:2]}***@{parts[1]}"
                elif match.type == "credit_card":
                    masked = f"****-****-****-{value[-4:]}"
                else:
                    masked = self.REDACTION_TEMPLATES.get(match.type, "[REDACTED]")
            else:
                masked = self.REDACTION_TEMPLATES.get(match.type, "[REDACTED]")
            
            redacted = redacted[:match.start] + masked + redacted[match.end:]
        
        return redacted, matches
    
    def contains_pii(self, text: str, min_confidence: float = 0.7) -> bool:
        """Quick check if text contains any PII"""
        return len(self.detect(text, min_confidence)) > 0


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    Supports per-client and per-IP rate limiting.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis
    
    def _get_key(self, identifier: str, limit_type: str = "default") -> str:
        """Generate rate limit key"""
        return f"ratelimit:{limit_type}:{identifier}"
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        limit_type: str = "default"
    ) -> RateLimitResult:
        """
        Check if request is within rate limit.
        Uses sliding window algorithm.
        """
        r = await self._get_redis()
        
        max_requests = limit or self.settings.rate_limit_requests
        window_seconds = window or self.settings.rate_limit_window
        
        key = self._get_key(identifier, limit_type)
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        pipe = r.pipeline()
        
        # Remove old entries outside window
        pipe.zremrangebyscore(key, "-inf", window_start.timestamp())
        
        # Count current entries
        pipe.zcard(key)
        
        # Get oldest entry in window
        pipe.zrange(key, 0, 0, withscores=True)
        
        results = await pipe.execute()
        current_count = results[1]
        
        reset_at = now + timedelta(seconds=window_seconds)
        
        if current_count >= max_requests:
            # Rate limited
            oldest = results[2]
            if oldest:
                oldest_time = datetime.fromtimestamp(oldest[0][1])
                retry_after = int((oldest_time + timedelta(seconds=window_seconds) - now).total_seconds())
                retry_after = max(1, retry_after)
            else:
                retry_after = window_seconds
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Add this request
        await r.zadd(key, {f"{now.timestamp()}": now.timestamp()})
        await r.expire(key, window_seconds + 1)
        
        return RateLimitResult(
            allowed=True,
            remaining=max_requests - current_count - 1,
            reset_at=reset_at
        )
    
    async def get_usage(self, identifier: str, limit_type: str = "default") -> Dict[str, Any]:
        """Get current rate limit usage for an identifier"""
        r = await self._get_redis()
        
        key = self._get_key(identifier, limit_type)
        window_seconds = self.settings.rate_limit_window
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Clean and count
        await r.zremrangebyscore(key, "-inf", window_start.timestamp())
        current_count = await r.zcard(key)
        
        return {
            "identifier": identifier,
            "current_requests": current_count,
            "max_requests": self.settings.rate_limit_requests,
            "window_seconds": window_seconds,
            "remaining": max(0, self.settings.rate_limit_requests - current_count)
        }


class ResponseCache:
    """
    Redis-based response cache for common queries.
    Caches AI responses to reduce latency and costs.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis
    
    def _generate_cache_key(self, query: str, model: str, context_hash: Optional[str] = None) -> str:
        """Generate cache key from query parameters"""
        # Normalize query
        normalized = query.lower().strip()
        query_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
        
        key = f"cache:response:{model}:{query_hash}"
        if context_hash:
            key = f"{key}:{context_hash}"
        
        return key
    
    async def get(
        self, 
        query: str, 
        model: str,
        context_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        r = await self._get_redis()
        key = self._generate_cache_key(query, model, context_hash)
        
        try:
            cached = await r.get(key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                logger.debug(f"Cache hit for key {key}")
                return data
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def set(
        self,
        query: str,
        model: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_hash: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache a response"""
        r = await self._get_redis()
        key = self._generate_cache_key(query, model, context_hash)
        
        try:
            data = {
                "response": response,
                "model": model,
                "cached_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            await r.setex(
                key,
                ttl or self.settings.cache_ttl,
                json.dumps(data)
            )
            logger.debug(f"Cached response with key {key}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def invalidate(
        self,
        pattern: Optional[str] = None,
        model: Optional[str] = None
    ) -> int:
        """Invalidate cached responses matching pattern"""
        r = await self._get_redis()
        
        if pattern:
            search_pattern = f"cache:response:*{pattern}*"
        elif model:
            search_pattern = f"cache:response:{model}:*"
        else:
            search_pattern = "cache:response:*"
        
        cursor = 0
        deleted = 0
        
        while True:
            cursor, keys = await r.scan(cursor, match=search_pattern, count=100)
            if keys:
                deleted += await r.delete(*keys)
            if cursor == 0:
                break
        
        logger.info(f"Invalidated {deleted} cached responses")
        return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        r = await self._get_redis()
        
        # Count cached entries
        cursor = 0
        count = 0
        
        while True:
            cursor, keys = await r.scan(cursor, match="cache:response:*", count=100)
            count += len(keys)
            if cursor == 0:
                break
        
        return {
            "cached_responses": count,
            "ttl_seconds": self.settings.cache_ttl
        }


class AuditLogger:
    """
    Audit logging service for compliance and monitoring.
    Logs security-relevant events to database and optionally to external systems.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._redis: Optional[redis.Redis] = None
        self._buffer: List[AuditEntry] = []
        self._buffer_size = 100
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection for real-time audit stream"""
        if self._redis is None:
            self._redis = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis
    
    async def log(
        self,
        action: AuditAction,
        details: Dict[str, Any],
        client_id: Optional[int] = None,
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log an audit event"""
        if not self.settings.audit_log_enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            action=action,
            client_id=client_id,
            user_ip=user_ip,
            user_agent=user_agent,
            details=details,
            success=success,
            error=error
        )
        
        # Add to buffer
        self._buffer.append(entry)
        
        # Log to Redis stream for real-time monitoring
        try:
            r = await self._get_redis()
            await r.xadd(
                "audit:stream",
                {
                    "action": action.value,
                    "timestamp": entry.timestamp.isoformat(),
                    "client_id": str(client_id) if client_id else "",
                    "user_ip": user_ip or "",
                    "success": str(success),
                    "details": json.dumps(details)[:1000]  # Limit size
                },
                maxlen=10000  # Keep last 10k entries
            )
        except Exception as e:
            logger.warning(f"Failed to write audit to Redis: {e}")
        
        # Flush buffer if full
        if len(self._buffer) >= self._buffer_size:
            await self._flush_buffer()
        
        # Log to standard logger for file-based logging
        log_msg = f"AUDIT: {action.value} | client={client_id} | ip={user_ip} | success={success}"
        if success:
            logger.info(log_msg)
        else:
            logger.warning(f"{log_msg} | error={error}")
    
    async def _flush_buffer(self) -> None:
        """Flush audit buffer to persistent storage"""
        if not self._buffer:
            return
        
        # In production, this would write to database
        # For now, just clear the buffer
        entries = self._buffer.copy()
        self._buffer.clear()
        
        logger.debug(f"Flushed {len(entries)} audit entries")
    
    async def get_recent_events(
        self,
        action: Optional[AuditAction] = None,
        client_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent audit events from Redis stream"""
        try:
            r = await self._get_redis()
            
            # Read from stream
            entries = await r.xrevrange("audit:stream", count=limit * 2)
            
            results = []
            for entry_id, data in entries:
                # Filter by action if specified
                if action and data.get("action") != action.value:
                    continue
                
                # Filter by client_id if specified
                if client_id and data.get("client_id") != str(client_id):
                    continue
                
                results.append({
                    "id": entry_id,
                    "action": data.get("action"),
                    "timestamp": data.get("timestamp"),
                    "client_id": data.get("client_id") or None,
                    "user_ip": data.get("user_ip") or None,
                    "success": data.get("success") == "True",
                    "details": json.loads(data.get("details", "{}"))
                })
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get audit events: {e}")
            return []


# Singleton instances
_pii_detector: Optional[PIIDetector] = None
_rate_limiter: Optional[RateLimiter] = None
_response_cache: Optional[ResponseCache] = None
_audit_logger: Optional[AuditLogger] = None


def get_pii_detector() -> PIIDetector:
    """Get PII detector singleton"""
    global _pii_detector
    if _pii_detector is None:
        settings = get_settings()
        _pii_detector = PIIDetector(enabled=settings.pii_detection_enabled)
    return _pii_detector


def get_rate_limiter() -> RateLimiter:
    """Get rate limiter singleton"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_response_cache() -> ResponseCache:
    """Get response cache singleton"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def get_audit_logger() -> AuditLogger:
    """Get audit logger singleton"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
