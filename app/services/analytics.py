"""
Analytics Service
Provides chat analytics, metrics, model performance tracking,
and customer satisfaction scoring for the admin dashboard.
"""
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import redis.asyncio as redis

from config import get_settings, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class ChatMetrics:
    """Metrics for a chat session"""
    session_id: str
    client_id: Optional[int]
    model_used: str
    provider: str
    messages_count: int
    tokens_used: int
    total_latency_ms: float
    avg_response_time_ms: float
    started_at: datetime
    ended_at: Optional[datetime]
    satisfaction_score: Optional[int]  # 1-5
    resolved: bool


@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model: str
    provider: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tokens: int
    avg_tokens_per_request: float
    cache_hit_rate: float
    error_rate: float


@dataclass
class DailyStats:
    """Daily statistics summary"""
    date: str
    total_chats: int
    total_messages: int
    unique_clients: int
    avg_messages_per_chat: float
    avg_response_time_ms: float
    satisfaction_avg: float
    resolution_rate: float
    model_usage: Dict[str, int]
    peak_hour: int
    total_tokens_used: int


class AnalyticsService:
    """
    Analytics service for tracking chat metrics, model performance,
    and customer satisfaction.
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
    
    # ========== Event Recording ==========
    
    async def record_chat_start(
        self,
        session_id: str,
        client_id: Optional[int] = None,
        model: str = None,
        provider: str = None
    ) -> None:
        """Record start of a chat session"""
        r = await self._get_redis()
        
        session_data = {
            "session_id": session_id,
            "client_id": client_id,
            "model": model or self.settings.ollama_default_model,
            "provider": provider or ModelProvider.OLLAMA.value,
            "started_at": datetime.utcnow().isoformat(),
            "messages_count": 0,
            "tokens_used": 0,
            "total_latency_ms": 0
        }
        
        await r.hset(f"chat:session:{session_id}", mapping=session_data)
        await r.expire(f"chat:session:{session_id}", 86400)  # 24 hours
        
        # Update daily counters
        today = datetime.utcnow().strftime("%Y-%m-%d")
        await r.hincrby(f"stats:daily:{today}", "total_chats", 1)
        
        if client_id:
            await r.sadd(f"stats:daily:{today}:clients", str(client_id))
    
    async def record_message(
        self,
        session_id: str,
        model: str,
        provider: str,
        tokens_used: int,
        latency_ms: float,
        cached: bool = False,
        success: bool = True
    ) -> None:
        """Record a chat message/response"""
        r = await self._get_redis()
        
        # Update session
        pipe = r.pipeline()
        pipe.hincrby(f"chat:session:{session_id}", "messages_count", 1)
        pipe.hincrby(f"chat:session:{session_id}", "tokens_used", tokens_used)
        pipe.hincrbyfloat(f"chat:session:{session_id}", "total_latency_ms", latency_ms)
        await pipe.execute()
        
        # Update daily stats
        today = datetime.utcnow().strftime("%Y-%m-%d")
        hour = datetime.utcnow().hour
        
        pipe = r.pipeline()
        pipe.hincrby(f"stats:daily:{today}", "total_messages", 1)
        pipe.hincrby(f"stats:daily:{today}", "total_tokens", tokens_used)
        pipe.hincrby(f"stats:daily:{today}:hours", str(hour), 1)
        pipe.hincrby(f"stats:daily:{today}:models", model, 1)
        
        if cached:
            pipe.hincrby(f"stats:daily:{today}", "cache_hits", 1)
        
        if not success:
            pipe.hincrby(f"stats:daily:{today}", "errors", 1)
        
        await pipe.execute()
        
        # Record latency for percentile calculations
        await r.zadd(
            f"stats:latency:{today}:{model}",
            {f"{datetime.utcnow().timestamp()}": latency_ms}
        )
        await r.expire(f"stats:latency:{today}:{model}", 172800)  # 48 hours
    
    async def record_chat_end(
        self,
        session_id: str,
        resolved: bool = False,
        satisfaction_score: Optional[int] = None
    ) -> None:
        """Record end of a chat session"""
        r = await self._get_redis()
        
        # Update session
        await r.hset(f"chat:session:{session_id}", mapping={
            "ended_at": datetime.utcnow().isoformat(),
            "resolved": str(resolved),
            "satisfaction_score": str(satisfaction_score) if satisfaction_score else ""
        })
        
        # Update daily stats
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        if resolved:
            await r.hincrby(f"stats:daily:{today}", "resolved_chats", 1)
        
        if satisfaction_score:
            await r.rpush(f"stats:daily:{today}:satisfaction", str(satisfaction_score))
    
    async def record_feedback(
        self,
        session_id: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> None:
        """Record customer feedback"""
        r = await self._get_redis()
        
        feedback_data = {
            "session_id": session_id,
            "rating": rating,
            "feedback_text": feedback_text or "",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await r.hset(f"chat:session:{session_id}", "satisfaction_score", str(rating))
        
        # Store detailed feedback
        await r.lpush("feedback:recent", json.dumps(feedback_data))
        await r.ltrim("feedback:recent", 0, 999)  # Keep last 1000
        
        # Update daily stats
        today = datetime.utcnow().strftime("%Y-%m-%d")
        await r.rpush(f"stats:daily:{today}:satisfaction", str(rating))
    
    # ========== Metrics Retrieval ==========
    
    async def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific session"""
        r = await self._get_redis()
        
        data = await r.hgetall(f"chat:session:{session_id}")
        if not data:
            return None
        
        messages_count = int(data.get("messages_count", 0))
        total_latency = float(data.get("total_latency_ms", 0))
        
        return {
            "session_id": session_id,
            "client_id": data.get("client_id"),
            "model": data.get("model"),
            "provider": data.get("provider"),
            "messages_count": messages_count,
            "tokens_used": int(data.get("tokens_used", 0)),
            "total_latency_ms": total_latency,
            "avg_response_time_ms": total_latency / messages_count if messages_count > 0 else 0,
            "started_at": data.get("started_at"),
            "ended_at": data.get("ended_at"),
            "resolved": data.get("resolved") == "True",
            "satisfaction_score": int(data.get("satisfaction_score")) if data.get("satisfaction_score") else None
        }
    
    async def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific day"""
        r = await self._get_redis()
        
        if not date:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Get main stats
        stats = await r.hgetall(f"stats:daily:{date}")
        
        total_chats = int(stats.get("total_chats", 0))
        total_messages = int(stats.get("total_messages", 0))
        resolved_chats = int(stats.get("resolved_chats", 0))
        cache_hits = int(stats.get("cache_hits", 0))
        errors = int(stats.get("errors", 0))
        
        # Get unique clients count
        unique_clients = await r.scard(f"stats:daily:{date}:clients")
        
        # Get satisfaction scores
        satisfaction_scores = await r.lrange(f"stats:daily:{date}:satisfaction", 0, -1)
        satisfaction_scores = [int(s) for s in satisfaction_scores if s]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
        
        # Get hourly distribution
        hourly = await r.hgetall(f"stats:daily:{date}:hours")
        hourly = {int(k): int(v) for k, v in hourly.items()}
        peak_hour = max(hourly.keys(), key=lambda h: hourly[h]) if hourly else 0
        
        # Get model usage
        model_usage = await r.hgetall(f"stats:daily:{date}:models")
        model_usage = {k: int(v) for k, v in model_usage.items()}
        
        return {
            "date": date,
            "total_chats": total_chats,
            "total_messages": total_messages,
            "unique_clients": unique_clients,
            "avg_messages_per_chat": total_messages / total_chats if total_chats > 0 else 0,
            "total_tokens_used": int(stats.get("total_tokens", 0)),
            "satisfaction_avg": round(avg_satisfaction, 2),
            "satisfaction_count": len(satisfaction_scores),
            "resolution_rate": resolved_chats / total_chats if total_chats > 0 else 0,
            "cache_hit_rate": cache_hits / total_messages if total_messages > 0 else 0,
            "error_rate": errors / total_messages if total_messages > 0 else 0,
            "model_usage": model_usage,
            "hourly_distribution": hourly,
            "peak_hour": peak_hour
        }
    
    async def get_model_performance(
        self,
        model: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        r = await self._get_redis()
        
        all_latencies = []
        total_requests = 0
        
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Get latencies
            latencies = await r.zrange(
                f"stats:latency:{date}:{model}",
                0, -1,
                withscores=True
            )
            
            for _, latency in latencies:
                all_latencies.append(latency)
            
            # Get model usage
            usage = await r.hget(f"stats:daily:{date}:models", model)
            total_requests += int(usage) if usage else 0
        
        if not all_latencies:
            return {
                "model": model,
                "total_requests": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0
            }
        
        # Calculate percentiles
        all_latencies.sort()
        n = len(all_latencies)
        
        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            return data[f] + (k - f) * (data[c] - data[f])
        
        return {
            "model": model,
            "total_requests": total_requests,
            "avg_latency_ms": round(sum(all_latencies) / n, 2),
            "min_latency_ms": round(min(all_latencies), 2),
            "max_latency_ms": round(max(all_latencies), 2),
            "p50_latency_ms": round(percentile(all_latencies, 50), 2),
            "p95_latency_ms": round(percentile(all_latencies, 95), 2),
            "p99_latency_ms": round(percentile(all_latencies, 99), 2),
            "sample_count": n
        }
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary data for admin dashboard"""
        r = await self._get_redis()
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Get today's stats
        today_stats = await self.get_daily_stats(today)
        yesterday_stats = await self.get_daily_stats(yesterday)
        
        # Calculate trends
        def calc_trend(today_val, yesterday_val):
            if yesterday_val == 0:
                return 100 if today_val > 0 else 0
            return round(((today_val - yesterday_val) / yesterday_val) * 100, 1)
        
        # Get weekly stats
        weekly_chats = 0
        weekly_messages = 0
        weekly_satisfaction = []
        
        for i in range(7):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            stats = await r.hgetall(f"stats:daily:{date}")
            weekly_chats += int(stats.get("total_chats", 0))
            weekly_messages += int(stats.get("total_messages", 0))
            
            sat_scores = await r.lrange(f"stats:daily:{date}:satisfaction", 0, -1)
            weekly_satisfaction.extend([int(s) for s in sat_scores if s])
        
        # Get recent feedback
        recent_feedback = await r.lrange("feedback:recent", 0, 9)
        recent_feedback = [json.loads(f) for f in recent_feedback]
        
        # Get model comparison
        models = ["llama3.2", "mistral", "dolphin-mixtral", "qwen2.5-coder"]
        model_performance = {}
        
        for model in models:
            perf = await self.get_model_performance(model, days=7)
            if perf.get("total_requests", 0) > 0:
                model_performance[model] = perf
        
        return {
            "today": {
                "chats": today_stats.get("total_chats", 0),
                "messages": today_stats.get("total_messages", 0),
                "unique_clients": today_stats.get("unique_clients", 0),
                "satisfaction": today_stats.get("satisfaction_avg", 0),
                "resolution_rate": today_stats.get("resolution_rate", 0),
                "cache_hit_rate": today_stats.get("cache_hit_rate", 0)
            },
            "trends": {
                "chats": calc_trend(
                    today_stats.get("total_chats", 0),
                    yesterday_stats.get("total_chats", 0)
                ),
                "messages": calc_trend(
                    today_stats.get("total_messages", 0),
                    yesterday_stats.get("total_messages", 0)
                ),
                "satisfaction": calc_trend(
                    today_stats.get("satisfaction_avg", 0),
                    yesterday_stats.get("satisfaction_avg", 0)
                )
            },
            "weekly": {
                "total_chats": weekly_chats,
                "total_messages": weekly_messages,
                "avg_satisfaction": round(
                    sum(weekly_satisfaction) / len(weekly_satisfaction), 2
                ) if weekly_satisfaction else 0,
                "feedback_count": len(weekly_satisfaction)
            },
            "model_performance": model_performance,
            "recent_feedback": recent_feedback[:5],
            "peak_hour": today_stats.get("peak_hour", 0),
            "model_usage_today": today_stats.get("model_usage", {})
        }
    
    async def get_response_time_trend(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get response time trend over days"""
        trend = []
        
        for i in range(days - 1, -1, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Aggregate latencies for all models
            all_latencies = []
            
            for model in ["llama3.2", "mistral", "dolphin-mixtral", "qwen2.5-coder"]:
                r = await self._get_redis()
                latencies = await r.zrange(
                    f"stats:latency:{date}:{model}",
                    0, -1,
                    withscores=True
                )
                all_latencies.extend([lat for _, lat in latencies])
            
            avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
            
            trend.append({
                "date": date,
                "avg_response_time_ms": round(avg_latency, 2),
                "request_count": len(all_latencies)
            })
        
        return trend
    
    async def get_satisfaction_distribution(self, days: int = 30) -> Dict[str, int]:
        """Get distribution of satisfaction scores"""
        r = await self._get_redis()
        
        distribution = {str(i): 0 for i in range(1, 6)}
        
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            scores = await r.lrange(f"stats:daily:{date}:satisfaction", 0, -1)
            
            for score in scores:
                if score in distribution:
                    distribution[score] += 1
        
        return distribution


# Singleton instance
_analytics_instance: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get analytics service singleton"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = AnalyticsService()
    return _analytics_instance
