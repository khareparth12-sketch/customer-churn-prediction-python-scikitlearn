import hashlib
import json
import logging
import redis.asyncio as aioredis
from typing import Optional

logger = logging.getLogger(__name__)

# Single async Redis client — reused across requests
redis_client: Optional[aioredis.Redis] = None

CACHE_TTL_SECONDS = 3600  # 1 hour


async def get_redis() -> Optional[aioredis.Redis]:
    global redis_client
    if redis_client is None:
        try:
            redis_client = aioredis.Redis(
                host="redis",      # Docker Compose service name
                port=6379,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Caching disabled.")
            redis_client = None
    return redis_client


def make_cache_key(features: dict) -> str:
    """Deterministic SHA-256 key from sorted feature dict."""
    payload = json.dumps(features, sort_keys=True, default=str)
    return "churn:" + hashlib.sha256(payload.encode()).hexdigest()


async def get_cached(key: str) -> Optional[dict]:
    r = await get_redis()
    if r is None:
        return None
    try:
        value = await r.get(key)
        return json.loads(value) if value else None
    except Exception as e:
        logger.warning(f"Cache GET failed: {e}")
        return None


async def set_cached(key: str, value: dict) -> None:
    r = await get_redis()
    if r is None:
        return
    try:
        await r.setex(key, CACHE_TTL_SECONDS, json.dumps(value))
    except Exception as e:
        logger.warning(f"Cache SET failed: {e}")