import os
import json
import time
import pickle
import hashlib
import logging

from functools import wraps
from prometheus_client import Counter
from typing import Any, Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MultiLayerCache:
    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_ttl: int = 3600,
        l3_max_size_mb: int = 100
    ):
        self.l1_max_size = l1_max_size
        self.l2_ttl = l2_ttl
        self.l3_max_size_mb = l3_max_size_mb
        
        # L1: In-memory cache (LRU)
        self.l1_cache: Dict[str, Tuple[Any, float]] = {}
        self.l1_access_order: List[str] = []
        
        # L2: Redis cache (if available)
        self.redis_client = self._init_redis()
        
        # L3: Disk cache
        self.l3_cache_dir = "cache/disk"
        os.makedirs(self.l3_cache_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0, 
            "l3_hits": 0, "l3_misses": 0
        }
    
    def _init_redis(self):
        """Initialize Redis client if available"""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            client.ping()  # Test connection
            logger.info("Redis L2 cache connected")
            return client
        except Exception as e:
            logger.warning(f"Redis L2 cache unavailable: {e}")
            return None
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters"""
        key_data = f"{query}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get from cache with L1 -> L2 -> L3 fallback"""
        cache_key = self._generate_key(query, **kwargs)
        
        # Try L1 (memory)
        result = self._get_l1(cache_key)
        if result is not None:
            self.stats["l1_hits"] += 1
            return result
        self.stats["l1_misses"] += 1
        
        # Try L2 (Redis) 
        result = self._get_l2(cache_key)
        if result is not None:
            self.stats["l2_hits"] += 1
            # Promote to L1
            self._set_l1(cache_key, result)
            return result
        self.stats["l2_misses"] += 1
        
        # Try L3 (disk)
        result = self._get_l3(cache_key)
        if result is not None:
            self.stats["l3_hits"] += 1
            # Promote to L2 and L1
            self._set_l2(cache_key, result)
            self._set_l1(cache_key, result)
            return result
        self.stats["l3_misses"] += 1
        
        return None
    
    def set(self, query: str, value: Any, **kwargs):
        """Set in all cache layers"""
        cache_key = self._generate_key(query, **kwargs)
        
        # Set in all layers
        self._set_l1(cache_key, value)
        self._set_l2(cache_key, value)
        self._set_l3(cache_key, value)
    
    def _get_l1(self, key: str) -> Optional[Any]:
        """Get from L1 memory cache"""
        if key in self.l1_cache:
            value, timestamp = self.l1_cache[key]
            # Move to end (most recently used)
            self.l1_access_order.remove(key)
            self.l1_access_order.append(key)
            return value
        return None
    
    def _set_l1(self, key: str, value: Any):
        """Set in L1 memory cache with LRU eviction"""
        if key in self.l1_cache:
            self.l1_access_order.remove(key)
        
        # Evict if at capacity
        while len(self.l1_cache) >= self.l1_max_size:
            oldest_key = self.l1_access_order.pop(0)
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = (value, time.time())
        self.l1_access_order.append(key)
    
    def _get_l2(self, key: str) -> Optional[Any]:
        """Get from L2 Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(f"rag:{key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"L2 cache get error: {e}")
        return None
    
    def _set_l2(self, key: str, value: Any):
        """Set in L2 Redis cache"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                f"rag:{key}", 
                self.l2_ttl, 
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.warning(f"L2 cache set error: {e}")
    
    def _get_l3(self, key: str) -> Optional[Any]:
        """Get from L3 disk cache"""
        cache_file = os.path.join(self.l3_cache_dir, f"{key}.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"L3 cache get error: {e}")
        return None
    
    def _set_l3(self, key: str, value: Any):
        """Set in L3 disk cache"""
        cache_file = os.path.join(self.l3_cache_dir, f"{key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Clean up if cache is too large
            self._cleanup_l3_cache()
        except Exception as e:
            logger.warning(f"L3 cache set error: {e}")
    
    def _cleanup_l3_cache(self):
        """Clean up L3 cache if it exceeds size limit"""
        try:
            total_size = sum(
                os.path.getsize(os.path.join(self.l3_cache_dir, f))
                for f in os.listdir(self.l3_cache_dir)
            ) / (1024 * 1024)  # Convert to MB
            
            if total_size > self.l3_max_size_mb:
                # Remove oldest files
                files = [
                    (f, os.path.getmtime(os.path.join(self.l3_cache_dir, f)))
                    for f in os.listdir(self.l3_cache_dir)
                ]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                # Remove oldest 25% of files
                files_to_remove = files[:len(files) // 4]
                for filename, _ in files_to_remove:
                    os.remove(os.path.join(self.l3_cache_dir, filename))
                
                logger.info(f"Cleaned up {len(files_to_remove)} old cache files")
        except Exception as e:
            logger.warning(f"L3 cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.stats.values())
        if total_requests == 0:
            return self.stats
        
        hit_rate = (
            self.stats["l1_hits"] + 
            self.stats["l2_hits"] + 
            self.stats["l3_hits"]
        ) / total_requests * 100
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "overall_hit_rate": round(hit_rate, 2),
            "l1_size": len(self.l1_cache),
            "l2_available": self.redis_client is not None
        }
    
    def clear(self):
        """Clear all cache layers"""
        # Clear L1
        self.l1_cache.clear()
        self.l1_access_order.clear()
        
        # Clear L2
        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter(match="rag:*"):
                    self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"L2 cache clear error: {e}")
        
        # Clear L3
        try:
            for filename in os.listdir(self.l3_cache_dir):
                os.remove(os.path.join(self.l3_cache_dir, filename))
        except Exception as e:
            logger.warning(f"L3 cache clear error: {e}")
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats}


cache = MultiLayerCache()

# --- Prometheus Metrics ---
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')

def cached_search(func):
    """Decorator for caching search results"""
    @wraps(func)
    def wrapper(self, query: str, **kwargs):
        # Check if cache should be bypassed
        bypass_cache = kwargs.pop('bypass_cache', False)
        
        if not bypass_cache:
            # Try cache first
            cached_result = cache.get(query, **kwargs)
            if cached_result is not None:
                # Track cache hit
                if CACHE_HITS:
                    CACHE_HITS.inc()
                return cached_result
        
        # Track cache miss
        if CACHE_MISSES:
            CACHE_MISSES.inc()
        
        # Execute function and cache result
        result = func(self, query, **kwargs)
        if not bypass_cache:
            cache.set(query, result, **kwargs)
        return result
    
    return wrapper
