import robin_stocks.robinhood as rr
from functools import wraps
from retry.api import retry_call
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from config import api_rate_limit, use_api_rate_limit, use_caching

# Cache TTL in seconds (5 minutes)
CACHE_TTL = 300

class CacheStats:
    """Tracks cache hit/miss statistics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def record_hit(self):
        with self.lock:
            self.hits += 1
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
    
    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                'caching_enabled': use_caching,
                'hits': self.hits,
                'misses': self.misses,
                'total': total,
                'hit_rate': hit_rate
            }
    
    def reset(self):
        with self.lock:
            self.hits = 0
            self.misses = 0
    
    def print_stats(self):
        stats = self.get_stats()
        caching_status = "ON" if use_caching else "OFF"
        print(f"\n----- Cache Stats -----")
        print(f"Caching: {caching_status}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit rate: {stats['hit_rate']:.1f}%")
        print(f"-----------------------\n")

# Global cache stats
cache_stats = CacheStats()

class TimedCache:
    """A cache with TTL (time-to-live) support"""
    def __init__(self, ttl=CACHE_TTL):
        self.cache = {}
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def get(self, key):
        # If caching is disabled, always return cache miss
        if not use_caching:
            return None, False
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    cache_stats.record_hit()
                    return value, True
                else:
                    # Expired
                    del self.cache[key]
            cache_stats.record_miss()
            return None, False
    
    def set(self, key, value):
        # Don't store in cache if caching is disabled
        if not use_caching:
            return
        
        with self.lock:
            self.cache[key] = (value, datetime.now())
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def size(self):
        with self.lock:
            return len(self.cache)

# Global timed caches for different endpoints
historicals_cache = TimedCache(ttl=CACHE_TTL)
instruments_cache = TimedCache(ttl=CACHE_TTL)
fundamentals_cache = TimedCache(ttl=CACHE_TTL)
watchlist_cache = TimedCache(ttl=60)  # Shorter TTL for watchlists
instrument_url_cache = TimedCache(ttl=CACHE_TTL)

def clear_all_caches():
    """Clear all caches - call at start of each scan"""
    historicals_cache.clear()
    instruments_cache.clear()
    fundamentals_cache.clear()
    watchlist_cache.clear()
    instrument_url_cache.clear()
    cache_stats.reset()
    print("🗑️  All caches cleared")

class APITracker:
    """Tracks API requests per minute and enforces rate limiting"""
    def __init__(self, rate_limit=api_rate_limit):
        self.requests = deque()  # Store timestamps of requests
        self.lock = threading.Lock()
        self.total_requests = 0
        self.rate_limit = rate_limit
        self.rate_limit_waits = 0  # Track how many times we had to wait
    
    def wait_if_needed(self):
        """Wait if we're at the rate limit (only if rate limiting is enabled)"""
        if not use_api_rate_limit:
            return
        
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            # Clean up old requests
            while self.requests and self.requests[0][0] < cutoff:
                self.requests.popleft()
            
            # If at or over limit, wait until oldest request expires
            if len(self.requests) >= self.rate_limit:
                oldest = self.requests[0][0]
                wait_time = (oldest + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    self.rate_limit_waits += 1
                    print(f"⏳ Rate limit reached ({self.rate_limit}/min). Waiting {wait_time:.1f}s...")
                    # Release lock while waiting
                    self.lock.release()
                    try:
                        time.sleep(wait_time + 0.1)  # Add small buffer
                    finally:
                        self.lock.acquire()
                    # Clean up again after waiting
                    now = datetime.now()
                    cutoff = now - timedelta(minutes=1)
                    while self.requests and self.requests[0][0] < cutoff:
                        self.requests.popleft()
    
    def record_request(self, endpoint=""):
        """Record an API request (with rate limiting if enabled)"""
        self.wait_if_needed()
        with self.lock:
            now = datetime.now()
            self.requests.append((now, endpoint))
            self.total_requests += 1
            # Clean up old requests (older than 1 minute)
            cutoff = now - timedelta(minutes=1)
            while self.requests and self.requests[0][0] < cutoff:
                self.requests.popleft()
    
    def get_requests_per_minute(self):
        """Get the number of requests in the last minute"""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            # Clean up old requests
            while self.requests and self.requests[0][0] < cutoff:
                self.requests.popleft()
            return len(self.requests)
    
    def get_stats(self):
        """Get detailed stats about API usage"""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            # Clean up old requests
            while self.requests and self.requests[0][0] < cutoff:
                self.requests.popleft()
            
            # Count requests by endpoint in last minute
            endpoint_counts = {}
            for timestamp, endpoint in self.requests:
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
            
            return {
                'requests_last_minute': len(self.requests),
                'total_requests': self.total_requests,
                'rate_limit': self.rate_limit,
                'rate_limit_enabled': use_api_rate_limit,
                'rate_limit_waits': self.rate_limit_waits,
                'by_endpoint': endpoint_counts
            }
    
    def print_stats(self):
        """Print API usage stats"""
        stats = self.get_stats()
        rate_limit_status = "ON" if stats['rate_limit_enabled'] else "OFF"
        print(f"\n----- API Request Stats -----")
        print(f"Rate limiting: {rate_limit_status} ({stats['rate_limit']}/min)")
        print(f"Requests in last minute: {stats['requests_last_minute']}")
        print(f"Total requests this session: {stats['total_requests']}")
        print(f"Rate limit waits: {stats['rate_limit_waits']}")
        if stats['by_endpoint']:
            print("By endpoint (last minute):")
            for endpoint, count in sorted(stats['by_endpoint'].items(), key=lambda x: -x[1]):
                print(f"  {endpoint}: {count}")
        print("-----------------------------\n")

# Global API tracker instance
api_tracker = APITracker(rate_limit=api_rate_limit)

class rsa:
    def get_fundamentals(stock):
        cache_key = f"fundamentals:{stock}"
        cached, found = fundamentals_cache.get(cache_key)
        if found:
            return cached
        api_tracker.record_request("get_fundamentals")
        result = rr.get_fundamentals(stock)
        fundamentals_cache.set(cache_key, result)
        return result

    def get_stock_historicals(stockTicker, interval='hour', span='week', bounds='regular', info=None):
        cache_key = f"historicals:{stockTicker}:{interval}:{span}:{bounds}"
        cached, found = historicals_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_stock_historicals, fargs=[stockTicker, interval, span, bounds, info], tries=5, backoff=5, delay=5)
        historicals_cache.set(cache_key, result)
        return result

    def try_get_stock_historicals(stockTicker, interval, span, bounds, info):
        api_tracker.record_request("get_stock_historicals")
        result = rr.get_stock_historicals(stockTicker, interval=interval, span=span, bounds=bounds, info=None)

        # If the result is None or an empty list, return an empty list to represent no historicals for this stock.
        if (result is None) or (len(result) == 0):
            return []
        
        test_value = result[0]['close_price']
        return result

    def get_instruments_by_symbols(stockTicker, info=None):
        cache_key = f"instruments:{stockTicker}:{info}"
        cached, found = instruments_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_instruments_by_symbols, fargs=[stockTicker], fkwargs={"info": info}, tries=3, backoff=5, delay=5)
        instruments_cache.set(cache_key, result)
        return result

    def try_get_instruments_by_symbols(stockTicker, info=None):
        api_tracker.record_request("get_instruments_by_symbols")
        result = rr.get_instruments_by_symbols(stockTicker, info=info)
        test = len(result)
        return result

    def get_instrument_by_url(url, info=None):
        cache_key = f"instrument_url:{url}:{info}"
        cached, found = instrument_url_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_instrument_by_url, fargs=[url], fkwargs={"info": info}, tries=3, backoff=5, delay=5)
        instrument_url_cache.set(cache_key, result)
        return result

    def try_get_instrument_by_url(url, info=None):
        api_tracker.record_request("get_instrument_by_url")
        result = rr.get_instrument_by_url(url, info=info)
        test = len(result)
        return result

    def get_watchlist_by_name(name):
        cache_key = f"watchlist:{name}"
        cached, found = watchlist_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_watchlist_by_name, fargs=[name], tries=3, backoff=5, delay=5)
        watchlist_cache.set(cache_key, result)
        return result
        
    def try_get_watchlist_by_name(name):
        api_tracker.record_request("get_watchlist_by_name")
        result = rr.get_watchlist_by_name(name=name)
        return result

    def get_all_open_stock_orders(url=None):
        if url:
            result = retry_call(rsa.try_get_all_open_stock_orders_with_url, fargs=[url], tries=3, backoff=5, delay=5)
        else:
            result = retry_call(rsa.try_get_all_open_stock_orders, tries=3, backoff=5, delay=5)
        return result
    
    def try_get_all_open_stock_orders_with_url(url):
        api_tracker.record_request("get_all_open_stock_orders")
        result = rr.get_all_open_stock_orders(url)
        return result

    def try_get_all_open_stock_orders():
        api_tracker.record_request("get_all_open_stock_orders")
        result = rr.get_all_open_stock_orders()
        return result
    
    def get_name_by_symbol(symbol):
        cache_key = f"name_symbol:{symbol}"
        cached, found = instruments_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_name_by_symbol, fargs=[symbol], tries=3, backoff=5, delay=5)
        instruments_cache.set(cache_key, result)
        return result

    def try_get_name_by_symbol(symbol):
        api_tracker.record_request("get_name_by_symbol")
        result = rr.get_name_by_symbol(symbol)
        return result
    
    def get_name_by_url(url):
        cache_key = f"name_url:{url}"
        cached, found = instrument_url_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_name_by_url, fargs=[url], tries=3, backoff=5, delay=5)
        instrument_url_cache.set(cache_key, result)
        return result
    
    def try_get_name_by_url(url):
        api_tracker.record_request("get_name_by_url")
        result = rr.get_name_by_url(url)
        return result
    
    def get_symbol_by_url(url):
        cache_key = f"symbol_url:{url}"
        cached, found = instrument_url_cache.get(cache_key)
        if found:
            return cached
        result = retry_call(rsa.try_get_symbol_by_url, fargs=[url], tries=3, backoff=5, delay=5)
        instrument_url_cache.set(cache_key, result)
        return result
    
    def try_get_symbol_by_url(url):
        api_tracker.record_request("get_symbol_by_url")
        result = retry_call(rr.get_symbol_by_url, fargs=[url], tries=3, backoff=5, delay=5)
        return result
    

