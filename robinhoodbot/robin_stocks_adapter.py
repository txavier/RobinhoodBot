import robin_stocks.robinhood as rr
from functools import cache
from retry.api import retry_call
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from config import api_rate_limit, use_api_rate_limit

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
                'rate_limit_waits': self.rate_limit_waits,
                'by_endpoint': endpoint_counts
            }
    
    def print_stats(self):
        """Print API usage stats"""
        stats = self.get_stats()
        print(f"\n----- API Request Stats -----")
        print(f"Rate limit: {stats['rate_limit']}/min")
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
    @cache
    def get_fundamentals(stock):
        api_tracker.record_request("get_fundamentals")
        result = rr.get_fundamentals(stock)
        return result

    @cache
    def get_stock_historicals(stockTicker, interval='hour', span='week', bounds='regular', info=None):
        # print('Cache not found for ' + stockTicker)
        result = retry_call(rsa.try_get_stock_historicals, fargs=[stockTicker, interval, span, bounds, info], tries=5, backoff=5, delay=5)
        return result

    def try_get_stock_historicals(stockTicker, interval, span, bounds, info):
        # print('Trying ' + stockTicker)
        api_tracker.record_request("get_stock_historicals")
        result = rr.get_stock_historicals(stockTicker, interval=interval, span=span, bounds=bounds, info=None)

        # If the result is None or an empty list, return an empty list to represent no historicals for this stock.
        if (result is None) or (len(result) == 0):
            return []
        
        test_value = result[0]['close_price']
        return result

    @cache
    def get_instruments_by_symbols(stockTicker, info=None):
        result = retry_call(rsa.try_get_instruments_by_symbols, fargs=[stockTicker], fkwargs={"info": info}, tries=3, backoff=5, delay=5)
        return result

    def try_get_instruments_by_symbols(stockTicker, info=None):
        api_tracker.record_request("get_instruments_by_symbols")
        result = rr.get_instruments_by_symbols(stockTicker, info=info)
        test = len(result)
        return result

    @cache
    def get_instrument_by_url(url, info=None):
        result = retry_call(rsa.try_get_instrument_by_url, fargs=[url], fkwargs={"info": info}, tries=3, backoff=5, delay=5)
        return result

    def try_get_instrument_by_url(url, info=None):
        api_tracker.record_request("get_instrument_by_url")
        result = rr.get_instrument_by_url(url, info=info)
        test = len(result)
        return result

    @cache
    def get_watchlist_by_name(name):
        result = retry_call(rsa.try_get_watchlist_by_name, fargs=[name], tries=3, backoff=5, delay=5)
        return result
        
    def try_get_watchlist_by_name(name):
        api_tracker.record_request("get_watchlist_by_name")
        result = rr.get_watchlist_by_name(name=name)
        return result

    @cache
    def get_all_open_stock_orders(url):
        result = retry_call(rsa.try_get_all_open_stock_orders, fargs=[url], tries=3, backoff=5, delay=5)
        return result
        
    def try_get_all_open_stock_orders(url):
        api_tracker.record_request("get_all_open_stock_orders")
        result = rr.get_all_open_stock_orders(url)
        return result

    @cache
    def get_all_open_stock_orders():
        result = retry_call(rsa.try_get_all_open_stock_orders, tries=3, backoff=5, delay=5)
        return result

    def try_get_all_open_stock_orders():
        api_tracker.record_request("get_all_open_stock_orders")
        result = rr.get_all_open_stock_orders()
        return result
    
    @cache
    def get_name_by_symbol(symbol):
        result = retry_call(rsa.try_get_name_by_symbol(symbol))
        return result

    def try_get_name_by_symbol(symbol):
        api_tracker.record_request("get_name_by_symbol")
        result = rr.get_name_by_symbol(symbol)
        return result
    
    @cache
    def get_name_by_url(url):
        result = retry_call(rsa.try_get_name_by_url(url))
        return result
    
    def try_get_name_by_url(url):
        api_tracker.record_request("get_name_by_url")
        result = rr.get_name_by_url(url)
        return result
    
    @cache
    def get_symbol_by_url(url):
        result = retry_call(rsa.try_get_symbol_by_url(url))
        return result
    
    def try_get_symbol_by_url(url):
        api_tracker.record_request("get_symbol_by_url")
        result = retry_call(rr.get_symbol_by_url(url))
        return result
    

