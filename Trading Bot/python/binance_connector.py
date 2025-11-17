# encoding: utf-8
"""
Optimized Binance API Connector
- Async/await support for non-blocking operations
- Connection pooling and caching
- Retry logic with exponential backoff
- Rate limiting and circuit breaker
- Performance optimizations
"""

import aiohttp
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import hmac
from urllib.parse import urlencode
import json
from functools import lru_cache
from collections import deque
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger("binance_connector")


@dataclass
class RateLimitTracker:
    """Track API rate limits to avoid being blocked"""
    requests_limit: int = 1200  # per minute
    requests_used: deque = field(default_factory=lambda: deque(maxlen=1200))
    
    def can_request(self) -> bool:
        now = time.time()
        # Remove old requests outside the 1-minute window
        while self.requests_used and (now - self.requests_used[0]) > 60:
            self.requests_used.popleft()
        return len(self.requests_used) < self.requests_limit
    
    def record_request(self):
        self.requests_used.append(time.time())


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    failure_threshold: int = 5
    reset_timeout: int = 60
    failures: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
    
    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.failures = 0
                return True
            return False
        return True


class BinanceConnector:
    """
    High-performance Binance API connector with async support
    Features:
    - Async/await for non-blocking operations
    - Connection pooling
    - Automatic caching of market data
    - Rate limiting and circuit breaker
    - Exponential backoff retry logic
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
            self.ws_url = "wss://stream.testnet.binance.vision:9443/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.ws_url = "wss://stream.binance.com:9443/ws"
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            ssl=True
        )
        
        # Rate limiting and resilience
        self.rate_limiter = RateLimitTracker()
        self.circuit_breaker = CircuitBreaker()
        
        # Caching
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self._klines_cache: Dict[str, pd.DataFrame] = {}
        self._cache_ttl = 1  # seconds
        
        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0
        self.request_times = deque(maxlen=100)
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        """Initialize async session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=self.connector)
    
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _get_request_headers(self) -> Dict[str, str]:
        """Generate request headers"""
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    
    def _sign_request(self, params: Dict) -> str:
        """Sign request for private endpoints"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return query_string + '&signature=' + signature
    
    async def _request_with_retry(self, method: str, endpoint: str, 
                                   params: Dict = None, is_private: bool = False,
                                   max_retries: int = 3) -> Dict:
        """
        Make HTTP request with exponential backoff retry
        """
        if not self.circuit_breaker.can_attempt():
            raise Exception("Circuit breaker is OPEN - API temporarily unavailable")
        
        for attempt in range(max_retries):
            # Check rate limit
            while not self.rate_limiter.can_request():
                await asyncio.sleep(0.1)
            
            try:
                url = f"{self.base_url}{endpoint}"
                params = params or {}
                headers = self._get_request_headers()
                
                if is_private:
                    params['timestamp'] = int(time.time() * 1000)
                    params['recvWindow'] = 5000
                    query_string = self._sign_request(params)
                    url += '?' + query_string
                
                start_time = time.time()
                self.rate_limiter.record_request()
                
                async with self.session.request(method, url, headers=headers, 
                                              params=None if is_private else params,
                                              timeout=aiohttp.ClientTimeout(total=10)) as response:
                    elapsed = time.time() - start_time
                    self.request_times.append(elapsed)
                    self.request_count += 1
                    
                    if response.status == 200:
                        self.circuit_breaker.record_success()
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited - wait longer
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"Rate limited. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        error_msg = await response.text()
                        raise Exception(f"API Error {response.status}: {error_msg}")
            
            except asyncio.TimeoutError:
                self.circuit_breaker.record_failure()
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            
            except Exception as e:
                self.circuit_breaker.record_failure()
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    # ============== PUBLIC ENDPOINTS ==============
    
    async def get_server_time(self) -> int:
        """Get server time"""
        response = await self._request_with_retry('GET', '/v3/time')
        return response['serverTime']
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol information"""
        response = await self._request_with_retry('GET', '/v3/exchangeInfo')
        return response
    
    async def get_price(self, symbol: str, use_cache: bool = True) -> float:
        """Get current price for symbol"""
        if use_cache:
            cached_price, cached_time = self._price_cache.get(symbol, (None, 0))
            if cached_price and (time.time() - cached_time) < self._cache_ttl:
                self.cache_hits += 1
                return cached_price
        
        response = await self._request_with_retry('GET', '/v3/ticker/price', {'symbol': symbol})
        price = float(response['price'])
        self._price_cache[symbol] = (price, time.time())
        return price
    
    async def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols in parallel"""
        tasks = [self.get_price(symbol) for symbol in symbols]
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            symbol: price for symbol, price in zip(symbols, prices)
            if not isinstance(price, Exception)
        }
    
    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100,
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Get klines (candlestick data) for symbol
        Optimized to return as DataFrame for technical analysis
        """
        cache_key = f"{symbol}_{interval}_{limit}"
        if use_cache and cache_key in self._klines_cache:
            cached_df, cached_time = self._klines_cache[cache_key], time.time()
            if (time.time() - cached_time) < self._cache_ttl:
                self.cache_hits += 1
                return cached_df
        
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = await self._request_with_retry('GET', '/v3/klines', params)
        
        # Convert to DataFrame for performance
        df = pd.DataFrame(response, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        self._klines_cache[cache_key] = df
        return df
    
    async def get_depth(self, symbol: str, limit: int = 5) -> Dict:
        """Get order book depth"""
        params = {'symbol': symbol, 'limit': limit}
        return await self._request_with_retry('GET', '/v3/depth', params)
    
    async def get_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent trades"""
        params = {'symbol': symbol, 'limit': limit}
        return await self._request_with_retry('GET', '/v3/trades', params)
    
    # ============== PRIVATE ENDPOINTS ==============
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: Optional[float] = None) -> Dict:
        """Place an order"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'timeInForce': 'GTC',
            'quantity': quantity
        }
        
        if order_type.upper() == 'LIMIT' and price:
            params['price'] = price
        
        return await self._request_with_retry('POST', '/v3/order', params, is_private=True)
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place market order"""
        return await self.place_order(symbol, side, 'MARKET', quantity)
    
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Place limit order"""
        return await self.place_order(symbol, side, 'LIMIT', quantity, price)
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an order"""
        params = {'symbol': symbol, 'orderId': order_id}
        return await self._request_with_retry('DELETE', '/v3/order', params, is_private=True)
    
    async def get_order(self, symbol: str, order_id: int) -> Dict:
        """Get order status"""
        params = {'symbol': symbol, 'orderId': order_id}
        return await self._request_with_retry('GET', '/v3/order', params, is_private=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request_with_retry('GET', '/v3/openOrders', params, is_private=True)
    
    async def get_account(self) -> Dict:
        """Get account information"""
        return await self._request_with_retry('GET', '/v3/account', {}, is_private=True)
    
    async def get_balance(self, asset: Optional[str] = None) -> Dict:
        """Get account balance"""
        account = await self.get_account()
        balances = {b['asset']: {
            'free': float(b['free']),
            'locked': float(b['locked']),
            'total': float(b['free']) + float(b['locked'])
        } for b in account['balances']}
        
        if asset:
            return balances.get(asset, {'free': 0, 'locked': 0, 'total': 0})
        return balances
    
    # ============== PERFORMANCE METRICS ==============
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        avg_request_time = np.mean(list(self.request_times)) if self.request_times else 0
        cache_hit_rate = (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            'total_requests': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'avg_request_time': f"{avg_request_time*1000:.2f}ms",
            'circuit_breaker_state': self.circuit_breaker.state,
            'rate_limiter_usage': f"{len(self.rate_limiter.requests_used)}/{self.rate_limiter.requests_limit}"
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self._price_cache.clear()
        self._klines_cache.clear()
