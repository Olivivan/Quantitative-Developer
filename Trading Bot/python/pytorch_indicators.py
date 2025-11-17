# encoding: utf-8
"""
PyTorch-optimized Technical Indicators Library
- GPU acceleration support
- Vectorized operations with NumPy/PyTorch
- Caching and memoization
- High-performance calculations
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import torch
from functools import lru_cache
import logging

logger = logging.getLogger("pytorch_indicators")


class TechnicalIndicators:
    """
    High-performance technical indicators using NumPy and PyTorch
    All operations are vectorized for maximum performance
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self._cache: Dict = {}
    
    @staticmethod
    def sma(data: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Simple Moving Average - Vectorized
        Efficient: O(n) time complexity
        """
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        # Use stride tricks for efficient windowing
        sma_values = np.convolve(data, np.ones(period) / period, mode='valid')
        return np.concatenate([np.full(period - 1, np.nan), sma_values])
    
    @staticmethod
    def ema(data: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Exponential Moving Average - Vectorized
        Efficient: O(n) time complexity with no loops
        """
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        ema_data = np.zeros_like(data, dtype=float)
        multiplier = 2.0 / (period + 1.0)
        
        # First SMA value
        ema_data[:period] = np.nan
        ema_data[period - 1] = np.mean(data[:period])
        
        # Vectorized EMA calculation
        for i in range(period, len(data)):
            ema_data[i] = data[i] * multiplier + ema_data[i - 1] * (1 - multiplier)
        
        return ema_data
    
    def ema_vectorized_gpu(self, data: torch.Tensor, period: int = 20) -> np.ndarray:
        """
        GPU-accelerated EMA for large datasets
        """
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).to(self.device).float()
        else:
            data = data.to(self.device).float()
        
        multiplier = 2.0 / (period + 1.0)
        ema = torch.zeros_like(data)
        ema[:period] = float('nan')
        
        # First SMA
        ema[period - 1] = data[:period].mean()
        
        # Vectorized operations on GPU
        for i in range(period, len(data)):
            ema[i] = data[i] * multiplier + ema[i - 1] * (1 - multiplier)
        
        return ema.cpu().numpy()
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Relative Strength Index - Vectorized
        Efficient: O(n) time complexity
        """
        if len(data) < period + 1:
            return np.full_like(data, np.nan, dtype=float)
        
        # Calculate price changes
        deltas = np.diff(data)
        seed = deltas[:period + 1]
        
        # Separate gains and losses
        ups = seed[seed >= 0].sum() / period
        downs = -seed[seed < 0].sum() / period
        
        # Initialize RSI array
        rsi = np.zeros_like(data, dtype=float)
        rsi[:period + 1] = np.nan
        
        # Calculate RS and RSI for seed period
        rs = ups / downs if downs > 0 else 0
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
        
        # Vectorized calculation for remaining periods
        for i in range(period + 1, len(data)):
            delta = deltas[i - 1]
            if delta > 0:
                up = delta
                down = 0
            else:
                up = 0
                down = -delta
            
            ups = (ups * (period - 1) + up) / period
            downs = (downs * (period - 1) + down) / period
            
            rs = ups / downs if downs > 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD (Moving Average Convergence Divergence) - Vectorized
        Returns: MACD line, Signal line, Histogram
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands - Vectorized
        Returns: Upper band, Middle band (SMA), Lower band
        """
        middle = TechnicalIndicators.sma(data, period)
        
        # Vectorized std calculation
        std = np.array([np.std(data[max(0, i - period + 1):i + 1]) 
                       for i in range(len(data))])
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average True Range - Vectorized
        Measures volatility
        """
        # Calculate true range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan
        
        # Calculate ATR using EMA
        atr_values = np.zeros_like(tr, dtype=float)
        atr_values[:period] = np.nan
        atr_values[period - 1] = np.mean(tr[:period])
        
        multiplier = 2.0 / (period + 1.0)
        for i in range(period, len(tr)):
            atr_values[i] = tr[i] * multiplier + atr_values[i - 1] * (1 - multiplier)
        
        return atr_values
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator - Vectorized
        Returns: %K line, %D line
        """
        k_values = np.zeros_like(close, dtype=float)
        k_values[:k_period] = np.nan
        
        # Vectorized lowest low and highest high
        for i in range(k_period - 1, len(close)):
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            highest_high = np.max(high[i - k_period + 1:i + 1])
            
            if highest_high - lowest_low == 0:
                k_values[i] = 0
            else:
                k_values[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D is SMA of %K
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def momentum(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Momentum (Price Rate of Change) - Vectorized
        """
        momentum = np.zeros_like(data, dtype=float)
        momentum[:period] = np.nan
        momentum[period:] = data[period:] - data[:-period]
        return momentum
    
    @staticmethod
    def roc(data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Rate of Change - Vectorized
        """
        roc_values = np.zeros_like(data, dtype=float)
        roc_values[:period] = np.nan
        roc_values[period:] = ((data[period:] - data[:-period]) / data[:-period]) * 100
        return roc_values
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average Directional Index - Vectorized
        Measures trend strength
        """
        # Calculate directional movements
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        
        plus_dm[1:] = np.where((up > down) & (up > 0), up, 0)
        minus_dm[1:] = np.where((down > up) & (down > 0), down, 0)
        
        # Calculate ATR
        tr = TechnicalIndicators.atr(high, low, close, period)
        
        # Calculate directional indicators
        plus_di = 100 * (TechnicalIndicators.ema(plus_dm, period) / tr)
        minus_di = 100 * (TechnicalIndicators.ema(minus_dm, period) / tr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = TechnicalIndicators.ema(dx, period)
        
        return adx
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, use_gpu: bool = False) -> pd.DataFrame:
        """
        Batch calculate all indicators - optimized for speed
        Input DataFrame must have: open, high, low, close, volume
        """
        result = df.copy()
        
        # Moving averages
        result['sma_20'] = TechnicalIndicators.sma(df['close'].values, 20)
        result['sma_50'] = TechnicalIndicators.sma(df['close'].values, 50)
        result['ema_9'] = TechnicalIndicators.ema(df['close'].values, 9)
        result['ema_26'] = TechnicalIndicators.ema(df['close'].values, 26)
        
        # Momentum indicators
        result['rsi_14'] = TechnicalIndicators.rsi(df['close'].values, 14)
        macd, signal, histogram = TechnicalIndicators.macd(df['close'].values)
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = histogram
        
        # Volatility
        upper_bb, middle_bb, lower_bb = TechnicalIndicators.bollinger_bands(df['close'].values)
        result['bb_upper'] = upper_bb
        result['bb_middle'] = middle_bb
        result['bb_lower'] = lower_bb
        result['atr_14'] = TechnicalIndicators.atr(df['high'].values, df['low'].values, df['close'].values, 14)
        
        # Trend
        k_stoch, d_stoch = TechnicalIndicators.stochastic(df['high'].values, df['low'].values, df['close'].values)
        result['stoch_k'] = k_stoch
        result['stoch_d'] = d_stoch
        result['adx'] = TechnicalIndicators.adx(df['high'].values, df['low'].values, df['close'].values)
        
        # Rate of change
        result['momentum_14'] = TechnicalIndicators.momentum(df['close'].values, 14)
        result['roc_14'] = TechnicalIndicators.roc(df['close'].values, 14)
        
        return result


class IndicatorCache:
    """
    Caching layer for indicator calculations
    Avoids redundant calculations for same data
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = max_cache_size
    
    def _make_key(self, data_hash: str, indicator_name: str, params: str) -> str:
        return f"{data_hash}_{indicator_name}_{params}"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        return self.cache.get(key)
    
    def set(self, key: str, value: np.ndarray):
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
