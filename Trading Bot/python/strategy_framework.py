# encoding: utf-8
"""
Vectorized Strategy Framework
High-performance strategy execution framework with NumPy/Pandas optimization.
Supports event-driven and vectorized execution modes.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
import warnings


# ============================================================================
# ENUMS AND ABSTRACT BASE CLASSES
# ============================================================================

class StrategyMode(Enum):
    """Strategy execution mode."""
    EVENT_DRIVEN = 'event_driven'      # Process bar by bar
    VECTORIZED = 'vectorized'          # Batch process all bars


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str = "Strategy"):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name for logging/reporting
        """
        self.name = name
        self.mode = StrategyMode.EVENT_DRIVEN
        self.position_size = 1.0  # Default: 1 unit
        self.is_initialized = False
    
    @abstractmethod
    def on_bar(self, bar: pd.Series, symbol: str) -> Dict:
        """
        Process a single bar and generate signals.
        
        Args:
            bar: OHLCV bar as pandas Series with columns: open, high, low, close, volume
            symbol: Asset symbol
        
        Returns:
            Dict with keys:
                - 'action': 'BUY', 'SELL', 'HOLD', 'CLOSE'
                - 'price': execution price (optional, defaults to close)
                - 'quantity': order quantity (optional, defaults to position_size)
        """
        pass
    
    def initialize(self, data: pd.DataFrame, **kwargs):
        """
        Initialize strategy with historical data (for warm-up).
        
        Args:
            data: Historical OHLCV data
            **kwargs: Additional parameters
        """
        self.is_initialized = True
    
    def reset(self):
        """Reset strategy state between backtests."""
        self.is_initialized = False


# ============================================================================
# TECHNICAL INDICATORS (OPTIMIZED FOR VECTORIZED OPERATIONS)
# ============================================================================

class TechnicalIndicators:
    """Vectorized technical indicator calculations."""
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        if len(prices) < period:
            return np.full_like(prices, np.nan, dtype=float)
        
        sma_values = np.convolve(prices, np.ones(period) / period, mode='valid')
        result = np.full_like(prices, np.nan, dtype=float)
        result[period-1:] = sma_values
        return result
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(prices) < period:
            return np.full_like(prices, np.nan, dtype=float)
        
        result = np.full_like(prices, np.nan, dtype=float)
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            result[i] = prices[i] * multiplier + result[i-1] * (1 - multiplier)
        
        return result
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return np.full_like(prices, np.nan, dtype=float)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi_values = np.zeros_like(prices, dtype=float)
        rsi_values[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up / down if down != 0 else 0
            rsi_values[i] = 100 - 100 / (1 + rs)
        
        result = np.full_like(prices, np.nan, dtype=float)
        result[period:] = rsi_values[period:]
        return result
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal)
        
        # Align signal line with macd line
        result_signal = np.full_like(macd_line, np.nan, dtype=float)
        valid_idx = ~np.isnan(macd_line)
        valid_count = np.sum(valid_idx)
        if len(signal_line) > 0:
            result_signal[valid_idx][-len(signal_line):] = signal_line
        
        histogram = macd_line - result_signal
        
        return macd_line, result_signal, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands."""
        sma_values = TechnicalIndicators.sma(prices, period)
        
        # Calculate rolling standard deviation
        std_values = np.zeros_like(prices, dtype=float)
        for i in range(period-1, len(prices)):
            std_values[i] = np.std(prices[i-period+1:i+1])
        
        upper_band = sma_values + (std_values * std_dev)
        lower_band = sma_values - (std_values * std_dev)
        
        return upper_band, sma_values, lower_band
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range."""
        tr = np.zeros_like(high, dtype=float)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        return TechnicalIndicators.ema(tr, period)
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator."""
        lowest_low = np.zeros_like(close, dtype=float)
        highest_high = np.zeros_like(close, dtype=float)
        
        for i in range(period-1, len(close)):
            lowest_low[i] = np.min(low[i-period+1:i+1])
            highest_high[i] = np.max(high[i-period+1:i+1])
        
        # Calculate K
        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        k_line = TechnicalIndicators.sma(k_raw, smooth_k)
        
        # Calculate D
        d_line = TechnicalIndicators.sma(k_line[~np.isnan(k_line)], smooth_d)
        
        # Align D line
        result_d = np.full_like(k_line, np.nan, dtype=float)
        valid_idx = ~np.isnan(k_line)
        valid_count = np.sum(valid_idx)
        if len(d_line) > 0:
            result_d[valid_idx][-len(d_line):] = d_line
        
        return k_line, result_d
    
    @staticmethod
    def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Price Momentum."""
        momentum = np.zeros_like(prices, dtype=float)
        momentum[period:] = prices[period:] - prices[:-period]
        momentum[:period] = np.nan
        return momentum
    
    @staticmethod
    def rate_of_change(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of Change (ROC)."""
        roc = np.zeros_like(prices, dtype=float)
        roc[period:] = ((prices[period:] - prices[:-period]) / prices[:-period]) * 100
        roc[:period] = np.nan
        return roc


# ============================================================================
# EXAMPLE STRATEGIES
# ============================================================================

class MovingAverageCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    Signals: BUY when SMA_fast > SMA_slow, SELL when SMA_fast < SMA_slow
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50, position_size: float = 1.0):
        super().__init__("MA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        self.in_position = False
    
    def on_bar(self, bar: pd.Series, symbol: str) -> Dict:
        """Generate signals based on MA crossover."""
        result = {'action': 'HOLD', 'price': bar['close'], 'quantity': self.position_size}
        
        # Need enough historical data
        if len(bar.history) < self.slow_period + 1:
            return result
        
        closes = bar.history['close'].values
        fast_sma = TechnicalIndicators.sma(closes, self.fast_period)[-1]
        slow_sma = TechnicalIndicators.sma(closes, self.slow_period)[-1]
        
        if np.isnan(fast_sma) or np.isnan(slow_sma):
            return result
        
        current_close = closes[-1]
        
        if fast_sma > slow_sma and not self.in_position:
            result['action'] = 'BUY'
            self.in_position = True
        elif fast_sma < slow_sma and self.in_position:
            result['action'] = 'SELL'
            self.in_position = False
        
        return result


class BinanceDayTrade(BaseStrategy):
    """
    Binance Spot Day-Trade Strategy (EMA+ATR)

    Logic (simple, robust base):
    - Use short and long EMA crossover to identify trend intraday
    - Confirm with ATR-based volatility to size stops
    - Enter on EMA short crossing above EMA long (momentum) and exit when
      short crosses below long or stop/take-profit hit
    - Designed for spot (no leverage) and small position sizing
    """

    def __init__(self,
                 ema_short: int = 8,
                 ema_long: int = 21,
                 atr_period: int = 14,
                 atr_mult: float = 1.5,
                 take_profit_mult: float = 3.0,
                 position_size: float = 0.01):
        super().__init__("Binance_DayTrade")
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.take_profit_mult = take_profit_mult
        self.position_size = position_size
        self.in_position = False
        self.entry_price = None
        self.stop_price = None
        self.take_profit = None

    def on_bar(self, bar: pd.Series, symbol: str) -> Dict:
        result = {'action': 'HOLD', 'price': bar['close'], 'quantity': self.position_size}

        # Require warm history
        history = getattr(bar, 'history', None)
        if history is None or len(history) < max(self.ema_long, self.atr_period) + 2:
            return result

        closes = history['close'].values
        highs = history['high'].values
        lows = history['low'].values

        ema_short_arr = TechnicalIndicators.ema(closes, self.ema_short)
        ema_long_arr = TechnicalIndicators.ema(closes, self.ema_long)
        atr_arr = TechnicalIndicators.atr(highs, lows, closes, self.atr_period)

        cur_ema_short = ema_short_arr[-1]
        cur_ema_long = ema_long_arr[-1]
        prev_ema_short = ema_short_arr[-2]
        prev_ema_long = ema_long_arr[-2]
        cur_atr = atr_arr[-1]

        if np.isnan(cur_ema_short) or np.isnan(cur_ema_long) or np.isnan(cur_atr):
            return result

        # Entry condition: short EMA crossed above long EMA (momentum)
        crossed_up = (prev_ema_short <= prev_ema_long) and (cur_ema_short > cur_ema_long)
        crossed_down = (prev_ema_short >= prev_ema_long) and (cur_ema_short < cur_ema_long)

        # If not in position and momentum appears, enter long
        if not self.in_position and crossed_up:
            entry = float(bar['close'])
            stop = entry - (self.atr_mult * float(cur_atr))
            tp = entry + (self.take_profit_mult * float(cur_atr))

            self.in_position = True
            self.entry_price = entry
            self.stop_price = max(stop, 0.0)
            self.take_profit = tp

            result.update({'action': 'BUY', 'price': entry, 'quantity': self.position_size})
            return result

        # If in position, check stop/take-profit and exit on crossover down
        if self.in_position:
            cur_price = float(bar['close'])

            # Stop loss
            if self.stop_price is not None and cur_price <= self.stop_price:
                # exit at market
                result.update({'action': 'SELL', 'price': cur_price, 'quantity': self.position_size})
                self._reset_position()
                return result

            # Take profit
            if self.take_profit is not None and cur_price >= self.take_profit:
                result.update({'action': 'SELL', 'price': cur_price, 'quantity': self.position_size})
                self._reset_position()
                return result

            # EMA trend flip
            if crossed_down:
                result.update({'action': 'SELL', 'price': cur_price, 'quantity': self.position_size})
                self._reset_position()
                return result

        return result

    def _reset_position(self):
        self.in_position = False
        self.entry_price = None
        self.stop_price = None
        self.take_profit = None


# ============================================================================
# STRATEGY EXECUTOR
# ============================================================================

class StrategyExecutor:
    """
    Executes strategies on market data.
    Bridges strategy logic with the backtesting engine.
    """
    
    def __init__(self, strategy: BaseStrategy, engine):
        """
        Initialize executor.
        
        Args:
            strategy: Strategy instance to execute
            engine: Backtesting engine instance
        """
        self.strategy = strategy
        self.engine = engine
        self.signals_history = []
        self.bar_count = 0
    
    def execute(self, data: pd.DataFrame, symbols: List[str]) -> Dict:
        """
        Execute strategy on provided data.
        
        Args:
            data: OHLCV data (MultiIndex DataFrame with symbol level)
            symbols: List of symbols to trade
        
        Returns:
            Dict with execution results including equity curve and trades
        """
        self.engine.reset()
        self.signals_history.clear()
        self.bar_count = 0
        
        # Warm up strategy with historical data if needed
        self.strategy.initialize(data)
        
        # Process data
        for timestamp in data.index.get_level_values(0).unique():
            bar_data = data.loc[timestamp]
            
            price_dict = {}
            
            for symbol in symbols:
                if symbol in bar_data.index:
                    bar = bar_data.loc[symbol]
                    price_dict[symbol] = bar['close']
                    
                    # Generate signal
                    signal = self.strategy.on_bar(bar, symbol)
                    
                    # Execute signal
                    if signal['action'] == 'BUY':
                        self.engine.submit_order(
                            symbol=symbol,
                            side=self.engine.orders[0].__class__.__bases__[0].__dict__.get('BUY', 'buy'),
                            quantity=signal.get('quantity', self.strategy.position_size),
                            price=signal.get('price', bar['close'])
                        )
                    elif signal['action'] == 'SELL':
                        self.engine.submit_order(
                            symbol=symbol,
                            side='sell',
                            quantity=signal.get('quantity', self.strategy.position_size),
                            price=signal.get('price', bar['close'])
                        )
                    
                    self.signals_history.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'signal': signal['action'],
                        'price': bar['close']
                    })
            
            # Step the engine
            if price_dict:
                self.engine.step(timestamp, price_dict)
            
            self.bar_count += 1
        
        # Close all positions at the end
        final_prices = {symbol: data.loc[data.index.get_level_values(0).max(), symbol]['close'] 
                       for symbol in symbols if symbol in data.loc[data.index.get_level_values(0).max()].index}
        self.engine.close_all_positions(final_prices)
        
        # Calculate metrics
        metrics = self.engine.calculate_metrics()
        
        return {
            'metrics': metrics,
            'equity_curve': self.engine.get_equity_curve(),
            'trades': self.engine.get_trade_log(),
            'signals': pd.DataFrame(self.signals_history)
        }
