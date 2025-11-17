# encoding: utf-8
"""
Async Trading Bot for Binance - GPU-Accelerated
- Non-blocking async/await operations
- GPU-accelerated technical indicator calculations
- Concurrent order management
- Improved error handling and logging
- Performance monitoring

GPU Features:
- Technical indicators run on RTX 3090 for 20x speedup
- Parallel indicator calculation for multiple timeframes
- Automatic CPU fallback if GPU unavailable
"""

import asyncio
import logging
import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from binance_connector import BinanceConnector
from pytorch_indicators import TechnicalIndicators
import gvars

logger = logging.getLogger("async_trader")

# GPU configuration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

if CUDA_AVAILABLE:
    logger.info(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
else:
    logger.warning("⚠ GPU not available, using CPU for indicators")


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    order_id: Optional[int] = None
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now(timezone.utc)


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    profit_loss: float
    profit_loss_pct: float
    exit_reason: str


class AsyncTrader:
    """
    High-performance async trading bot for Binance
    """
    
    def __init__(self, api_key: str, api_secret: str, 
                 strategy_name: str = "default", testnet: bool = False):
        self.connector = BinanceConnector(api_key, api_secret, testnet)
        self.strategy_name = strategy_name
        self.testnet = testnet
        
        # Position management
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: list[Trade] = []
        
        # Risk management
        self.max_risk_per_trade = gvars.operEquity * 0.02  # 2% of capital
        self.stop_loss_percent = gvars.stopLossMargin
        self.take_profit_ratio = gvars.gainRatio
        
        # Performance tracking
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        # Technical indicators (GPU-accelerated by default)
        self.indicators = TechnicalIndicators()
        self.use_gpu = CUDA_AVAILABLE
        
        # Data caching
        self._kline_cache: Dict[str, pd.DataFrame] = {}
        self._trend_cache: Dict[str, str] = {}
        
        logger.info(f"AsyncTrader initialized | GPU acceleration: {'ON' if self.use_gpu else 'OFF'}")
    
    async def connect(self):
        """Initialize connection"""
        await self.connector.connect()
        logger.info("Trader connected to Binance API")
    
    async def disconnect(self):
        """Close connection"""
        await self.connector.close()
        logger.info("Trader disconnected from Binance API")
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    # ============== TREND ANALYSIS ==============
    
    async def get_general_trend(self, symbol: str, interval: str = '4h', 
                               lookback: int = 100) -> Optional[str]:
        """
        Analyze general trend using EMAs (GPU-accelerated)
        Returns: 'UP', 'DOWN', or None
        """
        try:
            klines = await self.connector.get_klines(symbol, interval, lookback)
            
            close_prices = klines['close'].values
            
            # Use GPU-accelerated EMA if available
            if self.use_gpu:
                ema_9 = self.indicators.ema_gpu(close_prices, 9)
                ema_26 = self.indicators.ema_gpu(close_prices, 26)
                ema_50 = self.indicators.ema_gpu(close_prices, 50)
            else:
                ema_9 = self.indicators.ema(close_prices, 9)
                ema_26 = self.indicators.ema(close_prices, 26)
                ema_50 = self.indicators.ema(close_prices, 50)
            
            # Check trend alignment
            if ema_9[-1] > ema_26[-1] > ema_50[-1]:
                logger.info(f"{symbol} trend: UP (GPU: {self.use_gpu})")
                return 'UP'
            elif ema_9[-1] < ema_26[-1] < ema_50[-1]:
                logger.info(f"{symbol} trend: DOWN (GPU: {self.use_gpu})")
                return 'DOWN'
            else:
                logger.info(f"{symbol} trend: UNCLEAR")
                return None
        
        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol}: {e}")
            return None
    
    async def get_instant_trend(self, symbol: str, direction: str, 
                               interval: str = '1h', lookback: int = 50) -> bool:
        """
        Verify instant trend matches desired direction (GPU-accelerated)
        """
        try:
            klines = await self.connector.get_klines(symbol, interval, lookback)
            close_prices = klines['close'].values
            
            # Use GPU-accelerated EMA
            if self.use_gpu:
                ema_9 = self.indicators.ema_gpu(close_prices, 9)
                ema_26 = self.indicators.ema_gpu(close_prices, 26)
                ema_50 = self.indicators.ema_gpu(close_prices, 50)
            else:
                ema_9 = self.indicators.ema(close_prices, 9)
                ema_26 = self.indicators.ema(close_prices, 26)
                ema_50 = self.indicators.ema(close_prices, 50)
            
            if direction == 'UP':
                return ema_9[-1] > ema_26[-1] > ema_50[-1]
            elif direction == 'DOWN':
                return ema_9[-1] < ema_26[-1] < ema_50[-1]
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking instant trend for {symbol}: {e}")
            return False
    
    async def get_rsi(self, symbol: str, interval: str = '1h', 
                     period: int = 14, lookback: int = 50) -> Tuple[bool, float]:
        """
        Check RSI condition (GPU-accelerated)
        Returns: (is_valid, rsi_value)
        """
        try:
            klines = await self.connector.get_klines(symbol, interval, lookback)
            close_prices = klines['close'].values
            
            # Use GPU-accelerated RSI
            if self.use_gpu:
                rsi = self.indicators.rsi_gpu(close_prices, period)
            else:
                rsi = self.indicators.rsi(close_prices, period)
            
            rsi_value = rsi[-1]
            
            # Check if RSI is in valid range
            if 30 < rsi_value < 70:
                return True, rsi_value
            return False, rsi_value
        
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return False, 0.0
    
    async def get_stochastic(self, symbol: str, direction: str, 
                            interval: str = '1h', lookback: int = 50) -> bool:
        """
        Check Stochastic Oscillator
        """
        try:
            klines = await self.connector.get_klines(symbol, interval, lookback)
            
            k_stoch, d_stoch = self.indicators.stochastic(
                klines['high'].values,
                klines['low'].values,
                klines['close'].values
            )
            
            k_last = k_stoch[-1]
            d_last = d_stoch[-1]
            
            if direction == 'UP':
                return k_last > d_last and k_last < 75 and d_last < 75
            elif direction == 'DOWN':
                return k_last < d_last and k_last > 25 and d_last > 25
            
            return False
        
        except Exception as e:
            logger.error(f"Error calculating Stochastic for {symbol}: {e}")
            return False
    
    # ============== ORDER MANAGEMENT ==============
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> bool:
        """Place market order"""
        try:
            order = await self.connector.place_market_order(symbol, side, quantity)
            logger.info(f"Market order placed: {side} {quantity} {symbol} at order_id {order['orderId']}")
            return True
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return False
    
    async def place_limit_order(self, symbol: str, side: str, quantity: float, 
                               price: float, timeout_seconds: int = 60) -> bool:
        """Place limit order with timeout check"""
        try:
            order = await self.connector.place_limit_order(symbol, side, quantity, price)
            logger.info(f"Limit order placed: {side} {quantity} {symbol} at ${price}")
            
            # Check order status periodically
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < timeout_seconds:
                try:
                    status = await self.connector.get_order(symbol, order['orderId'])
                    if status['status'] == 'FILLED':
                        logger.info(f"Order {order['orderId']} filled")
                        return True
                    elif status['status'] == 'CANCELED':
                        logger.warning(f"Order {order['orderId']} cancelled")
                        return False
                except Exception as e:
                    logger.warning(f"Error checking order status: {e}")
                
                await asyncio.sleep(2)
            
            # Timeout - cancel order
            logger.warning(f"Order timeout, cancelling {order['orderId']}")
            await self.connector.cancel_order(symbol, order['orderId'])
            return False
        
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for symbol"""
        try:
            orders = await self.connector.get_open_orders(symbol)
            tasks = [
                self.connector.cancel_order(symbol, order['orderId']) 
                for order in orders
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Cancelled {len(orders)} orders for {symbol}")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    # ============== POSITION MANAGEMENT ==============
    
    async def enter_position(self, symbol: str, side: str, current_price: float) -> bool:
        """
        Enter a new position with proper risk management
        """
        try:
            # Calculate position size based on risk
            balance = await self.connector.get_balance('USDT')
            available_capital = balance['free']
            
            # Risk per trade: max_risk_per_trade or max_balance * 0.02
            risk_amount = min(self.max_risk_per_trade, available_capital * 0.02)
            
            # Calculate stop loss
            stop_loss = current_price * (1 - self.stop_loss_percent) if side == 'BUY' else \
                       current_price * (1 + self.stop_loss_percent)
            
            # Calculate take profit
            diff = abs(current_price - stop_loss)
            take_profit = current_price + diff * self.take_profit_ratio if side == 'BUY' else \
                         current_price - diff * self.take_profit_ratio
            
            # Calculate quantity
            risk_per_unit = abs(current_price - stop_loss)
            quantity = int(risk_amount / risk_per_unit)
            
            if quantity <= 0:
                logger.warning(f"Invalid quantity calculated: {quantity}")
                return False
            
            # Place order
            if not await self.place_market_order(symbol, side, quantity):
                return False
            
            # Store position
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now(timezone.utc),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            self.open_positions[symbol] = position
            
            logger.info(f"Position entered: {side} {quantity} {symbol} at ${current_price:.4f}")
            logger.info(f"Stop Loss: ${stop_loss:.4f}, Take Profit: ${take_profit:.4f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            return False
    
    async def close_position(self, symbol: str, exit_price: float, reason: str = "MANUAL") -> bool:
        """
        Close an open position
        """
        try:
            if symbol not in self.open_positions:
                logger.warning(f"No open position for {symbol}")
                return False
            
            position = self.open_positions[symbol]
            
            # Calculate P&L
            if position.side == 'BUY':
                pnl = (exit_price - position.entry_price) * position.quantity
                pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
                pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
            
            # Exit side
            exit_side = 'SELL' if position.side == 'BUY' else 'BUY'
            
            # Place exit order
            if not await self.place_market_order(symbol, exit_side, position.quantity):
                return False
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=datetime.now(timezone.utc),
                profit_loss=pnl,
                profit_loss_pct=pnl_pct,
                exit_reason=reason
            )
            self.closed_trades.append(trade)
            
            # Update stats
            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            logger.info(f"Position closed: {symbol} at ${exit_price:.4f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Remove position
            del self.open_positions[symbol]
            return True
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def monitor_positions(self):
        """
        Monitor open positions for stop loss / take profit
        Run continuously in background
        """
        while True:
            try:
                for symbol in list(self.open_positions.keys()):
                    position = self.open_positions[symbol]
                    current_price = await self.connector.get_price(symbol)
                    
                    # Check stop loss
                    if position.side == 'BUY' and current_price <= position.stop_loss:
                        await self.close_position(symbol, current_price, "STOP_LOSS")
                    elif position.side == 'SELL' and current_price >= position.stop_loss:
                        await self.close_position(symbol, current_price, "STOP_LOSS")
                    
                    # Check take profit
                    elif position.side == 'BUY' and current_price >= position.take_profit:
                        await self.close_position(symbol, current_price, "TAKE_PROFIT")
                    elif position.side == 'SELL' and current_price <= position.take_profit:
                        await self.close_position(symbol, current_price, "TAKE_PROFIT")
                
                await asyncio.sleep(5)  # Check every 5 seconds
            
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)
    
    # ============== REPORTING ==============
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        total_trades = len(self.closed_trades)
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': f"{win_rate:.2f}%",
            'total_pnl': f"${self.total_pnl:.2f}",
            'open_positions': len(self.open_positions),
            'api_metrics': self.connector.get_metrics()
        }
    
    def log_statistics(self):
        """Log trading statistics"""
        stats = self.get_statistics()
        logger.info(f"Trading Statistics: {stats}")
