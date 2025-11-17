# encoding: utf-8
"""
High-Performance Backtesting Engine
Core backtesting system with optimized data structures for high-frequency tick/minute data.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class OrderType(Enum):
    """Order types supported by the backtesting engine."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'


class OrderSide(Enum):
    """Order direction."""
    BUY = 'buy'
    SELL = 'sell'


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = 'pending'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


@dataclass
class Order:
    """Represents a trading order."""
    order_id: int
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float  # limit price or entry price for market orders
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    execution_timestamp: Optional[pd.Timestamp] = None
    commission: float = 0.0

    def __hash__(self):
        return hash(self.order_id)


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_price: float = 0.0
    trough_price: float = 0.0

    def update_market_price(self, current_price: float, position_size: Optional[float] = None):
        """Update unrealized P&L based on current market price."""
        if position_size is None:
            position_size = self.quantity
        
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * position_size
            self.peak_price = max(self.peak_price, current_price)
            self.trough_price = self.trough_price or current_price
            self.trough_price = min(self.trough_price, current_price)
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - current_price) * position_size
            self.trough_price = min(self.trough_price, current_price) if self.trough_price else current_price
            self.peak_price = self.peak_price or current_price
            self.peak_price = max(self.peak_price, current_price)


@dataclass
class BacktestMetrics:
    """Performance metrics for backtesting results."""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Risk metrics
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0


# ============================================================================
# TRANSACTION COSTS MODULE
# ============================================================================

class TransactionCostModel:
    """Handles commission, slippage, and spread calculations."""
    
    def __init__(self, 
                 commission_type: str = 'fixed',  # 'fixed' or 'percentage'
                 commission_amount: float = 0.0,
                 slippage_type: str = 'fixed',    # 'fixed' or 'percentage'
                 slippage_amount: float = 0.0,
                 bid_ask_spread: float = 0.0):
        """
        Initialize transaction cost model.
        
        Args:
            commission_type: 'fixed' (per trade) or 'percentage' (of trade value)
            commission_amount: Commission value (dollar amount or percentage)
            slippage_type: 'fixed' or 'percentage'
            slippage_amount: Slippage value (ticks or percentage)
            bid_ask_spread: Half-spread in percentage (applied to all orders)
        """
        self.commission_type = commission_type
        self.commission_amount = commission_amount
        self.slippage_type = slippage_type
        self.slippage_amount = slippage_amount
        self.bid_ask_spread = bid_ask_spread
    
    def calculate_execution_price(self, 
                                 order_price: float,
                                 side: OrderSide,
                                 order_type: OrderType = OrderType.MARKET) -> Tuple[float, float]:
        """
        Calculate actual execution price including slippage and bid-ask spread.
        
        Args:
            order_price: Original order price
            side: BUY or SELL
            order_type: Type of order (affects slippage calculation)
        
        Returns:
            Tuple of (execution_price, slippage_cost)
        """
        slippage = 0.0
        
        # Add bid-ask spread (always against trader)
        spread_cost = order_price * self.bid_ask_spread
        
        # Add slippage
        if order_type == OrderType.MARKET:
            if self.slippage_type == 'percentage':
                slippage = order_price * self.slippage_amount
            else:  # fixed
                slippage = self.slippage_amount
        
        # Apply slippage and spread in unfavorable direction
        if side == OrderSide.BUY:
            execution_price = order_price + spread_cost + slippage
        else:  # SELL
            execution_price = order_price - spread_cost - slippage
        
        return execution_price, spread_cost + slippage
    
    def calculate_commission(self, trade_value: float, side: OrderSide) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            trade_value: Value of the trade (quantity * price)
            side: BUY or SELL
        
        Returns:
            Commission cost
        """
        if self.commission_type == 'percentage':
            return trade_value * self.commission_amount
        else:  # fixed
            return self.commission_amount


# ============================================================================
# LOOK-AHEAD BIAS PREVENTION
# ============================================================================

class TemporalDataBuffer:
    """Ensures no future data leakage in strategy execution."""
    
    def __init__(self, bar_count: int = 1):
        """
        Initialize temporal buffer.
        
        Args:
            bar_count: Number of bars to buffer (minimum 1)
        """
        self.bar_count = max(1, bar_count)
        self.buffer = []
        self.current_index = 0
    
    def add_bar(self, bar: pd.Series) -> Optional[pd.Series]:
        """
        Add a new bar to buffer and get safe (non-leaking) bar for strategy.
        
        Args:
            bar: OHLCV bar data
        
        Returns:
            Safe bar for strategy execution (delayed by bar_count), or None if insufficient data
        """
        self.buffer.append(bar)
        
        if len(self.buffer) > self.bar_count:
            safe_bar = self.buffer.pop(0)
            return safe_bar
        
        return None
    
    def flush(self) -> List[pd.Series]:
        """Flush remaining safe bars from buffer (for end of backtest)."""
        result = []
        while len(self.buffer) > self.bar_count:
            result.append(self.buffer.pop(0))
        return result
    
    def reset(self):
        """Reset the buffer."""
        self.buffer = []
        self.current_index = 0


# ============================================================================
# CORE BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """High-performance backtesting engine for trading strategies."""
    
    def __init__(self,
                 initial_capital: float = 100000,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 prevent_lookahead_bias: bool = True,
                 lookahead_bars: int = 1):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital in dollars
            transaction_cost_model: Model for commissions, slippage, and spreads
            prevent_lookahead_bias: Whether to implement look-ahead bias prevention
            lookahead_bars: Number of bars to delay for bias prevention
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.prevent_lookahead_bias = prevent_lookahead_bias
        self.temporal_buffer = TemporalDataBuffer(lookahead_bars) if prevent_lookahead_bias else None
        
        # Order tracking
        self.orders: Dict[int, Order] = {}
        self.next_order_id = 1
        self.pending_orders: List[Order] = []
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        
        # Equity history for metrics calculation
        self.equity_history = []
        self.drawdown_history = []
        
        # Trade history
        self.closed_trades = []
        self.trade_pnl = []
        
        # Current state
        self.current_timestamp: Optional[pd.Timestamp] = None
        self.current_prices: Dict[str, float] = {}
    
    def submit_order(self,
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    price: Optional[float] = None,
                    order_type: OrderType = OrderType.MARKET,
                    stop_price: Optional[float] = None) -> Optional[int]:
        """
        Submit a trading order.
        
        Args:
            symbol: Asset symbol
            side: BUY or SELL
            quantity: Number of shares/contracts
            price: Limit price (required for LIMIT orders, optional for MARKET)
            order_type: Type of order
            stop_price: Stop price (for STOP orders)
        
        Returns:
            Order ID if successful, None otherwise
        """
        if quantity <= 0:
            warnings.warn(f"Invalid quantity: {quantity}")
            return None
        
        if order_type == OrderType.LIMIT and price is None:
            warnings.warn("LIMIT orders require a price")
            return None
        
        # Use current price if not specified for market orders
        if price is None:
            price = self.current_prices.get(symbol, 0)
            if price == 0:
                warnings.warn(f"No price available for {symbol}")
                return None
        
        # Check if we have sufficient capital for buy orders
        if side == OrderSide.BUY:
            required_capital = price * quantity
            if required_capital > self.available_capital:
                warnings.warn(f"Insufficient capital: required {required_capital}, available {self.available_capital}")
                return None
        
        # Create order
        order = Order(
            order_id=self.next_order_id,
            timestamp=self.current_timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        self.orders[order.order_id] = order
        self.pending_orders.append(order)
        self.next_order_id += 1
        
        return order.order_id
    
    def _process_order(self, order: Order, current_price: float) -> bool:
        """
        Process a pending order for execution.
        
        Args:
            order: Order to process
            current_price: Current market price
        
        Returns:
            True if order was filled, False otherwise
        """
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, current_price)
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, current_price)
        elif order.order_type == OrderType.STOP:
            return self._execute_stop_order(order, current_price)
        
        return False
    
    def _execute_market_order(self, order: Order, current_price: float) -> bool:
        """Execute a market order."""
        execution_price, slippage = self.transaction_cost_model.calculate_execution_price(
            current_price, order.side, OrderType.MARKET
        )
        
        trade_value = execution_price * order.quantity
        commission = self.transaction_cost_model.calculate_commission(trade_value, order.side)
        
        # Check capital availability
        if order.side == OrderSide.BUY and trade_value + commission > self.available_capital:
            return False
        
        # Execute order
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.status = OrderStatus.FILLED
        order.execution_timestamp = self.current_timestamp
        
        # Update position
        self._update_position(order, execution_price)
        
        # Update capital
        if order.side == OrderSide.BUY:
            self.available_capital -= (trade_value + commission)
        else:  # SELL
            self.available_capital += (trade_value - commission)
        
        return True
    
    def _execute_limit_order(self, order: Order, current_price: float) -> bool:
        """Execute a limit order."""
        # Limit orders fill if price reaches limit
        should_fill = False
        
        if order.side == OrderSide.BUY and current_price <= order.price:
            should_fill = True
        elif order.side == OrderSide.SELL and current_price >= order.price:
            should_fill = True
        
        if not should_fill:
            return False
        
        # Fill at limit price (or better in real scenarios, here we use limit)
        execution_price = order.price
        trade_value = execution_price * order.quantity
        commission = self.transaction_cost_model.calculate_commission(trade_value, order.side)
        
        # Check capital availability
        if order.side == OrderSide.BUY and trade_value + commission > self.available_capital:
            return False
        
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.status = OrderStatus.FILLED
        order.execution_timestamp = self.current_timestamp
        
        self._update_position(order, execution_price)
        
        if order.side == OrderSide.BUY:
            self.available_capital -= (trade_value + commission)
        else:
            self.available_capital += (trade_value - commission)
        
        return True
    
    def _execute_stop_order(self, order: Order, current_price: float) -> bool:
        """Execute a stop order."""
        should_trigger = False
        
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            should_trigger = True
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            should_trigger = True
        
        if not should_trigger:
            return False
        
        # Convert to market order and execute
        execution_price, slippage = self.transaction_cost_model.calculate_execution_price(
            current_price, order.side, OrderType.MARKET
        )
        
        trade_value = execution_price * order.quantity
        commission = self.transaction_cost_model.calculate_commission(trade_value, order.side)
        
        if order.side == OrderSide.BUY and trade_value + commission > self.available_capital:
            return False
        
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.status = OrderStatus.FILLED
        order.execution_timestamp = self.current_timestamp
        
        self._update_position(order, execution_price)
        
        if order.side == OrderSide.BUY:
            self.available_capital -= (trade_value + commission)
        else:
            self.available_capital += (trade_value - commission)
        
        return True
    
    def _update_position(self, order: Order, execution_price: float):
        """Update or create a position based on filled order."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            if order.side == OrderSide.SELL:
                # Short sale
                position = Position(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=order.filled_quantity,
                    entry_price=execution_price,
                    entry_timestamp=order.execution_timestamp
                )
            else:
                position = Position(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=order.filled_quantity,
                    entry_price=execution_price,
                    entry_timestamp=order.execution_timestamp
                )
            self.positions[symbol] = position
        else:
            position = self.positions[symbol]
            
            # Check if order is in same direction (adding to position) or opposite (closing)
            if order.side == position.side:
                # Add to position
                new_qty = position.quantity + order.filled_quantity
                position.entry_price = (
                    (position.entry_price * position.quantity + 
                     execution_price * order.filled_quantity) / new_qty
                )
                position.quantity = new_qty
            else:
                # Reduce or close position
                if order.filled_quantity >= position.quantity:
                    # Position closed or reversed
                    pnl = 0
                    if position.side == OrderSide.BUY:
                        pnl = (execution_price - position.entry_price) * position.quantity
                    else:
                        pnl = (position.entry_price - execution_price) * position.quantity
                    
                    self.trade_pnl.append({
                        'symbol': symbol,
                        'entry_price': position.entry_price,
                        'exit_price': execution_price,
                        'quantity': position.quantity,
                        'pnl': pnl,
                        'entry_time': position.entry_timestamp,
                        'exit_time': order.execution_timestamp
                    })
                    
                    if order.filled_quantity > position.quantity:
                        # Reverse position
                        excess_qty = order.filled_quantity - position.quantity
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            side=order.side,
                            quantity=excess_qty,
                            entry_price=execution_price,
                            entry_timestamp=order.execution_timestamp
                        )
                    else:
                        del self.positions[symbol]
                else:
                    # Reduce position
                    position.quantity -= order.filled_quantity
    
    def update_prices(self, symbol: str, price: float, ohlcv: Optional[pd.Series] = None):
        """
        Update current price for a symbol.
        
        Args:
            symbol: Asset symbol
            price: Current price
            ohlcv: Optional OHLCV data for the bar
        """
        self.current_prices[symbol] = price
        
        # Update position market values
        if symbol in self.positions:
            position = self.positions[symbol]
            position.update_market_price(price, position.quantity)
    
    def step(self, timestamp: pd.Timestamp, price_data: Dict[str, float]):
        """
        Execute one timestep of the backtest.
        
        Args:
            timestamp: Current timestamp
            price_data: Dict of {symbol: price} for current bar
        """
        self.current_timestamp = timestamp
        
        # Update all prices
        for symbol, price in price_data.items():
            self.update_prices(symbol, price)
        
        # Process pending orders
        filled_orders = []
        for order in self.pending_orders[:]:
            if self._process_order(order, price_data.get(order.symbol, 0)):
                filled_orders.append(order)
                self.pending_orders.remove(order)
        
        # Update equity
        self._update_equity()
    
    def _update_equity(self):
        """Update equity history and drawdown tracking."""
        equity = self.available_capital
        
        for position in self.positions.values():
            symbol = position.symbol
            current_price = self.current_prices.get(symbol, position.entry_price)
            position.update_market_price(current_price, position.quantity)
            equity += position.quantity * current_price if position.side == OrderSide.BUY else -position.quantity * current_price + position.quantity * position.entry_price
        
        self.equity_history.append(equity)
        
        # Track drawdown
        if self.equity_history:
            peak = max(self.equity_history)
            drawdown = (peak - equity) / peak if peak > 0 else 0
            self.drawdown_history.append(drawdown)
    
    def close_all_positions(self, close_prices: Dict[str, float]):
        """
        Close all open positions at specified prices (used at end of backtest).
        
        Args:
            close_prices: Dict of {symbol: price} for closing positions
        """
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            close_price = close_prices.get(symbol, self.current_prices.get(symbol, 0))
            
            # Create and process closing order
            side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            order = Order(
                order_id=self.next_order_id,
                timestamp=self.current_timestamp,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                price=close_price
            )
            
            self._execute_market_order(order, close_price)
            self.next_order_id += 1
    
    def calculate_metrics(self) -> BacktestMetrics:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            BacktestMetrics object with all calculated metrics
        """
        metrics = BacktestMetrics()
        
        if not self.equity_history:
            return metrics
        
        equity_array = np.array(self.equity_history)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Basic return metrics
        final_equity = equity_array[-1]
        metrics.total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Annual return (assuming 252 trading days)
        if len(equity_array) > 1:
            days = len(equity_array)  # Approximate
            years = days / 252
            if years > 0:
                metrics.annual_return = (final_equity / self.initial_capital) ** (1 / years) - 1
        
        # Sharpe ratio (assuming daily returns, 252 trading days per year)
        if len(returns) > 0:
            sharpe_divisor = np.std(returns) * np.sqrt(252)
            if sharpe_divisor > 0:
                metrics.sharpe_ratio = np.mean(returns) * 252 / sharpe_divisor
        
        # Sortino ratio (only downside volatility)
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
            if downside_std > 0:
                metrics.sortino_ratio = np.mean(returns) * 252 / downside_std
        
        # Max drawdown
        if len(equity_array) > 0:
            cumulative_returns = equity_array / self.initial_capital
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown_array = (cumulative_returns - running_max) / running_max
            metrics.max_drawdown = np.min(drawdown_array)
            
            # Drawdown duration
            if len(drawdown_array) > 0:
                peak_idx = np.argmax(running_max)
                recovery_idx = np.where(running_max == running_max[peak_idx])[0]
                if len(recovery_idx) > peak_idx:
                    metrics.drawdown_duration = recovery_idx[-1] - peak_idx
        
        # Trade statistics
        if self.trade_pnl:
            metrics.total_trades = len(self.trade_pnl)
            pnls = [trade['pnl'] for trade in self.trade_pnl]
            
            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            metrics.winning_trades = len(winning_pnls)
            metrics.losing_trades = len(losing_pnls)
            
            if metrics.winning_trades > 0:
                metrics.avg_win = np.mean(winning_pnls)
            if metrics.losing_trades > 0:
                metrics.avg_loss = np.mean(losing_pnls)
            
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            if len(losing_pnls) > 0:
                metrics.profit_factor = sum(winning_pnls) / (-sum(losing_pnls)) if sum(losing_pnls) != 0 else 0
        
        # Calmar ratio
        if metrics.max_drawdown < 0:
            metrics.calmar_ratio = metrics.annual_return / (-metrics.max_drawdown)
        
        # Recovery factor
        total_pnl = sum([t['pnl'] for t in self.trade_pnl]) if self.trade_pnl else 0
        if metrics.max_drawdown < 0:
            max_loss = metrics.max_drawdown * self.initial_capital
            if max_loss != 0:
                metrics.recovery_factor = total_pnl / abs(max_loss)
        
        return metrics
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        return pd.Series(self.equity_history, index=pd.RangeIndex(len(self.equity_history)))
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log as pandas DataFrame."""
        if not self.trade_pnl:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_pnl)
    
    def reset(self):
        """Reset the engine for a new backtest."""
        self.current_capital = self.initial_capital
        self.available_capital = self.initial_capital
        self.orders.clear()
        self.next_order_id = 1
        self.pending_orders.clear()
        self.positions.clear()
        self.equity_history.clear()
        self.drawdown_history.clear()
        self.closed_trades.clear()
        self.trade_pnl.clear()
        self.current_timestamp = None
        self.current_prices.clear()
        if self.temporal_buffer:
            self.temporal_buffer.reset()
