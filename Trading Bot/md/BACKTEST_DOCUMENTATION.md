# High-Performance Backtesting Engine Documentation

## Overview

This is a production-grade backtesting engine built with performance and accuracy as primary goals. It's designed to simulate trading strategies using historical high-frequency (tick or minute-level) data with proper handling of:

- **Look-ahead bias prevention** - Temporal data buffers ensure strategies only see past data
- **Transaction costs** - Commission, slippage, and bid-ask spread modeling
- **Position management** - Tracking of entries, exits, and P&L calculations
- **Performance metrics** - Sharpe ratio, Sortino ratio, max drawdown, win rate, and more

## Architecture

### Core Components

#### 1. **BacktestEngine** (`backtest_engine.py`)
The main backtesting engine that orchestrates order execution, position tracking, and equity management.

**Key Features:**
- Order submission and execution with different order types (MARKET, LIMIT, STOP)
- Position tracking with unrealized/realized P&L
- Capital management and available funds tracking
- Equity curve tracking for metrics calculation
- Trade history logging

**Usage:**
```python
from backtest_engine import BacktestEngine, TransactionCostModel, OrderSide, OrderType

# Create engine with realistic transaction costs
costs = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.001,    # 0.1% per trade
    slippage_type='percentage',
    slippage_amount=0.002,      # 0.2% slippage
    bid_ask_spread=0.0001       # 0.01% spread
)

engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost_model=costs,
    prevent_lookahead_bias=True,
    lookahead_bars=1  # 1-bar delay
)

# Submit orders
order_id = engine.submit_order(
    symbol='STOCK',
    side=OrderSide.BUY,
    quantity=100,
    price=50.0,
    order_type=OrderType.MARKET
)

# Step through time
engine.step(timestamp, {'STOCK': current_price})

# Get results
metrics = engine.calculate_metrics()
trades = engine.get_trade_log()
equity_curve = engine.get_equity_curve()
```

#### 2. **TransactionCostModel** (`backtest_engine.py`)
Realistic modeling of trading costs including commissions, slippage, and bid-ask spreads.

**Commission Types:**
- `'fixed'` - Fixed dollar amount per trade
- `'percentage'` - Percentage of trade value

**Slippage Types:**
- `'fixed'` - Fixed ticks/points
- `'percentage'` - Percentage of price

**Example:**
```python
# Stocks with fixed commission
costs_stocks = TransactionCostModel(
    commission_type='fixed',
    commission_amount=1.0,          # $1 per trade
    slippage_type='percentage',
    slippage_amount=0.005,          # 0.5% slippage
    bid_ask_spread=0.0005           # 0.05% spread
)

# High-frequency trading with percentage commission
costs_hft = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.0001,       # 0.01% commission
    slippage_type='percentage',
    slippage_amount=0.001,          # 0.1% slippage
    bid_ask_spread=0.00005          # 0.005% spread
)

# Futures with fixed costs
costs_futures = TransactionCostModel(
    commission_type='fixed',
    commission_amount=2.5,          # $2.50 per contract
    slippage_type='fixed',
    slippage_amount=0.25,           # 0.25 index points
    bid_ask_spread=0.05             # 0.05 point spread
)
```

#### 3. **TemporalDataBuffer** (`backtest_engine.py`)
Prevents look-ahead bias by delaying strategy signals by a configurable number of bars.

**How It Works:**
- Buffers incoming OHLCV bars
- Returns "safe" bars delayed by N bars to the strategy
- Ensures strategies cannot peek into future data

**Example:**
```python
buffer = TemporalDataBuffer(bar_count=2)

# These would be called in your backtest loop
safe_bar_1 = buffer.add_bar(current_bar)  # Returns None (buffering)
safe_bar_2 = buffer.add_bar(current_bar)  # Returns None (buffering)
safe_bar_3 = buffer.add_bar(current_bar)  # Returns bar from 2 steps ago
```

#### 4. **Strategy Framework** (`strategy_framework.py`)
High-performance vectorized strategy interface with built-in technical indicators.

**Available Strategies:**
1. **MovingAverageCrossover** - Classic MA crossover signal
2. **RSIMeanReversion** - RSI oversold/overbought trading
3. **BollingerBandBreakout** - Bollinger Bands breakout strategy
4. **MacdCrossover** - MACD signal line crossover

**Available Indicators:**
- SMA, EMA (Simple/Exponential Moving Averages)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator
- Momentum, Rate of Change

**Creating Custom Strategies:**
```python
from strategy_framework import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, fast_period=20, slow_period=50):
        super().__init__("MyStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.in_position = False
    
    def on_bar(self, bar, symbol):
        """Process each bar and return trading signal."""
        result = {
            'action': 'HOLD',           # 'BUY', 'SELL', 'HOLD', 'CLOSE'
            'price': bar['close'],      # Execution price
            'quantity': 1.0             # Order quantity
        }
        
        # Your strategy logic here
        if self.should_buy():
            result['action'] = 'BUY'
        elif self.should_sell():
            result['action'] = 'SELL'
        
        return result
    
    def should_buy(self):
        # Your logic here
        pass
    
    def should_sell(self):
        # Your logic here
        pass
```

## Performance Metrics

The engine calculates comprehensive performance metrics:

### Return Metrics
- **Total Return** - Total percentage return from initial capital
- **Annual Return** - Annualized return (assuming 252 trading days)
- **Calmar Ratio** - Annual return / max drawdown (recovery efficiency)
- **Recovery Factor** - Total P&L / max drawdown (ability to recover losses)

### Risk-Adjusted Metrics
- **Sharpe Ratio** - Return per unit of risk (assumes normal distribution)
- **Sortino Ratio** - Return per unit of downside risk (only penalizes downside volatility)
- **Max Drawdown** - Largest peak-to-trough decline
- **Drawdown Duration** - Number of bars to recover from max drawdown

### Trade Statistics
- **Total Trades** - Number of completed round-trip trades
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Sum of wins / sum of losses
- **Average Win/Loss** - Mean profit/loss per trade
- **Winning/Losing Trades** - Counts of each

### Transaction Costs
- **Total Commission** - Sum of all commissions paid
- **Total Slippage** - Sum of all slippage costs

## Order Types

### Market Orders
Execute immediately at current market price with slippage applied.

```python
engine.submit_order(
    symbol='STOCK',
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.MARKET
)
```

### Limit Orders
Execute only if price reaches the limit level.

```python
engine.submit_order(
    symbol='STOCK',
    side=OrderSide.BUY,
    quantity=100,
    price=50.00,  # Execute only at $50 or better
    order_type=OrderType.LIMIT
)
```

### Stop Orders
Activate when price reaches the stop level, then execute as market.

```python
engine.submit_order(
    symbol='STOCK',
    side=OrderSide.SELL,
    quantity=100,
    stop_price=48.00,  # Activate at $48
    order_type=OrderType.STOP
)
```

## High-Frequency Data Handling

The engine is optimized for tick or minute-level data:

### Best Practices

1. **Use appropriate look-ahead delay:**
```python
# For minute data
engine = BacktestEngine(
    prevent_lookahead_bias=True,
    lookahead_bars=2  # 2-minute delay
)

# For tick data
engine = BacktestEngine(
    prevent_lookahead_bias=True,
    lookahead_bars=5  # 5-tick delay
)
```

2. **Adjust transaction costs for frequency:**
```python
# Minute-level trading (lower costs)
costs_minute = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.0001,  # 0.01%
    slippage_type='percentage',
    slippage_amount=0.0005     # 0.05%
)

# Daily trading (higher costs)
costs_daily = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.001,   # 0.1%
    slippage_type='percentage',
    slippage_amount=0.002      # 0.2%
)
```

3. **Handle data points efficiently:**
```python
# For large datasets, use numpy arrays internally
import numpy as np

# Vectorized operations on price arrays
closes = df['close'].values  # Convert to numpy for speed
returns = np.diff(closes) / closes[:-1]
```

## Look-Ahead Bias Prevention

Look-ahead bias occurs when strategies use future data to make present decisions. This engine prevents it with automatic temporal buffering:

```python
# Without bias prevention (risky)
engine = BacktestEngine(prevent_lookahead_bias=False)

# With bias prevention (safe)
engine = BacktestEngine(
    prevent_lookahead_bias=True,
    lookahead_bars=1  # Data from 1 bar ago only
)
```

### How It Works

The temporal buffer stores incoming bars and only releases "safe" bars to the strategy:

```
Bar 1: Buffered (reserved for next iteration)
Bar 2: Buffered (reserved for next iteration)
Bar 3: Strategy receives Bar 1, Buffers Bar 3
Bar 4: Strategy receives Bar 2, Buffers Bar 4
...
```

This ensures the strategy operates on historical data only.

## Complete Workflow Example

```python
import pandas as pd
from backtest_engine import BacktestEngine, TransactionCostModel, OrderSide
from strategy_framework import MovingAverageCrossover

# Step 1: Prepare your data
dates = pd.date_range('2024-01-01', periods=252, freq='D')
prices = [100] + [100 * (1 + 0.0005) ** i for i in range(1, 252)]

# Step 2: Configure costs
costs = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.001,
    slippage_type='percentage',
    slippage_amount=0.002
)

# Step 3: Initialize engine
engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost_model=costs,
    prevent_lookahead_bias=True
)

# Step 4: Run backtest
for timestamp, price in zip(dates, prices):
    engine.current_timestamp = timestamp
    engine.current_prices['STOCK'] = price
    
    # Your signal generation logic here
    if price > 102:
        engine.submit_order('STOCK', OrderSide.BUY, 100, price)
    elif price < 98:
        engine.submit_order('STOCK', OrderSide.SELL, 100, price)
    
    engine.step(timestamp, {'STOCK': price})

# Step 5: Analyze results
metrics = engine.calculate_metrics()
trades = engine.get_trade_log()
equity = engine.get_equity_curve()

print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
print(f"Win Rate: {metrics.win_rate*100:.2f}%")
print(f"\nTrades:\n{trades}")
```

## Performance Optimization Tips

1. **Use NumPy for calculations:**
   - Vectorized operations are 10-100x faster than Python loops

2. **Pre-allocate arrays:**
   - Avoid resizing arrays during calculations

3. **Cache calculations:**
   - Store indicator values instead of recalculating

4. **Batch process data:**
   - Process multiple bars together when possible

5. **Use appropriate data types:**
   - Use float32 for large arrays to save memory

## Common Issues and Solutions

### Issue: Negative cash after buying
**Solution:** Check capital availability before submitting orders:
```python
required = price * quantity
if required > engine.available_capital:
    # Reduce quantity or skip
    pass
```

### Issue: Positions not closing
**Solution:** Ensure you call `close_all_positions()` at end of backtest:
```python
engine.close_all_positions({'STOCK': final_price})
```

### Issue: Unrealistic returns
**Solution:** Increase transaction costs:
```python
# Too low - might be unrealistic
costs = TransactionCostModel(commission_amount=0.00001)

# More realistic for daily trading
costs = TransactionCostModel(commission_amount=0.001)
```

### Issue: Look-ahead bias suspected
**Solution:** Increase look-ahead delay:
```python
# Current
engine = BacktestEngine(lookahead_bars=1)

# More conservative
engine = BacktestEngine(lookahead_bars=3)
```

## Data Format Requirements

OHLCV data should be a pandas DataFrame with columns:
- `open` - Opening price
- `high` - Highest price in period
- `low` - Lowest price in period
- `close` - Closing price
- `volume` - Trading volume

Index should be timestamps (DatetimeIndex).

Example:
```
                      open    high     low   close   volume
2024-01-01 00:00:00   100.0  101.5   99.0  100.5  1000000
2024-01-01 01:00:00   100.5  101.2   99.8  101.0  1500000
...
```

## Advanced Features

### Custom Order Types
Extend the `Order` class for specialized order handling:
```python
class TrailingStopOrder(Order):
    def __init__(self, *args, trailing_amount=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.trailing_amount = trailing_amount
        self.highest_price = self.price
```

### Strategy Parameter Optimization
```python
results = {}
for fast_period in [10, 20, 30]:
    for slow_period in [40, 50, 60]:
        engine.reset()
        # Run backtest...
        results[(fast_period, slow_period)] = metrics
```

### Walk-Forward Analysis
Segment data into training and testing periods:
```python
for train_start, test_start, test_end in walk_forward_windows:
    # Optimize parameters on train_start:test_start
    # Evaluate on test_start:test_end
```

## References

- **Sharpe Ratio**: Risk-adjusted return metric (assumes normal distribution)
- **Sortino Ratio**: Similar to Sharpe but only penalizes downside volatility
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Drawdown**: Peak-to-trough decline in cumulative returns
- **Win Rate**: Percentage of profitable trades

## License

This backtesting engine is designed for educational and professional use. Please ensure compliance with all applicable trading regulations and disclaimer requirements.

## Support

For issues or questions:
1. Check this documentation
2. Review the examples in `backtest_examples.py`
3. Examine the source code comments in `backtest_engine.py`
