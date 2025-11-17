# High-Performance Backtesting Engine - Quick Reference

## Installation & Setup

```python
# Required imports
import numpy as np
import pandas as pd
from backtest_engine import BacktestEngine, TransactionCostModel, OrderSide, OrderType
from strategy_framework import MovingAverageCrossover, TechnicalIndicators
```

## 30-Second Setup

```python
# 1. Create costs
costs = TransactionCostModel(commission_amount=0.001)

# 2. Create engine
engine = BacktestEngine(initial_capital=100000, 
                       transaction_cost_model=costs)

# 3. Run backtest
for timestamp, price in zip(dates, prices):
    if price > threshold:
        engine.submit_order('STOCK', OrderSide.BUY, 100, price)
    engine.step(timestamp, {'STOCK': price})

# 4. Get results
engine.close_all_positions({'STOCK': prices[-1]})
metrics = engine.calculate_metrics()
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

## Common Tasks

### Submit Orders
```python
# Market order (execute immediately)
engine.submit_order('STOCK', OrderSide.BUY, 100, order_type=OrderType.MARKET)

# Limit order (execute only at specified price)
engine.submit_order('STOCK', OrderSide.BUY, 100, price=50.0, 
                   order_type=OrderType.LIMIT)

# Stop order (activate at stop price)
engine.submit_order('STOCK', OrderSide.SELL, 100, stop_price=48.0,
                   order_type=OrderType.STOP)
```

### Configure Transaction Costs
```python
# For stocks
costs = TransactionCostModel(
    commission_type='fixed',
    commission_amount=1.0,      # $1 per trade
    slippage_type='percentage',
    slippage_amount=0.005,      # 0.5% slippage
    bid_ask_spread=0.0005       # 0.05% spread
)

# For high-frequency trading
costs = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.0001,   # 0.01% commission
    slippage_type='percentage',
    slippage_amount=0.001       # 0.1% slippage
)

# For futures
costs = TransactionCostModel(
    commission_type='fixed',
    commission_amount=2.5,      # $2.50 per contract
    slippage_type='fixed',
    slippage_amount=0.25        # 0.25 points
)
```

### Use Technical Indicators
```python
closes = df['close'].values

# Moving averages
sma_20 = TechnicalIndicators.sma(closes, 20)
ema_12 = TechnicalIndicators.ema(closes, 12)

# Momentum indicators
rsi_14 = TechnicalIndicators.rsi(closes, 14)
macd, signal, histogram = TechnicalIndicators.macd(closes)

# Volatility indicators
upper, middle, lower = TechnicalIndicators.bollinger_bands(closes, 20)
atr_14 = TechnicalIndicators.atr(df['high'].values, df['low'].values, closes, 14)

# Stochastic
k, d = TechnicalIndicators.stochastic(df['high'].values, df['low'].values, closes)

# Other indicators
momentum = TechnicalIndicators.momentum(closes, 10)
roc = TechnicalIndicators.rate_of_change(closes, 10)
```

### Create Custom Strategy
```python
from strategy_framework import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
        self.in_position = False
    
    def on_bar(self, bar, symbol):
        result = {'action': 'HOLD', 'price': bar['close']}
        
        if self.calculate_signal(bar) and not self.in_position:
            result['action'] = 'BUY'
            self.in_position = True
        elif self.exit_condition(bar) and self.in_position:
            result['action'] = 'SELL'
            self.in_position = False
        
        return result
    
    def calculate_signal(self, bar):
        # Your logic here
        pass
    
    def exit_condition(self, bar):
        # Your logic here
        pass
```

### Use Pre-built Strategies
```python
from strategy_framework import (
    MovingAverageCrossover,
    RSIMeanReversion,
    BollingerBandBreakout,
    MacdCrossover
)

# MA Crossover
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# RSI Mean Reversion
strategy = RSIMeanReversion(rsi_period=14)

# Bollinger Band Breakout
strategy = BollingerBandBreakout(period=20, std_dev=2.0)

# MACD Crossover
strategy = MacdCrossover()
```

### Analyze Results
```python
# Get metrics
metrics = engine.calculate_metrics()

# Print common metrics
print(f"Total Return: {metrics.total_return*100:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
print(f"Win Rate: {metrics.win_rate*100:.2f}%")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Total Trades: {metrics.total_trades}")

# Get equity curve
equity = engine.get_equity_curve()

# Get trade log
trades = engine.get_trade_log()
print(trades)

# Get signals (if using strategy framework)
# signals = executor.signals_history
# signals_df = pd.DataFrame(signals)
```

## Performance Metrics Reference

| Metric | Good | Excellent | Formula |
|--------|------|-----------|---------|
| Sharpe Ratio | > 1.0 | > 2.0 | (Return - Rf) / Volatility |
| Sortino Ratio | > 1.0 | > 2.0 | (Return - Rf) / Downside Volatility |
| Win Rate | > 50% | > 60% | Wins / Total Trades |
| Profit Factor | > 1.5 | > 2.0 | Gross Profit / Gross Loss |
| Calmar Ratio | > 0.5 | > 1.0 | Annual Return / Max DD |
| Max Drawdown | < -20% | < -10% | Lowest Return from Peak |

## Data Format

```python
# OHLCV DataFrame
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.DatetimeIndex(...))

# Use in backtest
for timestamp, row in df.iterrows():
    engine.current_timestamp = timestamp
    engine.current_prices['STOCK'] = row['close']
    engine.step(timestamp, {'STOCK': row['close']})
```

## Debugging Tips

```python
# Check available capital
print(f"Available: ${engine.available_capital:.2f}")

# Check open positions
for symbol, position in engine.positions.items():
    print(f"{symbol}: {position.quantity} @ ${position.entry_price:.2f}")

# Check pending orders
for order in engine.pending_orders:
    print(f"Order {order.order_id}: {order.quantity} {order.symbol} {order.side.value}")

# Check recent trades
trades = engine.get_trade_log()
print(trades.tail())

# Check equity curve
equity = engine.get_equity_curve()
print(f"Current Equity: ${equity.iloc[-1]:.2f}")
print(f"Peak Equity: ${equity.max():.2f}")
print(f"Lowest Equity: ${equity.min():.2f}")
```

## Optimization Tips

```python
# Use NumPy arrays for calculations
closes = df['close'].values  # NumPy array, not Series

# Pre-allocate arrays
results = np.zeros(len(df))
for i, row in enumerate(df.itertuples()):
    results[i] = calculate(row)

# Cache expensive calculations
if not hasattr(self, '_cached_sma'):
    self._cached_sma = TechnicalIndicators.sma(closes, 20)

# Use float32 for large arrays
prices = np.array(prices, dtype=np.float32)

# Aggregate high-frequency data
minute_data = tick_data.resample('1min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 
    'close': 'last', 'volume': 'sum'
})
```

## Common Errors & Fixes

```python
# Error: "Insufficient capital"
# Fix: Reduce quantity or check available_capital
if quantity * price > engine.available_capital:
    quantity = int(engine.available_capital / price)

# Error: "Order not filled"
# Fix: Check if price reached limit
if order_type == OrderType.LIMIT and current_price > limit_price:
    # Order won't fill

# Error: "No positions to close"
# Fix: Check if positions exist
if symbol in engine.positions:
    engine.close_all_positions({symbol: price})

# Error: "NaN values in indicators"
# Fix: Need enough data for indicator period
if len(prices) < indicator_period:
    # Not enough data
```

## File Reference

| File | Purpose |
|------|---------|
| `backtest_engine.py` | Core backtesting engine |
| `strategy_framework.py` | Strategies and indicators |
| `backtest_examples.py` | Complete working examples |
| `BACKTEST_DOCUMENTATION.md` | Detailed technical docs |
| `BACKTEST_README.md` | User-friendly guide |
| `OPTIMIZATION_GUIDE.md` | Performance optimization |
| `IMPLEMENTATION_SUMMARY.md` | Project overview |

## Useful Links in Documentation

- **Data format requirements**: See BACKTEST_DOCUMENTATION.md → "Data Format Requirements"
- **Order types explained**: See BACKTEST_DOCUMENTATION.md → "Order Types"
- **Look-ahead bias**: See BACKTEST_DOCUMENTATION.md → "Look-Ahead Bias Prevention"
- **Performance optimization**: See OPTIMIZATION_GUIDE.md
- **Complete examples**: See backtest_examples.py

## Advanced Features

### Walk-Forward Analysis
```python
for train_end, test_end in windows:
    # Train
    train_data = data[:train_end]
    strategy.optimize(train_data)
    
    # Test
    test_data = data[train_end:test_end]
    results = run_backtest(strategy, test_data)
```

### Parameter Optimization
```python
results = {}
for fast in range(10, 30):
    for slow in range(40, 60):
        engine.reset()
        # Run backtest with (fast, slow)
        metrics = run_backtest(fast, slow)
        results[(fast, slow)] = metrics

best_params = max(results, key=lambda k: results[k].sharpe_ratio)
```

### Multi-Processing
```python
from multiprocessing import Pool

def test_params(params):
    fast, slow = params
    return run_backtest(fast, slow)

param_grid = [(f, s) for f in range(10, 30) for s in range(40, 60)]

with Pool(processes=4) as pool:
    results = pool.map(test_params, param_grid)
```

## Performance Targets

- **Daily data**: Process 1M+ bars/second
- **Minute data**: Process 100K+ bars/second
- **Tick data**: Process 1K+ bars/second (10K+ aggregated)

## When to Use What

| Scenario | Use |
|----------|-----|
| Quick backtest | `BacktestEngine` directly |
| Strategy testing | `StrategyFramework` + custom strategy |
| Parameter search | Multiple backtests in loop |
| Walk-forward test | Custom loop with window sliding |
| Portfolio | Multiple symbols in single backtest |
| Real-time | Event-driven mode with live data |

---

**For more details, see the full documentation files.**
