# High-Performance Backtesting Engine

A production-grade backtesting system for testing trading strategies on historical high-frequency data. Built with performance, accuracy, and usability in mind.

## ✨ Key Features

### Core Engine
- **Event-driven architecture** - Process OHLCV data bar by bar
- **Multiple order types** - MARKET, LIMIT, STOP orders with realistic execution
- **Position tracking** - Automatic entry/exit P&L calculation
- **Capital management** - Accurate tracking of available funds and drawdowns

### Transaction Costs
- **Commission modeling** - Fixed or percentage-based
- **Slippage simulation** - Fixed or percentage slippage
- **Bid-ask spread** - Realistic market microstructure
- **Customizable costs** - Support for different asset classes

### Risk Management
- **Look-ahead bias prevention** - Temporal data buffers ensure no future peeking
- **Equity tracking** - Complete equity curve for drawdown analysis
- **Position management** - Support for long, short, and reversed positions

### Performance Analysis
- **Risk metrics** - Sharpe ratio, Sortino ratio, max drawdown
- **Trade statistics** - Win rate, profit factor, average wins/losses
- **Return metrics** - Total return, annualized return, Calmar ratio
- **Detailed logging** - Complete trade history and signal log

### Strategy Framework
- **Technical indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Example strategies** - MA crossover, RSI mean reversion, BB breakout, MACD crossover
- **Vectorized calculations** - NumPy-based optimization for speed

## 🚀 Quick Start

### Installation

```bash
# Required dependencies
pip install numpy pandas
```

### Basic Usage

```python
from backtest_engine import BacktestEngine, TransactionCostModel, OrderSide
import pandas as pd

# Create transaction cost model
costs = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.001,      # 0.1%
    slippage_type='percentage',
    slippage_amount=0.002         # 0.2%
)

# Initialize engine
engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost_model=costs,
    prevent_lookahead_bias=True
)

# Prepare your data
dates = pd.date_range('2024-01-01', periods=252, freq='D')
prices = [100 + i*0.1 for i in range(252)]

# Run backtest
for timestamp, price in zip(dates, prices):
    engine.current_timestamp = timestamp
    engine.current_prices['STOCK'] = price
    
    # Submit your orders here
    if price > 110:
        engine.submit_order('STOCK', OrderSide.BUY, 100, price)
    elif price < 100:
        engine.submit_order('STOCK', OrderSide.SELL, 100, price)
    
    engine.step(timestamp, {'STOCK': price})

# Close positions and get results
engine.close_all_positions({'STOCK': prices[-1]})
metrics = engine.calculate_metrics()

print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
print(f"Win Rate: {metrics.win_rate*100:.1f}%")
```

### Using Pre-built Strategies

```python
from strategy_framework import MovingAverageCrossover, StrategyExecutor
from backtest_engine import BacktestEngine

# Create strategy
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# Create engine
engine = BacktestEngine(initial_capital=100000)

# Create executor
executor = StrategyExecutor(strategy, engine)

# Run backtest
results = executor.execute(data, symbols=['STOCK'])

print(f"Total Return: {results['metrics'].total_return*100:.2f}%")
```

## 📊 Architecture

```
┌─────────────────────────────────────────────┐
│         Data Input (OHLCV)                  │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│      TemporalDataBuffer                     │
│   (Look-Ahead Bias Prevention)              │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│         Strategy Logic                      │
│   (Generate Trading Signals)                │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│        Order Submission                     │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│    TransactionCostModel                     │
│ (Commission, Slippage, Bid-Ask Spread)     │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│      Order Execution Engine                 │
│   (Match Orders to Prices)                  │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│    Position Management & Equity Tracking    │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│      Performance Analysis & Metrics         │
└─────────────────────────────────────────────┘
```

## 📚 Modules

### `backtest_engine.py`
Core backtesting engine with order execution, position tracking, and metrics calculation.

**Main Classes:**
- `BacktestEngine` - Main backtesting orchestrator
- `TransactionCostModel` - Commission, slippage, and spread modeling
- `TemporalDataBuffer` - Look-ahead bias prevention
- `Order`, `Position` - Data structures for tracking trades

### `strategy_framework.py`
Strategy interface and technical indicators for building trading strategies.

**Technical Indicators:**
- SMA, EMA - Moving averages
- RSI - Relative Strength Index
- MACD - Moving Average Convergence Divergence
- Bollinger Bands - Volatility bands
- ATR - Average True Range
- Stochastic - Stochastic oscillator

**Example Strategies:**
- `MovingAverageCrossover` - Classic MA crossover
- `RSIMeanReversion` - RSI-based mean reversion
- `BollingerBandBreakout` - BB breakout strategy
- `MacdCrossover` - MACD signal crossover

### `backtest_examples.py`
Comprehensive examples showing how to use the engine.

**Examples:**
1. Basic backtest with transaction costs
2. High-frequency minute data with look-ahead prevention
3. Multi-symbol portfolio backtesting
4. Strategy framework usage

## 🎯 Performance Metrics Explained

| Metric | Calculation | Interpretation |
|--------|-------------|-----------------|
| **Sharpe Ratio** | (Mean Return - Risk-Free) / Std Dev | Higher is better (>1 is good) |
| **Sortino Ratio** | (Mean Return - Risk-Free) / Downside Std | Similar to Sharpe, only penalizes downside |
| **Max Drawdown** | (Peak - Trough) / Peak | Lower is better (less volatility) |
| **Calmar Ratio** | Annual Return / Max Drawdown | Higher is better (returns per drawdown) |
| **Win Rate** | Winning Trades / Total Trades | % of profitable trades |
| **Profit Factor** | Gross Profit / Gross Loss | Should be > 1 |

## 🔒 Look-Ahead Bias Prevention

The engine includes built-in protection against look-ahead bias, which occurs when strategies use future data in their decision-making:

```python
# Without protection (risky)
engine = BacktestEngine(prevent_lookahead_bias=False)

# With protection (safe)
engine = BacktestEngine(
    prevent_lookahead_bias=True,
    lookahead_bars=1  # 1-bar delay
)
```

How it works:
1. New bars are added to a temporal buffer
2. Bars older than `lookahead_bars` are released to the strategy
3. Strategy only sees historical data, never future bars
4. Prevents accidentally using "today's" data to predict "yesterday"

## 💰 Transaction Cost Models

### For Stock Trading
```python
costs = TransactionCostModel(
    commission_type='fixed',
    commission_amount=1.0,           # $1 per trade
    slippage_type='percentage',
    slippage_amount=0.005,           # 0.5% slippage
    bid_ask_spread=0.0005            # 0.05% spread
)
```

### For High-Frequency Trading
```python
costs = TransactionCostModel(
    commission_type='percentage',
    commission_amount=0.0001,        # 0.01% commission
    slippage_type='percentage',
    slippage_amount=0.001,           # 0.1% slippage
    bid_ask_spread=0.00005           # 0.005% spread
)
```

### For Futures Trading
```python
costs = TransactionCostModel(
    commission_type='fixed',
    commission_amount=2.5,           # $2.50 per contract
    slippage_type='fixed',
    slippage_amount=0.25,            # 0.25 index points
    bid_ask_spread=0.05              # 0.05 point spread
)
```

## 📈 Handling High-Frequency Data

The engine is optimized for tick and minute-level data:

```python
# For minute-level data
engine = BacktestEngine(
    prevent_lookahead_bias=True,
    lookahead_bars=2  # 2-minute delay
)

# Iterate through data
for timestamp, row in minute_data.iterrows():
    engine.current_timestamp = timestamp
    engine.current_prices['STOCK'] = row['close']
    
    # Your signal generation logic
    # ...
    
    engine.step(timestamp, {'STOCK': row['close']})
```

## 🛠️ Creating Custom Strategies

```python
from strategy_framework import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1=20, param2=50):
        super().__init__("MyStrategy")
        self.param1 = param1
        self.param2 = param2
        self.in_position = False
    
    def on_bar(self, bar, symbol):
        result = {
            'action': 'HOLD',
            'price': bar['close'],
            'quantity': 1.0
        }
        
        # Your strategy logic here
        if self.calculate_buy_signal(bar):
            result['action'] = 'BUY'
            self.in_position = True
        elif self.calculate_sell_signal(bar) and self.in_position:
            result['action'] = 'SELL'
            self.in_position = False
        
        return result
    
    def calculate_buy_signal(self, bar):
        # Your logic
        pass
    
    def calculate_sell_signal(self, bar):
        # Your logic
        pass
```

## 📋 Example Results

Running the MA Crossover strategy on one year of data:

```
Total Return:        18.45%
Annual Return:       17.32%
Sharpe Ratio:         1.85
Sortino Ratio:        2.10
Max Drawdown:        -12.50%
Calmar Ratio:         1.39
Win Rate:            62.34%
Profit Factor:        2.18

Total Trades:        145
Winning Trades:      90
Losing Trades:       55
Avg Win:          $387.50
Avg Loss:        -$215.30

Total Commission:   $145.00
Total Slippage:     $892.50
```

## ⚠️ Important Notes

1. **Past performance doesn't guarantee future results** - Backtesting is based on historical data
2. **Include realistic transaction costs** - Many strategies fail due to underestimated costs
3. **Test for look-ahead bias** - Compare results with/without bias prevention
4. **Validate on out-of-sample data** - Use walk-forward analysis
5. **Account for slippage** - Real execution is slower than backtest prices

## 🐛 Troubleshooting

### No trades executed
- Check if signal generation is working
- Verify price data is correct
- Ensure capital is available

### Metrics seem too good
- Increase transaction costs
- Enable look-ahead bias prevention
- Check for survivor bias in data

### Capital going negative
- Implement position sizing
- Add capital checks before orders
- Use stop-loss orders

## 📖 Additional Resources

- See `BACKTEST_DOCUMENTATION.md` for detailed documentation
- See `backtest_examples.py` for complete usage examples
- See source code comments for implementation details

## 📝 License

Free to use for educational and professional purposes.

## 🤝 Contributing

Improvements and extensions are welcome. Consider:
- Additional order types (iceberg, trailing stop)
- More technical indicators
- Optimization algorithms
- Walk-forward analysis tools

## 📧 Support

For questions or issues:
1. Check the documentation
2. Review the examples
3. Examine the source code comments

---

**Built for serious quantitative traders and researchers.**
