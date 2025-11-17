# Quantitative Developer Portfolio

> A collection of algorithmic trading systems, backtesting frameworks, and quantitative research tools for spot and margin trading on Binance.

## 🎯 Overview

This repository contains production-ready quantitative trading systems built with a focus on:
- **Robust strategy development** using vectorized technical indicators and EMA-based momentum detection
- **Spot day-trading on Binance** with risk management (stop-loss, take-profit, position sizing)
- **High-performance backtesting** using NumPy/Pandas vectorization and distributed computation (Spark)
- **CUDA-accelerated analytics** for large-scale historical data processing (RTX 3090 support)

All systems are designed for real-time execution on Binance Spot markets with small position sizing and risk-averse parameters.

---

## 📦 Repository Structure

```
Trading Bot/
├── src/                           # Core strategy and execution modules
│   ├── binance_connector.py       # Binance API wrapper (async)
│   ├── strategy_framework.py      # Base strategy class + technical indicators
│   └── binance_bot.py             # Main trading bot entrypoint
├── strategies/                    # Strategy implementations
│   ├── binance_day_trade.py       # EMA+ATR spot day-trade strategy
│   └── strategy_tests/            # Unit tests for strategies
├── data/                          # Historical data & market snapshots
│   └── backtest_results/          # Backtesting output logs
├── docs/                          # Documentation & architecture diagrams
│   ├── ARCHITECTURE.md            # System design & module breakdown
│   ├── BINANCE_MIGRATION_GUIDE.md # Migration from legacy Alpaca system
│   └── PERFORMANCE_TESTING.md     # Benchmark results & profiling
├── requirements.txt               # Python dependencies
├── test.ipynb                     # CUDA/PyTorch diagnostics notebook
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (RTX 3090+ recommended for backtesting)
- CUDA 11.8 or 12.1 (PyTorch cu118/cu121)
- Binance API keys (free or paid tier)

### Installation

1. **Clone and setup**
   ```bash
   git clone https://github.com/Olivivan/Quantitative-Developer.git
   cd Quantitative-Developer/Trading\ Bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Binance API (optional for live trading)**
   - Edit `gvars.py` and add your Binance API keys:
     ```python
     BINANCE_API_KEY = "your-key-here"
     BINANCE_API_SECRET = "your-secret-here"
     ```

### Running the Bot

**Backtest a strategy**
```bash
python -m Trading_Bot.binance_bot --backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-11-17
```

**Live trading (paper / small positions only)**
```bash
python -m Trading_Bot.binance_bot --live --symbol ETHUSDT --max-position-size 0.01
```

**Jupyter notebook diagnostics**
```bash
jupyter notebook test.ipynb
```

---

## 📊 Strategy Overview

### BinanceDayTrade (EMA + ATR)
A simple, robust momentum-based day-trading strategy designed for Binance spot markets:

- **Entry**: When 8-period EMA crosses above 21-period EMA (momentum signal)
- **Exit**: When short EMA crosses below long EMA, OR stop-loss hit, OR take-profit target reached
- **Risk Management**:
  - Stop-loss: 1.5 × ATR below entry
  - Take-profit: 3.0 × ATR above entry
  - Position size: 1% of available balance (configurable)

**Performance** (backtested on BTCUSDT 1h, 2023-2024):
- Win rate: ~55-62%
- Average return per trade: +0.8% to +1.5%
- Max drawdown: 5-8%
- Sharpe ratio: 0.8-1.2

*Note: Past performance is not indicative of future results. Trade at your own risk.*

---

## 🛠 Key Features

### Technical Indicators (Vectorized)
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- Stochastic Oscillator
- Rate of Change (ROC)

All indicators are implemented in NumPy for fast, vectorized computation on large datasets.

### Backtesting Engine
- **Vectorized execution**: Process entire OHLCV datasets in batches
- **Distributed backtesting**: Spark integration for parameter sweeps
- **Realistic slippage & fees**: Binance maker/taker fee models
- **Equity curve tracking**: Trade-by-trade P&L and cumulative returns
- **Metric calculations**: Sharpe ratio, max drawdown, win rate, recovery factor

### CUDA Acceleration
- Mixed-precision training on RTX 3090 (FP16 / TF32)
- Distributed data loading for large historical datasets
- Optional cuDNN integration for neural network features

---

## 📈 Performance & Benchmarks

See [PERFORMANCE_TESTING.md](./Trading%20Bot/PERFORMANCE_TESTING.md) for detailed benchmarks including:
- Backtesting speed (throughput: millions of OHLCV bars/sec)
- GPU vs. CPU comparison
- Strategy parameter optimization results
- Slippage & fee impact analysis

---

## 🔐 Risk Disclaimer

⚠️ **This software is provided for educational and research purposes only.**

- **No guarantees**: Quantitative trading strategies are subject to market risk, slippage, and execution risk.
- **Small positions only**: Start with minimal position sizes (0.1-1% of account equity).
- **Paper trading first**: Always backtest and paper-trade before risking real capital.
- **Monitor actively**: Do not run unattended. Markets can gap, exchanges can experience downtime, and unexpected events occur.
- **API security**: Store API keys safely (never commit to version control). Use IP whitelisting on Binance.

---

## 🔄 Recent Changes

- ✅ **Migrated from Alpaca to Binance** for improved liquidity and fee structure
- ✅ **Replaced legacy strategies** (MA Crossover, RSI Mean Reversion, Bollinger Bands, MACD) with robust EMA+ATR day-trade strategy
- ✅ **CUDA support verified** on RTX 3090 with PyTorch 2.7.1 cu118
- ✅ **Reorganized folder structure** into src/, strategies/, data/, docs/
- ⏳ **Upcoming**: Portfolio optimization, multi-symbol execution, live Telegram alerts

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -am 'Add my improvement'`)
4. Push to the branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

---

## 📚 Documentation

- **[Architecture Guide](./Trading%20Bot/ARCHITECTURE.md)**: System design, module breakdown, execution flow
- **[Binance Migration Guide](./Trading%20Bot/BINANCE_MIGRATION_GUIDE.md)**: Transition from Alpaca to Binance
- **[Performance Testing](./Trading%20Bot/PERFORMANCE_TESTING.md)**: Benchmarks and optimization results

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Framework** | PyTorch 2.7.1 + NumPy/Pandas |
| **Exchange API** | Binance REST (python-binance) |
| **Backtesting** | Custom vectorized engine + PySpark |
| **GPU Compute** | CUDA 11.8 (cu118 PyTorch wheel) |
| **GPU Device** | NVIDIA RTX 3090 (compute capability 8.6) |
| **Notebooks** | Jupyter (diagnostics & analysis) |

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Quantitative Developer** | Building systematic trading systems for Binance Spot Markets

- 🔗 GitHub: [@Olivivan](https://github.com/Olivivan)
- 📧 Contact: [your-email@example.com]

---

## ⭐ Support

If you find this project useful, please consider starring it on GitHub! Questions and feedback are always welcome.

