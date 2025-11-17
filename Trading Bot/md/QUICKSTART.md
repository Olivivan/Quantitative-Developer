# ⚡ Quick Start Guide - Binance Trading Bot

## 5-Minute Setup

### Step 1: Install Python packages (2 min)
```bash
cd "d:\Quantitative-Developer\Trading Bot"
pip install -r requirements.txt
```

### Step 2: Get Binance API Keys (2 min)
**For Paper Trading (Recommended First):**
1. Go to https://testnet.binance.vision
2. Click "Generate HMAC SHA256 Key"
3. Copy API Key and Secret Key

**For Live Trading (After testing):**
1. Go to https://www.binance.com/account/api-management
2. Create new key
3. Enable trading, disable withdrawals
4. Whitelist your IP

### Step 3: Configure Bot (1 min)
**Option A: Environment Variables**
```powershell
# Windows PowerShell
$env:BINANCE_API_KEY = "paste_your_api_key_here"
$env:BINANCE_API_SECRET = "paste_your_api_secret_here"
$env:BINANCE_TESTNET = "true"
```

**Option B: Config File**
```bash
# Copy and edit the example config
copy config.yaml.example config.yaml
# Edit config.yaml with your API keys
```

### Step 4: Create Asset List (1 min)
Create or update `_raw_assets.csv`:
```csv
symbol
BTCUSDT
ETHUSDT
BNBUSDT
ADAUSDT
XRPUSDT
DOGEUSDT
```

### Step 5: Run the Bot! (< 1 min)
```bash
python binance_bot.py
```

✅ **Done!** Bot is running and logs are in `./logs/`

---

## 📊 What to Expect

### First Run Output
```
2025-01-17 14:23:45 - INFO - Bot initialized for environment: development
2025-01-17 14:23:46 - INFO - Connected to Binance API
2025-01-17 14:23:47 - INFO - Worker 0 started
2025-01-17 14:23:47 - INFO - Worker 1 started
...
2025-01-17 14:23:48 - INFO - Chosen asset: BTCUSDT
2025-01-17 14:23:50 - INFO - BTCUSDT trend: UP
2025-01-17 14:23:52 - INFO - Position entered: BUY 0.001 BTCUSDT at $42150.00
```

### Logs Location
- **Main logs**: `./logs/bot_*.log`
- **Strategy logs**: `./logs/bot_*.log`
- **Per-minute**: Each minute in separate file

## 🎯 Common Tasks

### Check Bot Status
```bash
# Watch logs in real-time
Get-Content ./logs/*.log -Wait
```

### Stop the Bot
```bash
# Ctrl+C in terminal (graceful shutdown)
# Closes open positions and saves logs
```

### Switch to Live Trading
```bash
# Edit config.yaml
api:
  testnet: false  # Change to false

# Or use environment variable
$env:BINANCE_TESTNET = "false"
```

### Adjust Trading Parameters
Edit `config.yaml`:
```yaml
trading:
  oper_equity: 1000           # Amount per trade
  stop_loss_margin: 0.05      # 5% stop loss
  take_profit_ratio: 1.5      # 1.5:1 risk:reward
  max_workers: 10             # Concurrent trades
```

### Enable GPU Acceleration (Optional)
```yaml
performance:
  use_pytorch: true
  use_gpu: true  # Requires NVIDIA GPU + CUDA
```

## 🧪 Testing the Setup

### Test 1: API Connection
```python
# Save as test_connection.py
import asyncio
from binance_connector import BinanceConnector
import os

async def test():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    async with BinanceConnector(api_key, api_secret, testnet=True) as conn:
        price = await conn.get_price("BTCUSDT")
        print(f"BTC Price: ${price}")

asyncio.run(test())
```

Run:
```bash
python test_connection.py
# Output: BTC Price: $42150.25
```

### Test 2: Indicators
```python
# Save as test_indicators.py
from pytorch_indicators import TechnicalIndicators
import numpy as np

# Create random price data
prices = np.random.randn(100).cumsum() + 100

# Calculate indicators
sma = TechnicalIndicators.sma(prices, 20)
rsi = TechnicalIndicators.rsi(prices, 14)

print(f"SMA: {sma[-1]:.2f}")
print(f"RSI: {rsi[-1]:.2f}")
```

Run:
```bash
python test_indicators.py
# Output:
# SMA: 100.15
# RSI: 55.42
```

## 📈 Monitoring Performance

### Check Daily Stats (in log file)
```
========================================
TRADING STATISTICS
========================================
total_trades: 12
wins: 8
losses: 4
win_rate: 66.67%
total_pnl: $1234.56
========================================
```

### Key Metrics to Watch
```
✓ Win Rate: Should be > 50%
✓ P&L: Should be positive
✓ Max Drawdown: Should be < 10%
✓ API Latency: Should be < 20ms
✓ Cache Hit Rate: Should be > 80%
✓ Errors: Should be 0 or rare
```

## ⚠️ Important Warnings

### ❌ DO NOT:
- Start with live trading immediately
- Use large position sizes initially
- Ignore stop-loss orders
- Trade without monitoring logs
- Share your API secret key
- Leave testnet: false by mistake

### ✅ DO:
- Start with paper trading (testnet: true)
- Start with small positions (oper_equity: 100)
- Monitor for 1-2 weeks
- Keep API key safe
- Check logs daily
- Paper trade for 2+ weeks before live

## 🔧 Troubleshooting

### Problem: "API credentials not configured"
```bash
# Check env vars are set
echo $env:BINANCE_API_KEY
# Should show your key, not be empty
```

### Problem: "No trades happening"
```bash
# Check logs for errors
Get-Content ./logs/*.log | Select-String "ERROR"
# Check asset file
cat _raw_assets.csv
# Make sure at least 1 symbol is there
```

### Problem: "Connection timeout"
```bash
# Check internet connection
Test-NetConnection -ComputerName api.binance.com -Port 443
# Should show "TcpTestSucceeded : True"
```

### Problem: "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
# Try the problematic module directly
pip install aiohttp
```

## 📚 Next Steps

1. **Run paper trading** for 1-2 weeks
2. **Analyze results** in logs
3. **Optimize parameters** based on performance
4. **Read**: `BINANCE_MIGRATION_GUIDE.md` for full features
5. **Read**: `PERFORMANCE_TESTING.md` for advanced tuning

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Start bot | `python binance_bot.py` |
| Stop bot | `Ctrl+C` |
| View logs | `Get-Content ./logs/*.log -Wait` |
| Test connection | `python test_connection.py` |
| Switch testnet/live | Edit `config.yaml` or env var |
| Change position size | Edit `trading.oper_equity` |
| Change stop loss | Edit `trading.stop_loss_margin` |
| View configuration | `python -c "from config import get_config; print(get_config().to_dict())"` |

## 🎉 Success Checklist

After 1-2 weeks of paper trading, you should see:
- [ ] Consistent positive returns
- [ ] Win rate > 50%
- [ ] Max drawdown < 10%
- [ ] No API errors in logs
- [ ] Confidence in strategy
- [ ] All positions closed properly
- [ ] Logs are organized and readable

**If all boxes checked → Ready for live trading! 🚀**

---

**Setup Time**: ~5 minutes
**Paper Trading Duration**: 1-2 weeks minimum
**Support**: Check docs in BINANCE_MIGRATION_GUIDE.md

Good luck! 📈
