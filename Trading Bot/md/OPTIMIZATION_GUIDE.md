# Performance Optimization Guide for High-Frequency Backtesting

This guide covers optimization techniques for achieving maximum performance when backtesting high-frequency trading strategies.

## 1. Memory Optimization

### Use NumPy for Vectorized Operations

**❌ Slow (Python loops):**
```python
def calculate_sma_slow(prices, period):
    result = []
    for i in range(len(prices)):
        if i < period:
            result.append(float('nan'))
        else:
            result.append(sum(prices[i-period:i]) / period)
    return result
```

**✅ Fast (NumPy vectorized):**
```python
def calculate_sma_fast(prices, period):
    import numpy as np
    sma = np.convolve(prices, np.ones(period) / period, mode='valid')
    result = np.full(len(prices), np.nan)
    result[period-1:] = sma
    return result
```

**Performance improvement: 10-50x faster**

### Use float32 for Large Arrays

```python
import numpy as np

# ❌ Uses more memory
prices_64 = np.array(prices, dtype=np.float64)

# ✅ Uses half the memory
prices_32 = np.array(prices, dtype=np.float32)

# Typically no significant precision loss for trading data
```

### Pre-allocate Arrays

```python
# ❌ Slow - reallocates on each iteration
results = []
for i in range(1000000):
    results.append(calculate(i))

# ✅ Fast - allocate once
results = np.empty(1000000)
for i in range(1000000):
    results[i] = calculate(i)
```

## 2. Calculation Optimization

### Cache Indicator Values

```python
# ❌ Recalculates every bar
def on_bar(self, bar):
    sma = self.calculate_sma(bar.history)
    rsi = self.calculate_rsi(bar.history)
    macd = self.calculate_macd(bar.history)
    # ...

# ✅ Calculate once and reuse
def on_bar(self, bar):
    if not hasattr(self, 'sma_cache'):
        self.sma_cache = self.calculate_sma(bar.history)
        self.rsi_cache = self.calculate_rsi(bar.history)
        self.macd_cache = self.calculate_macd(bar.history)
    
    sma = self.sma_cache[-1]
    rsi = self.rsi_cache[-1]
    macd = self.macd_cache[-1]
```

### Use Efficient Data Structures

```python
import numpy as np

# ❌ Slow - appending to list
prices = []
for bar in data:
    prices.append(bar.close)

# ✅ Fast - pre-allocate and fill
prices = np.zeros(len(data))
for i, bar in enumerate(data):
    prices[i] = bar.close
```

### Minimize Function Calls in Loops

```python
# ❌ Slow - many function calls
for i in range(len(data)):
    if should_trade(data[i]):
        price = get_price(data[i])
        quantity = calculate_quantity(data[i])
        engine.submit_order(symbol, side, quantity, price)

# ✅ Fast - cache frequently accessed values
prices = data['close'].values
should_trade_array = np.array([should_trade(d) for d in data])
for i, should_trade_val in enumerate(should_trade_array):
    if should_trade_val:
        engine.submit_order(symbol, side, 100, prices[i])
```

## 3. Data Structure Optimization

### Choose Appropriate Data Types

```python
import pandas as pd

# When reading CSV
df = pd.read_csv('data.csv', dtype={
    'close': 'float32',      # Price data
    'volume': 'int32',       # Volume
    'symbol': 'category'     # Repeated strings
})

# This reduces memory usage by 50-70% compared to default
```

### Use Pandas MultiIndex for Multi-Symbol Data

```python
# ❌ Slow - loop through dictionary
for symbol, data in symbol_dict.items():
    for timestamp, row in data.iterrows():
        # process...

# ✅ Fast - use MultiIndex
multi_index_data = pd.concat([data.assign(symbol=sym) 
                               for sym, data in symbol_dict.items()],
                              keys=symbol_dict.keys(),
                              names=['symbol', 'timestamp'])
for (symbol, timestamp), row in multi_index_data.iterrows():
    # process...
```

## 4. I/O Optimization

### Read Data Efficiently

```python
import pandas as pd

# ❌ Slow - default read
df = pd.read_csv('large_file.csv')

# ✅ Fast - specify data types and columns
df = pd.read_csv(
    'large_file.csv',
    dtype={'close': 'float32', 'volume': 'int32'},
    usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    index_col='timestamp',
    parse_dates=['timestamp']
)

# Even faster - use binary format
import pickle
df = pickle.load(open('data.pkl', 'rb'))
```

### Use Efficient Data Formats

```python
# ❌ Slowest - CSV
df.to_csv('data.csv')  # Slow to read/write

# ✅ Medium - Parquet
df.to_parquet('data.parquet')  # 10x faster, smaller file

# ✅ Fast - HDF5
df.to_hdf('data.h5', 'data')  # Very fast, compression

# For real-time use
import pickle
df.to_pickle('data.pkl')  # Fastest load
```

## 5. Algorithm Optimization

### Use Incremental Updates Instead of Recalculation

```python
# ❌ Recalculates entire EMA each bar
def on_bar(self, bar):
    closes = self.history['close'].values
    ema = TechnicalIndicators.ema(closes, 20)
    current_ema = ema[-1]

# ✅ Incrementally update EMA
def on_bar(self, bar):
    if not hasattr(self, 'ema_value'):
        self.ema_value = bar.close
    
    multiplier = 2 / (20 + 1)
    self.ema_value = bar.close * multiplier + self.ema_value * (1 - multiplier)
    current_ema = self.ema_value
```

### Use Vectorized Backtesting for Parameter Optimization

```python
import numpy as np

# ❌ Slow - one parameter set at a time
results = []
for fast in range(10, 30):
    for slow in range(40, 60):
        # Run full backtest
        result = run_backtest(fast, slow)
        results.append(result)

# ✅ Fast - vectorized calculation
fast_periods = np.arange(10, 30)
slow_periods = np.arange(40, 60)
# Calculate all combinations at once using NumPy broadcasting
```

## 6. Backtesting Engine Optimization

### Batch Process Orders

```python
# ❌ Slow - process each order individually
for order in orders:
    engine.submit_order(...)

# ✅ Fast - batch orders
orders_batch = [
    {'symbol': 'A', 'side': BUY, 'quantity': 100},
    {'symbol': 'B', 'side': BUY, 'quantity': 50},
]
for order in orders_batch:
    engine.submit_order(**order)
```

### Use Period-Based Aggregation

```python
# ❌ Too slow - process every tick
for tick in tick_data:
    engine.step(tick)

# ✅ Faster - aggregate to minute bars
minute_data = tick_data.resample('1min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
for timestamp, row in minute_data.iterrows():
    engine.step(timestamp, row)
```

## 7. Profiling and Benchmarking

### Identify Performance Bottlenecks

```python
import cProfile
import pstats

def run_backtest():
    # Your backtest code here
    pass

# Profile the code
profiler = cProfile.Profile()
profiler.enable()
run_backtest()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Print top 20 functions
```

### Use timeit for Micro-Benchmarks

```python
import timeit

# Compare performance
def method1():
    result = []
    for i in range(1000000):
        result.append(i)

def method2():
    result = np.arange(1000000)

time1 = timeit.timeit(method1, number=10)
time2 = timeit.timeit(method2, number=10)

print(f"Method 1: {time1:.4f}s")
print(f"Method 2: {time2:.4f}s")
print(f"Speedup: {time1/time2:.1f}x")
```

## 8. Parallelization

### Multi-Processing for Parameter Sweeps

```python
from multiprocessing import Pool
import numpy as np

def backtest_params(params):
    fast, slow = params
    return run_backtest(fast, slow)

# Generate all parameter combinations
param_grid = [
    (fast, slow) 
    for fast in range(10, 30)
    for slow in range(40, 60)
]

# Run in parallel
with Pool(processes=4) as pool:
    results = pool.map(backtest_params, param_grid)
```

### Threading for I/O Operations

```python
from concurrent.futures import ThreadPoolExecutor

def download_data(symbol):
    # Download and process data
    return data

symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']

with ThreadPoolExecutor(max_workers=4) as executor:
    all_data = executor.map(download_data, symbols)
```

## 9. Optimization Checklist

- [ ] Profile code to identify bottlenecks
- [ ] Use NumPy/Pandas for numerical operations
- [ ] Use float32 for large arrays when precision allows
- [ ] Pre-allocate arrays
- [ ] Cache frequently calculated values
- [ ] Use efficient data structures (MultiIndex for multi-symbol)
- [ ] Read/write data in binary formats (Parquet, HDF5)
- [ ] Minimize function calls in loops
- [ ] Use vectorized operations instead of loops
- [ ] Implement incremental indicator updates
- [ ] Batch process orders and data
- [ ] Use period-based aggregation for high-frequency data
- [ ] Consider multi-processing for parameter optimization
- [ ] Profile again to verify improvements

## 10. Expected Performance

### Typical Processing Speeds

| Data Frequency | Processing Rate | Notes |
|----------------|-----------------|-------|
| Daily | 1M+ bars/sec | Negligible time for strategies |
| Hourly | 100K+ bars/sec | Fast optimization possible |
| Minute | 10K+ bars/sec | Reasonable for parameter sweeps |
| Second | 1K+ bars/sec | Consider aggregation or sampling |
| Tick | 100+ bars/sec | Focus on core indicator logic |

### Memory Usage

| Data Points | NumPy float64 | NumPy float32 | Reduction |
|-------------|---------------|---------------|-----------|
| 1M | 8 MB | 4 MB | 50% |
| 10M | 80 MB | 40 MB | 50% |
| 100M | 800 MB | 400 MB | 50% |
| 1B | 8 GB | 4 GB | 50% |

## 11. Advanced Optimization Techniques

### Use Numba JIT Compilation

```python
from numba import jit

# ❌ Slow - pure Python
def calculate_returns(prices):
    returns = np.zeros(len(prices) - 1)
    for i in range(len(prices) - 1):
        returns[i] = (prices[i+1] - prices[i]) / prices[i]
    return returns

# ✅ Fast - JIT compiled
@jit(nopython=True)
def calculate_returns_fast(prices):
    returns = np.zeros(len(prices) - 1)
    for i in range(len(prices) - 1):
        returns[i] = (prices[i+1] - prices[i]) / prices[i]
    return returns

# First call is slow (compilation), subsequent calls are fast
```

### Use Cython for Critical Paths

```cython
# indicators.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_rsi_fast(double[:] prices, int period):
    cdef int n = len(prices)
    cdef double[:] rsi = np.zeros(n)
    # Implementation...
    return np.asarray(rsi)
```

## 12. Common Pitfalls to Avoid

1. **Memory leaks in loops** - Clear caches periodically
2. **Unnecessary string operations** - Use numeric IDs
3. **Repeated file I/O** - Load data once at startup
4. **Inefficient sorting** - Use pre-sorted data when possible
5. **Memory fragmentation** - Use contiguous arrays
6. **Context switching** - Limit number of threads
7. **I/O bottlenecks** - Use binary formats and compression

---

**Target: Process 1M+ daily bars per second on modern hardware**
