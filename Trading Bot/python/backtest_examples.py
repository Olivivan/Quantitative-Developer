# encoding: utf-8
"""
High-Performance Backtesting Engine - Complete Usage Guide and Examples

This module demonstrates how to use the backtesting engine with various strategies
and data sources, including proper handling of transaction costs and look-ahead bias.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtest_engine import (
    BacktestEngine, TransactionCostModel, OrderSide, OrderType
)
from strategy_framework import (
    MovingAverageCrossover, RSIMeanReversion, BollingerBandBreakout,
    MacdCrossover, StrategyExecutor, TechnicalIndicators
)


# ============================================================================
# EXAMPLE 1: SIMPLE BACKTEST WITH TRANSACTION COSTS
# ============================================================================

def example_basic_backtest():
    """
    Demonstrates basic backtesting with transaction costs.
    This example shows how to:
    1. Create realistic transaction costs
    2. Submit orders
    3. Track positions
    4. Calculate metrics
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Backtest with Transaction Costs")
    print("="*70)
    
    # Create transaction cost model (realistic for stocks)
    # Commission: $0.01 per share
    # Slippage: 0.5% on market orders
    # Bid-ask spread: 0.05%
    costs = TransactionCostModel(
        commission_type='fixed',
        commission_amount=1.0,  # $1 per trade
        slippage_type='percentage',
        slippage_amount=0.005,  # 0.5%
        bid_ask_spread=0.0005   # 0.05%
    )
    
    # Initialize engine with $100,000 capital
    engine = BacktestEngine(
        initial_capital=100000,
        transaction_cost_model=costs,
        prevent_lookahead_bias=True,
        lookahead_bars=1
    )
    
    # Simulate price data
    base_price = 100.0
    prices = [base_price]
    np.random.seed(42)
    
    for i in range(100):
        # Random walk simulation
        change = np.random.normal(0, 1)
        new_price = prices[-1] * (1 + change/1000)
        prices.append(new_price)
    
    # Execute trades over the simulated data
    timestamp = pd.Timestamp('2024-01-01')
    
    # Buy 50 shares at price 100
    engine.current_timestamp = timestamp
    engine.current_prices['TEST'] = prices[0]
    buy_order_id = engine.submit_order('TEST', OrderSide.BUY, 50, prices[0])
    engine.step(timestamp, {'TEST': prices[0]})
    
    print(f"\nBuy order ID: {buy_order_id}")
    print(f"Initial capital: ${engine.initial_capital:,.2f}")
    print(f"Available capital after buy: ${engine.available_capital:,.2f}")
    
    # Simulate price movement and sell
    for i in range(1, len(prices)):
        timestamp = timestamp + timedelta(days=1)
        engine.current_timestamp = timestamp
        engine.current_prices['TEST'] = prices[i]
        engine.step(timestamp, {'TEST': prices[i]})
    
    # Sell position
    if 'TEST' in engine.positions:
        position = engine.positions['TEST']
        sell_order_id = engine.submit_order(
            'TEST', 
            OrderSide.SELL, 
            position.quantity, 
            prices[-1]
        )
        engine.step(timestamp, {'TEST': prices[-1]})
        print(f"\nSell order ID: {sell_order_id}")
    
    # Close all remaining positions
    engine.close_all_positions({'TEST': prices[-1]})
    
    # Calculate and display metrics
    metrics = engine.calculate_metrics()
    print_metrics(metrics)
    
    # Display trade log
    print("\nTrade Log:")
    trades = engine.get_trade_log()
    if not trades.empty:
        print(trades.to_string())


# ============================================================================
# EXAMPLE 2: HIGH-FREQUENCY MINUTE DATA WITH LOOKAHEAD PREVENTION
# ============================================================================

def example_hft_minute_data():
    """
    Demonstrates backtesting with high-frequency minute-level data
    and proper look-ahead bias prevention.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: High-Frequency Minute Data with Look-ahead Bias Prevention")
    print("="*70)
    
    # Generate minute-level OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1440, freq='1min')  # 1 day of minute data
    
    data = {
        'open': np.random.uniform(99, 101, 1440),
        'high': np.random.uniform(101, 102, 1440),
        'low': np.random.uniform(98, 100, 1440),
        'close': np.random.uniform(99, 101, 1440),
        'volume': np.random.uniform(1000, 5000, 1440)
    }
    
    df_minute = pd.DataFrame(data, index=dates)
    
    # Create transaction costs suitable for high-frequency trading
    costs = TransactionCostModel(
        commission_type='percentage',
        commission_amount=0.0001,  # 0.01% commission
        slippage_type='percentage',
        slippage_amount=0.001,     # 0.1% slippage
        bid_ask_spread=0.0001      # 0.01% spread
    )
    
    engine = BacktestEngine(
        initial_capital=50000,
        transaction_cost_model=costs,
        prevent_lookahead_bias=True,
        lookahead_bars=2  # 2-minute delay to prevent look-ahead bias
    )
    
    print(f"Data points: {len(df_minute)}")
    print(f"Time range: {df_minute.index[0]} to {df_minute.index[-1]}")
    print(f"Look-ahead bias prevention: Enabled (2-bar delay)")
    
    # Simple momentum-based strategy
    for i, (timestamp, row) in enumerate(df_minute.iterrows()):
        price_dict = {'STOCK': row['close']}
        engine.current_timestamp = timestamp
        engine.current_prices['STOCK'] = row['close']
        
        # Simple momentum signal (just for demonstration)
        if i > 20:
            last_20_closes = df_minute['close'].iloc[max(0, i-20):i].values
            momentum = (row['close'] - last_20_closes[0]) / last_20_closes[0]
            
            # Entry signal
            if momentum > 0.01 and 'STOCK' not in engine.positions:
                engine.submit_order(
                    'STOCK', 
                    OrderSide.BUY, 
                    100, 
                    row['close']
                )
            
            # Exit signal
            elif momentum < -0.01 and 'STOCK' in engine.positions:
                engine.submit_order(
                    'STOCK', 
                    OrderSide.SELL, 
                    100, 
                    row['close']
                )
        
        engine.step(timestamp, price_dict)
    
    # Close positions
    engine.close_all_positions({'STOCK': df_minute['close'].iloc[-1]})
    
    metrics = engine.calculate_metrics()
    print("\nHFT Backtest Results:")
    print_metrics(metrics)


# ============================================================================
# EXAMPLE 3: MULTIPLE SYMBOLS WITH PORTFOLIO MANAGEMENT
# ============================================================================

def example_portfolio_backtest():
    """
    Demonstrates backtesting a strategy across multiple symbols
    with proper portfolio allocation.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multi-Symbol Portfolio Backtest")
    print("="*70)
    
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    np.random.seed(42)
    
    # Generate price data for 3 stocks
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')  # 1 year
    
    data = {}
    for symbol in symbols:
        base_price = 100
        prices = [base_price]
        for _ in range(251):
            change = np.random.normal(0.0005, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Prevent negative prices
        
        data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1e6, 5e6, 252)
        }, index=dates)
    
    # Transaction costs
    costs = TransactionCostModel(
        commission_type='percentage',
        commission_amount=0.001,    # 0.1%
        slippage_type='percentage',
        slippage_amount=0.002,      # 0.2%
        bid_ask_spread=0.0002       # 0.02%
    )
    
    engine = BacktestEngine(
        initial_capital=100000,
        transaction_cost_model=costs,
        prevent_lookahead_bias=True
    )
    
    print(f"Symbols: {symbols}")
    print(f"Initial capital: ${engine.initial_capital:,.2f}")
    print(f"Daily allocation per symbol: ${engine.initial_capital/len(symbols):,.2f}")
    
    # Simple equal-weight portfolio rebalancing strategy
    allocation_per_symbol = engine.initial_capital / len(symbols)
    
    for i, timestamp in enumerate(dates):
        price_dict = {}
        
        for symbol in symbols:
            close_price = data[symbol]['close'].iloc[i]
            price_dict[symbol] = close_price
            
            # Rebalance every 20 days
            if i > 0 and i % 20 == 0:
                # Check current position size
                if symbol in engine.positions:
                    current_qty = engine.positions[symbol].quantity
                    target_qty = int(allocation_per_symbol / close_price)
                    
                    if target_qty < current_qty:
                        # Sell excess
                        engine.submit_order(
                            symbol,
                            OrderSide.SELL,
                            current_qty - target_qty,
                            close_price
                        )
                    elif target_qty > current_qty:
                        # Buy more
                        engine.submit_order(
                            symbol,
                            OrderSide.BUY,
                            target_qty - current_qty,
                            close_price
                        )
                else:
                    # New position
                    target_qty = int(allocation_per_symbol / close_price)
                    if target_qty > 0:
                        engine.submit_order(
                            symbol,
                            OrderSide.BUY,
                            target_qty,
                            close_price
                        )
        
        engine.step(timestamp, price_dict)
    
    # Close all positions
    final_prices = {symbol: data[symbol]['close'].iloc[-1] for symbol in symbols}
    engine.close_all_positions(final_prices)
    
    metrics = engine.calculate_metrics()
    print("\nPortfolio Backtest Results:")
    print_metrics(metrics)
    
    print("\nPosition Summary:")
    for symbol in symbols:
        price = data[symbol]['close'].iloc[-1]
        print(f"  {symbol}: Close price ${price:.2f}")


# ============================================================================
# EXAMPLE 4: STRATEGY FRAMEWORK WITH TECHNICAL INDICATORS
# ============================================================================

def example_strategy_framework():
    """
    Demonstrates using pre-built strategies from the strategy framework.
    Shows how technical indicators work with the backtesting engine.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Strategy Framework with Technical Indicators")
    print("="*70)
    
    # Generate synthetic daily OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    closes = [100]
    for _ in range(251):
        change = np.random.normal(0.0005, 0.015)
        closes.append(closes[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': [c * (1 + np.random.normal(0, 0.005)) for c in closes],
        'high': [c * (1 + abs(np.random.normal(0, 0.01))) for c in closes],
        'low': [c * (1 - abs(np.random.normal(0, 0.01))) for c in closes],
        'close': closes,
        'volume': np.random.uniform(1e6, 5e6, 252)
    }, index=dates)
    
    # Demonstrate technical indicators
    print("\nTechnical Indicators (last 5 bars):")
    print("-" * 70)
    
    closes_array = df['close'].values
    
    # Calculate indicators
    sma_20 = TechnicalIndicators.sma(closes_array, 20)
    sma_50 = TechnicalIndicators.sma(closes_array, 50)
    rsi = TechnicalIndicators.rsi(closes_array, 14)
    macd, signal, hist = TechnicalIndicators.macd(closes_array)
    upper, middle, lower = TechnicalIndicators.bollinger_bands(closes_array, 20)
    
    # Display last 5 values
    for i in range(-5, 0):
        print(f"\nBar {i}:")
        print(f"  Close: {df['close'].iloc[i]:.2f}")
        print(f"  SMA(20): {sma_20[i]:.2f}")
        print(f"  SMA(50): {sma_50[i]:.2f}")
        print(f"  RSI(14): {rsi[i]:.2f}")
        print(f"  MACD: {macd[i]:.4f}, Signal: {signal[i]:.4f}")
        print(f"  BB Upper: {upper[i]:.2f}, Middle: {middle[i]:.2f}, Lower: {lower[i]:.2f}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_metrics(metrics):
    """Pretty print backtest metrics."""
    print("\n" + "-"*70)
    print("BACKTEST METRICS")
    print("-"*70)
    print(f"Total Return:        {metrics.total_return*100:>10.2f}%")
    print(f"Annual Return:       {metrics.annual_return*100:>10.2f}%")
    print(f"Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:        {metrics.max_drawdown*100:>10.2f}%")
    print(f"Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")
    print(f"Recovery Factor:     {metrics.recovery_factor:>10.2f}")
    print(f"Win Rate:            {metrics.win_rate*100:>10.2f}%")
    print(f"Profit Factor:       {metrics.profit_factor:>10.2f}")
    print(f"-"*70)
    print(f"Total Trades:        {metrics.total_trades:>10}")
    print(f"Winning Trades:      {metrics.winning_trades:>10}")
    print(f"Losing Trades:       {metrics.losing_trades:>10}")
    print(f"Avg Win:             ${metrics.avg_win:>10.2f}")
    print(f"Avg Loss:            ${metrics.avg_loss:>10.2f}")
    print(f"-"*70)
    print(f"Total Commission:    ${metrics.total_commission:>10.2f}")
    print(f"Total Slippage:      ${metrics.total_slippage:>10.2f}")
    print("-"*70 + "\n")


def generate_sample_data(symbol: str, days: int = 252, 
                         start_price: float = 100, 
                         volatility: float = 0.02) -> pd.DataFrame:
    """
    Generate realistic OHLCV data using geometric Brownian motion.
    
    Args:
        symbol: Asset symbol
        days: Number of trading days
        start_price: Initial price
        volatility: Annualized volatility
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    closes = [start_price]
    daily_returns = np.random.normal(0.0005, volatility/np.sqrt(252), days)
    
    for ret in daily_returns[1:]:
        closes.append(closes[-1] * (1 + ret))
    
    closes = np.array(closes)
    
    data = pd.DataFrame({
        'open': closes * (1 + np.random.normal(0, 0.005, days)),
        'high': closes * (1 + abs(np.random.normal(0, 0.01, days))),
        'low': closes * (1 - abs(np.random.normal(0, 0.01, days))),
        'close': closes,
        'volume': np.random.uniform(1e6, 5e6, days)
    }, index=dates)
    
    return data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("HIGH-PERFORMANCE BACKTESTING ENGINE - EXAMPLES")
    print("="*70)
    
    try:
        example_basic_backtest()
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    try:
        example_hft_minute_data()
    except Exception as e:
        print(f"Error in Example 2: {e}")
    
    try:
        example_portfolio_backtest()
    except Exception as e:
        print(f"Error in Example 3: {e}")
    
    try:
        example_strategy_framework()
    except Exception as e:
        print(f"Error in Example 4: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
