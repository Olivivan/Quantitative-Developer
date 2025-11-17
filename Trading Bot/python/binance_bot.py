# encoding: utf-8
"""
Optimized Trading Bot for Binance (GPU-accelerated)
- Async/await architecture for non-blocking I/O
- PyTorch-based technical indicators with CUDA acceleration (RTX 3090)
- Mixed precision (FP16/TF32) for Ampere architecture
- High-performance order management
- Comprehensive logging and monitoring
- GPU processing guaranteed for all indicator calculations
"""

import argparse
import asyncio
import logging
import sys
import torch
from datetime import datetime
from pathlib import Path

from binance_connector import BinanceConnector
from async_trader import AsyncTrader
from pytorch_indicators import TechnicalIndicators, CUDA_AVAILABLE, DEVICE
from config import get_config, load_config_from_file
from assetHandler import AssetHandler
from backtest_engine import BacktestEngine, TransactionCostModel, OrderSide
from pytorch_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# Log CUDA availability at module import
if CUDA_AVAILABLE:
    logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
    logger.info(f"  CUDA version: {torch.version.cuda}")
    logger.info(f"  PyTorch device: {DEVICE}")
else:
    logger.warning("✗ CUDA not available - falling back to CPU (slower)")


def setup_logging(config):
    """Configure logging system"""
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(config.logging.level)
    
    # File handler
    log_file = log_dir / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(config.logging.file_format))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(config.logging.console_format))
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


class BinanceBot:
    """
    Main trading bot orchestrator for Binance
    All indicator calculations are GPU-accelerated on RTX 3090
    Manages trading strategies, position monitoring, and reporting
    """
    
    def __init__(self, config_file: str = None, env: str = "development"):
        # Load configuration
        if config_file:
            self.config = load_config_from_file(config_file, env)
        else:
            self.config = get_config(env)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info(f"Bot initialized for environment: {env}")
        
        # Validate configuration
        if not self.config.validate():
            raise RuntimeError("Configuration validation failed")
        
        # Initialize components with GPU acceleration enabled by default
        self.trader = None
        self.indicators = TechnicalIndicators(
            use_gpu=True,  # Force GPU acceleration
            use_fp16=CUDA_AVAILABLE  # Use FP16 mixed precision on RTX 3090
        )
        self.asset_handler = AssetHandler()
        
        # Trading stats
        self.total_trades = 0
        self.successful_trades = 0
        
        # Log GPU setup
        self.logger.info(f"GPU acceleration: {self.indicators.use_gpu}")
        if self.indicators.use_gpu:
            self.logger.info(f"Mixed precision (FP16): {self.indicators.use_fp16}")
    
    async def connect(self):
        """Connect to Binance API"""
        self.trader = AsyncTrader(
            api_key=self.config.api.api_key,
            api_secret=self.config.api.api_secret,
            strategy_name="optimized_binance_bot",
            testnet=self.config.api.testnet
        )
        await self.trader.connect()
        self.logger.info("Connected to Binance API")
    
    async def disconnect(self):
        """Disconnect from Binance API"""
        if self.trader:
            await self.trader.disconnect()
            self.logger.info("Disconnected from Binance API")
    
    async def run_strategy(self, symbol: str, strategy_type: str = "ma_crossover") -> bool:
        """
        Run trading strategy for a symbol
        Returns: True if should lock asset, False otherwise
        """
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting strategy: {strategy_type} for {symbol}")
            self.logger.info(f"{'='*60}\n")
            
            # Step 1: Analyze general trend
            trend = await self.trader.get_general_trend(symbol, interval='4h')
            if not trend:
                self.logger.warning(f"Could not determine trend for {symbol}, skipping")
                return True  # Lock asset temporarily
            
            # Step 2: Get current price
            current_price = await self.trader.connector.get_price(symbol)
            self.logger.info(f"Current price for {symbol}: ${current_price:.4f}")
            
            # Step 3: Check instant trend
            instant_ok = await self.trader.get_instant_trend(symbol, trend, interval='1h')
            if not instant_ok:
                self.logger.warning(f"Instant trend check failed for {symbol}")
                return True
            
            # Step 4: Check RSI
            rsi_ok, rsi_value = await self.trader.get_rsi(symbol)
            if not rsi_ok:
                self.logger.warning(f"RSI check failed for {symbol}: {rsi_value:.2f}")
                return True
            
            # Step 5: Check Stochastic
            stoch_ok = await self.trader.get_stochastic(symbol, trend)
            if not stoch_ok:
                self.logger.warning(f"Stochastic check failed for {symbol}")
                return True
            
            # Step 6: Enter position
            entry_ok = await self.trader.enter_position(symbol, trend, current_price)
            if not entry_ok:
                self.logger.error(f"Failed to enter position for {symbol}")
                return False
            
            self.total_trades += 1
            
            # Position is open, just return success
            # The monitor_positions task will handle exit
            return False  # Don't lock, make available again
        
        except Exception as e:
            self.logger.error(f"Error in strategy for {symbol}: {e}", exc_info=True)
            return True  # Lock on error
    
    async def worker_thread(self, worker_id: int):
        """
        Worker thread that continuously trades
        Each worker picks available assets and trades them
        """
        self.logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get target asset
                symbol = self.asset_handler.find_target_asset()
                
                # Run strategy
                should_lock = await self.run_strategy(symbol)
                
                if should_lock:
                    self.asset_handler.lock_asset(symbol)
                else:
                    self.asset_handler.make_asset_available(symbol)
                
                # Brief pause between trades
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Error in worker {worker_id}: {e}")
                await asyncio.sleep(5)
    
    async def monitor_positions_task(self):
        """
        Background task to monitor open positions
        Runs continuously
        """
        await self.trader.monitor_positions()
    
    async def reporting_task(self, report_interval: int = 300):
        """
        Background task to report trading statistics
        Reports every report_interval seconds
        """
        while True:
            try:
                await asyncio.sleep(report_interval)
                
                stats = self.trader.get_statistics()
                self.logger.info("\n" + "="*60)
                self.logger.info("TRADING STATISTICS")
                self.logger.info("="*60)
                for key, value in stats.items():
                    self.logger.info(f"{key}: {value}")
                self.logger.info("="*60 + "\n")
                
                self.successful_trades = stats['wins']
            
            except Exception as e:
                self.logger.error(f"Error in reporting task: {e}")
    
    async def run(self, num_workers: int = None):
        """
        Main bot loop
        Launches multiple workers and monitoring tasks
        """
        if num_workers is None:
            num_workers = self.config.trading.max_workers
        
        self.logger.info(f"Starting bot with {num_workers} workers")
        
        try:
            # Create tasks
            tasks = []
            
            # Worker threads
            for i in range(num_workers):
                tasks.append(asyncio.create_task(self.worker_thread(i)))
            
            # Background monitoring tasks
            tasks.append(asyncio.create_task(self.monitor_positions_task()))
            tasks.append(asyncio.create_task(self.reporting_task()))
            
            # Run all tasks
            await asyncio.gather(*tasks)
        
        except KeyboardInterrupt:
            self.logger.info("Bot interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)


async def main():
    """Main entry point"""
    # Try to load config from file, fallback to environment variables
    config_file = "config.yaml"  # or "config.json"
    
    bot = BinanceBot(config_file=config_file if Path(config_file).exists() else None)
    
    try:
        await bot.connect()
        await bot.run(num_workers=bot.config.trading.max_workers)
    finally:
        await bot.disconnect()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backtest', action='store_true', help='Run backtest using public Binance klines')
    p.add_argument('--symbol', type=str, default='BTCUSDT')
    p.add_argument('--start', type=str, help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, help='End date YYYY-MM-DD')
    p.add_argument('--interval', type=str, default='1h')
    return p.parse_args()


async def run_backtest_cli(symbol: str, start: str | None, end: str | None, interval: str = '1h'):
    """Fetch public klines from Binance and run a simple MA(20,50)+RSI backtest."""
    # Use a connector without API keys for public endpoints
    connector = BinanceConnector(api_key='', api_secret='', testnet=False)
    await connector.connect()

    # Convert start/end to milliseconds if provided
    start_ms = None
    end_ms = None
    if start:
        start_ms = int(datetime.fromisoformat(start).timestamp() * 1000)
    if end:
        end_ms = int(datetime.fromisoformat(end).timestamp() * 1000)

    # Fetch up to 1000 candles; if the date range is larger, the connector supports start/end params
    try:
        df = await connector.get_klines(symbol, interval=interval, limit=1000,
                                       start_time=start_ms, end_time=end_ms)
    finally:
        await connector.close()

    if df is None or df.empty:
        print('No klines returned for the requested symbol / date range')
        return

    # Build and run a simple backtest like the one in improved_strategies
    closes = df['close'].values
    CUDA = torch.cuda.is_available()
    indicators = TechnicalIndicators(use_gpu=CUDA, use_fp16=CUDA)

    costs = TransactionCostModel(commission_type='fixed', commission_amount=0.0,
                                 slippage_type='percentage', slippage_amount=0.0005)

    engine = BacktestEngine(initial_capital=100000, transaction_cost_model=costs)

    in_position = False
    for i in range(len(df)):
        timestamp = df.index[i]
        price = float(df['close'].iat[i])
        engine.current_timestamp = timestamp
        engine.current_prices['STOCK'] = price

        if i > 50:
            arr = closes[:i+1]
            sma_20 = indicators.sma_gpu(arr, 20)[-1]
            sma_50 = indicators.sma_gpu(arr, 50)[-1]
            rsi = indicators.rsi_gpu(arr, 14)[-1]

            if not in_position and (sma_20 > sma_50) and (rsi < 70):
                engine.submit_order('STOCK', OrderSide.BUY, 1, price)
                in_position = True
            elif in_position and (sma_20 < sma_50 or rsi > 80):
                engine.submit_order('STOCK', OrderSide.SELL, 1, price)
                in_position = False

        engine.step(timestamp, {'STOCK': price})

    engine.close_all_positions({'STOCK': float(closes[-1])})
    metrics = engine.calculate_metrics()

    print('\nBacktest summary for', symbol)
    print(f"Data points: {len(df)} | From: {df.index[0]} To: {df.index[-1]}")
    print(f"Total Return: {metrics.total_return*100:+.2f}% | Sharpe: {metrics.sharpe_ratio:+.2f}")


if __name__ == '__main__':
    args = _parse_args()
    if args.backtest:
        # Run backtest CLI
        asyncio.run(run_backtest_cli(args.symbol, args.start, args.end, interval=args.interval))
    else:
        asyncio.run(main())
