# encoding: utf-8
"""
Optimized Trading Bot for Binance
- Async/await architecture
- PyTorch-based technical indicators
- High-performance order management
- Comprehensive logging and monitoring
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from binance_connector import BinanceConnector
from async_trader import AsyncTrader
from pytorch_indicators import TechnicalIndicators
from config import get_config, load_config_from_file
from assetHandler import AssetHandler

# Configure logging
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
        
        # Initialize components
        self.trader = None
        self.indicators = TechnicalIndicators(
            use_gpu=self.config.performance.use_gpu
        )
        self.asset_handler = AssetHandler()
        
        # Trading stats
        self.total_trades = 0
        self.successful_trades = 0
    
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


if __name__ == '__main__':
    # Run the bot
    asyncio.run(main())
