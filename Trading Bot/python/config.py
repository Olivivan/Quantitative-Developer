# encoding: utf-8
"""
Optimized Configuration System
- Support for environment variables
- Structured config with validation
- Multiple environments (dev, test, prod)
- Type-safe configuration
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict
from pathlib import Path
import json
import yaml
import logging

logger = logging.getLogger("config")


@dataclass
class APIConfig:
    """API Configuration"""
    api_key: str
    api_secret: str
    testnet: bool = True
    base_url: str = "https://api.binance.com/api"
    timeout_seconds: int = 10
    max_retries: int = 3


@dataclass
class TradingConfig:
    """Trading Strategy Configuration"""
    max_workers: int = 10
    oper_equity: float = 10000.0  # Per trade capital
    stop_loss_margin: float = 0.05  # 5%
    take_profit_ratio: float = 1.5  # 1.5:1
    limit_order_margin: float = 0.001  # 0.1%
    
    # Risk management
    max_risk_per_trade: float = 0.02  # 2% of capital
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_positions: int = 5


@dataclass
class IndicatorConfig:
    """Technical Indicator Configuration"""
    # Moving averages
    sma_fast: int = 20
    sma_slow: int = 50
    ema_9: int = 9
    ema_26: int = 26
    
    # Momentum
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_min_buy: float = 25
    stoch_max_buy: float = 75
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatility
    bb_period: int = 20
    bb_std_dev: float = 2.0
    atr_period: int = 14


@dataclass
class TimeConfig:
    """Time and Interval Configuration"""
    # Fetch intervals
    fetch_small: str = "5min"
    fetch_large: str = "4h"
    fetch_daily: str = "1d"
    
    # Sleep times (seconds)
    general_trend: int = 10 * 60  # 10 minutes
    instant_trend: int = 2 * 60   # 2 minutes
    rsi_check: int = 60            # 1 minute
    stochastic_check: int = 60     # 1 minute
    position_monitor: int = 5      # 5 seconds
    asset_unlock: int = 10 * 60    # 10 minutes


@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: str = "INFO"
    log_dir: str = "./logs"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_format: str = "%(asctime)s - %(levelname)s - %(message)s"


@dataclass
class PerformanceConfig:
    """Performance and Optimization Configuration"""
    use_pytorch: bool = True
    use_gpu: bool = False  # Requires CUDA
    use_spark: bool = False  # Requires PySpark
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60
    rate_limit_per_minute: int = 1200  # Binance default
    connection_pool_size: int = 100


class Config:
    """
    Master configuration class
    Loads from environment variables, files, or defaults
    """
    
    def __init__(self, env: str = "development"):
        self.env = env
        
        # Load base config from environment
        self.api = APIConfig(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            api_secret=os.getenv('BINANCE_API_SECRET', ''),
            testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        )
        
        self.trading = TradingConfig(
            max_workers=int(os.getenv('MAX_WORKERS', '10')),
            oper_equity=float(os.getenv('OPER_EQUITY', '10000')),
            stop_loss_margin=float(os.getenv('STOP_LOSS_MARGIN', '0.05')),
            take_profit_ratio=float(os.getenv('TAKE_PROFIT_RATIO', '1.5')),
            max_positions=int(os.getenv('MAX_POSITIONS', '5'))
        )
        
        self.indicators = IndicatorConfig()
        self.time = TimeConfig()
        self.logging = LoggingConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
        self.performance = PerformanceConfig(
            use_pytorch=os.getenv('USE_PYTORCH', 'true').lower() == 'true',
            use_gpu=os.getenv('USE_GPU', 'false').lower() == 'true',
            use_spark=os.getenv('USE_SPARK', 'false').lower() == 'true'
        )
        
        # Validate API credentials
        if not self.api.api_key or not self.api.api_secret:
            logger.warning("API credentials not configured via environment variables")
        
        logger.info(f"Configuration loaded for environment: {env}")
    
    @classmethod
    def from_file(cls, config_file: str, env: str = "development") -> 'Config':
        """Load configuration from JSON or YAML file"""
        config = cls(env)
        
        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            return config
        
        try:
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    data = json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {config_file}")
                return config
            
            # Update config from file
            if 'api' in data:
                config.api = APIConfig(**data['api'])
            if 'trading' in data:
                config.trading = TradingConfig(**data['trading'])
            if 'indicators' in data:
                config.indicators = IndicatorConfig(**data['indicators'])
            if 'time' in data:
                config.time = TimeConfig(**data['time'])
            if 'logging' in data:
                config.logging = LoggingConfig(**data['logging'])
            if 'performance' in data:
                config.performance = PerformanceConfig(**data['performance'])
            
            logger.info(f"Configuration loaded from file: {config_file}")
        
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'api': asdict(self.api),
            'trading': asdict(self.trading),
            'indicators': asdict(self.indicators),
            'time': asdict(self.time),
            'logging': asdict(self.logging),
            'performance': asdict(self.performance)
        }
    
    def to_file(self, filename: str):
        """Save configuration to file"""
        try:
            with open(filename, 'w') as f:
                if filename.endswith('.json'):
                    json.dump(self.to_dict(), f, indent=2)
                elif filename.endswith('.yaml') or filename.endswith('.yml'):
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
            
            logger.info(f"Configuration saved to: {filename}")
        
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not self.api.api_key:
            errors.append("API key is required")
        if not self.api.api_secret:
            errors.append("API secret is required")
        if self.trading.oper_equity <= 0:
            errors.append("Operating equity must be positive")
        if self.trading.max_workers <= 0:
            errors.append("Max workers must be positive")
        
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False
        
        logger.info("Configuration validation passed")
        return True


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(env: str = "development") -> Config:
    """Get or create global config instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config(env)
    return _global_config


def set_config(config: Config):
    """Set global config instance"""
    global _global_config
    _global_config = config


def load_config_from_file(config_file: str, env: str = "development") -> Config:
    """Load config from file and set as global"""
    config = Config.from_file(config_file, env)
    set_config(config)
    return config


# Example usage:
# config = get_config()
# api_key = config.api.api_key
# max_workers = config.trading.max_workers
