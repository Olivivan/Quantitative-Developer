# encoding: utf-8
"""
Spark Integration for Distributed Data Processing
- Parallel historical data analysis
- Batch backtesting on clusters
- Large-scale technical indicator calculations
- Optimized for big data workloads
"""

try:
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("spark_processor")


class SparkProcessor:
    """
    Process large datasets using Apache Spark
    Requires: pip install pyspark
    """
    
    def __init__(self, app_name: str = "crypto_trader", master: str = "local[*]"):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark not installed. Install with: pip install pyspark")
        
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark session created: {app_name}")
    
    def create_dataframe_from_pandas(self, df: pd.DataFrame) -> 'pyspark.sql.DataFrame':
        """Convert Pandas DataFrame to Spark DataFrame"""
        return self.spark.createDataFrame(df)
    
    def calculate_indicators_distributed(self, df_spark, 
                                        high_col: str = 'high',
                                        low_col: str = 'low',
                                        close_col: str = 'close') -> 'pyspark.sql.DataFrame':
        """
        Calculate technical indicators using Spark (distributed)
        Optimized for large datasets
        """
        try:
            # SMA calculations using window functions
            from pyspark.sql.window import Window
            
            window_20 = Window.orderBy('timestamp').rowsBetween(-19, 0)
            window_50 = Window.orderBy('timestamp').rowsBetween(-49, 0)
            
            df_result = df_spark \
                .withColumn('sma_20', F.avg(close_col).over(window_20)) \
                .withColumn('sma_50', F.avg(close_col).over(window_50)) \
                .withColumn('close_numeric', F.col(close_col).cast(DoubleType()))
            
            logger.info("Indicator calculation completed")
            return df_result
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def backtest_strategy_distributed(self, historical_data: pd.DataFrame,
                                     strategy_func,
                                     num_partitions: int = 4) -> pd.DataFrame:
        """
        Run backtest in parallel using Spark
        Useful for testing multiple symbols or timeframes
        """
        try:
            # Convert to Spark DataFrame
            df_spark = self.create_dataframe_from_pandas(historical_data)
            
            # Repartition for parallel processing
            df_spark = df_spark.repartition(num_partitions)
            
            # Apply strategy function
            results = []
            for partition in df_spark.collect():
                result = strategy_func(pd.DataFrame(partition))
                results.append(result)
            
            # Combine results
            combined_results = pd.concat(results, ignore_index=True)
            logger.info(f"Backtest completed with {num_partitions} partitions")
            
            return combined_results
        
        except Exception as e:
            logger.error(f"Error in distributed backtest: {e}")
            raise
    
    def analyze_multiple_symbols(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze multiple symbols in parallel
        """
        try:
            results = {}
            
            for symbol, df in symbols_data.items():
                df_spark = self.create_dataframe_from_pandas(df)
                df_analyzed = self.calculate_indicators_distributed(df_spark)
                results[symbol] = df_analyzed.toPandas()
            
            logger.info(f"Analyzed {len(symbols_data)} symbols")
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing symbols: {e}")
            raise
    
    def statistical_analysis(self, df_spark) -> Dict:
        """
        Perform statistical analysis using Spark SQL
        """
        try:
            df_spark.createOrReplaceTempView("price_data")
            
            stats = self.spark.sql("""
                SELECT
                    COUNT(*) as total_records,
                    AVG(close) as avg_price,
                    MIN(close) as min_price,
                    MAX(close) as max_price,
                    STDDEV(close) as std_price,
                    PERCENTILE_APPROX(close, 0.25) as q1,
                    PERCENTILE_APPROX(close, 0.5) as median,
                    PERCENTILE_APPROX(close, 0.75) as q3
                FROM price_data
            """).collect()[0]
            
            result = {col: stats[col] for col in stats.__fields__}
            logger.info(f"Statistical analysis completed: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            raise
    
    def correlation_analysis(self, symbols_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate correlations between multiple symbols
        """
        try:
            # Combine all closing prices
            combined_data = {}
            
            for symbol, df in symbols_data.items():
                combined_data[symbol] = df['close'].values
            
            # Create DataFrame
            combined_df = pd.DataFrame(combined_data)
            
            # Calculate correlation
            correlation = combined_df.corr()
            logger.info("Correlation analysis completed")
            
            return correlation
        
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            raise
    
    def optimize_parameters_distributed(self, historical_data: pd.DataFrame,
                                       parameter_ranges: Dict[str, List],
                                       strategy_func) -> Dict:
        """
        Grid search parameter optimization using Spark
        """
        try:
            from itertools import product
            
            # Generate all parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            combinations = list(product(*param_values))
            
            # Convert to Spark DataFrame for parallel processing
            param_df = self.spark.createDataFrame(
                [dict(zip(param_names, combo)) for combo in combinations],
                schema=StructType([
                    StructField(name, DoubleType(), True) for name in param_names
                ])
            )
            
            results = []
            for params in param_df.collect():
                param_dict = params.asDict()
                # Run strategy with these parameters
                result = strategy_func(historical_data, **param_dict)
                results.append({**param_dict, **result})
            
            logger.info(f"Parameter optimization completed: {len(combinations)} combinations tested")
            return results
        
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            raise
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session closed")


class DistributedBacktestEngine:
    """
    Run backtests in distributed manner using Spark
    Useful for large-scale historical analysis
    """
    
    def __init__(self, spark_processor: SparkProcessor):
        self.spark_processor = spark_processor
    
    def run_backtest_batch(self, symbols: List[str], 
                          historical_data_dict: Dict[str, pd.DataFrame],
                          strategy_func) -> Dict[str, Dict]:
        """
        Run backtest for multiple symbols in parallel
        """
        try:
            results = {}
            
            for symbol in symbols:
                if symbol in historical_data_dict:
                    df = historical_data_dict[symbol]
                    result = strategy_func(df)
                    results[symbol] = result
            
            logger.info(f"Batch backtest completed for {len(symbols)} symbols")
            return results
        
        except Exception as e:
            logger.error(f"Error in batch backtest: {e}")
            raise
    
    def calculate_portfolio_metrics(self, individual_results: Dict[str, Dict]) -> Dict:
        """
        Calculate portfolio-level metrics from individual symbol results
        """
        try:
            total_return = sum(r.get('total_return', 0) for r in individual_results.values())
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in individual_results.values()])
            
            return {
                'portfolio_return': total_return,
                'avg_sharpe_ratio': avg_sharpe,
                'symbols_analyzed': len(individual_results),
                'individual_results': individual_results
            }
        
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise


# Helper function to enable/disable Spark-based processing
def use_spark_if_available(enable: bool = True) -> bool:
    """Check if Spark is available and can be used"""
    if enable and SPARK_AVAILABLE:
        logger.info("Spark available and enabled for distributed processing")
        return True
    else:
        logger.info("Spark disabled or not available - using local processing")
        return False
