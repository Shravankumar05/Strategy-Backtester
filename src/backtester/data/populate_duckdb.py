import sys
import os
from datetime import date, timedelta
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backtester.data.duckdb_manager import DuckDBManager
from backtester.recommendation.synthetic_data_generator import SyntheticDataGenerator

def populate_duckdb_with_synthetic_data():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing DuckDB manager and synthetic data generator...")
        db_manager = DuckDBManager()
        data_generator = SyntheticDataGenerator()
        tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            # Financial
            'JPM', 'BAC', 'GS', 'WFC', 'C',
            # Consumer
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI'
        ]
        
        strategies = [
            'BollingerBandsStrategy',
            'RSIStrategy', 
            'MovingAverageCrossoverStrategy',
            'StochasticOscillatorStrategy'
        ]
        
        end_date = date.today()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        logger.info(f"Generating data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
                market_data = data_generator.generate_market_data(ticker, start_date, end_date)
                
                if market_data.empty:
                    logger.warning(f"No data generated for {ticker}")
                    continue
                
                success = db_manager.store_market_data(ticker, market_data)
                if not success:
                    logger.warning(f"Failed to store market data for {ticker}")
                    continue
                
                market_features = data_generator._calculate_market_features(market_data)
                if market_features:
                    db_manager.store_market_features(ticker, market_features, end_date)
                
                for strategy in strategies:
                    performance_metrics = data_generator._simulate_strategy_performance(strategy, market_features)
                    db_manager.store_strategy_performance(ticker=ticker, strategy_name=strategy, start_date=start_date, end_date=end_date, performance_metrics=performance_metrics)
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                continue
        
        summary = db_manager.get_data_summary()
        logger.info("Data population completed!")
        logger.info(f"Summary:")
        logger.info(f"  - Market data records: {summary.get('market_data_records', 0)}")
        logger.info(f"  - Unique tickers: {summary.get('unique_tickers', 0)}")
        logger.info(f"  - Strategy performance records: {summary.get('strategy_performance_records', 0)}")
        logger.info(f"  - Database size: {summary.get('database_size_mb', 0):.2f} MB")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to populate DuckDB: {e}")
        raise

if __name__ == "__main__":
    populate_duckdb_with_synthetic_data()