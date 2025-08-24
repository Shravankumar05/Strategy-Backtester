import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

from ..data.yfinance_fetcher import YFinanceDataFetcher
from ..data.cache_manager import CacheManager
from ..strategy.strategy import StrategyRegistry
from ..simulation.engine import SimulationEngine
from ..simulation.config import SimulationConfig
from ..metrics.performance import PerformanceMetrics

@dataclass
class BacktestRecord:
    ticker: str
    strategy_name: str
    strategy_params: Dict[str, Any]
    start_date: date
    end_date: date
    metrics: Dict[str, float]
    market_features: Dict[str, float]
    data_quality: Dict[str, Any]
    backtest_timestamp: datetime

class DataCollector:
    def __init__(self, cache_dir: str = "recommendation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.data_fetcher = YFinanceDataFetcher(CacheManager())

        self.simulation_config = SimulationConfig(
            initial_capital=10000.0,
            leverage=1.0,
            transaction_cost=0.001,
            slippage=0.0005,
            position_sizing="fixed_fraction",
            position_size=0.1
        )

        self.default_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "JPM", "BAC", "GS", "WFC", "C",
            "PG", "KO", "PEP", "WMT", "HD", "MCD",
            "JNJ", "UNH", "PFE", "ABBV", "TMO",
            "XOM", "CVX", "COP", "SLB",
            "BA", "CAT", "GE", "LMT",
            "SPY", "QQQ", "IWM", "VTI", "EFA"
        ]

    def collect_training_data(
        self,
        tickers: Optional[List[str]] = None,
        lookback_days: int = 252,
        min_data_points: int = 100,
        max_workers: int = 4
    ) -> List[BacktestRecord]:
        if tickers is None:
            tickers = self.default_tickers

        strategy_names = StrategyRegistry.list_strategies()
        strategy_names = [name for name in strategy_names if name != "CustomStrategy"]

        self.logger.info(f"Collecting data for {len(tickers)} tickers and {len(strategy_names)} strategies")

        records = []

        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for ticker in tickers:
                for strategy_name in strategy_names:
                    future = executor.submit(
                        self._run_single_backtest,
                        ticker, strategy_name, start_date, end_date, min_data_points
                    )
                    futures.append(future)

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    record = future.result()
                    if record is not None:
                        records.append(record)

                    if i % 10 == 0:
                        self.logger.info(f"Completed {i}/{len(futures)} backtests")

                except Exception as e:
                    self.logger.warning(f"Backtest failed: {e}")
                    continue

        self.logger.info(f"Collected {len(records)} successful backtest records")
        return records

    def _run_single_backtest(
        self,
        ticker: str,
        strategy_name: str,
        start_date: date,
        end_date: date,
        min_data_points: int
    ) -> Optional[BacktestRecord]:
        try:
            data = self.data_fetcher.fetch_ohlcv(ticker, start_date, end_date)

            if len(data) < min_data_points:
                self.logger.debug(f"Insufficient data for {ticker}: {len(data)} points")
                return None

            strategy = StrategyRegistry.create_strategy(strategy_name)

            signals = strategy.generate_signals(data)

            engine = SimulationEngine(self.simulation_config)
            result = engine.run_simulation(data, signals)

            equity_series = result.equity_curve['equity'] if 'equity' in result.equity_curve.columns else result.equity_curve.iloc[:, 0]
            metrics = PerformanceMetrics.calculate_all_metrics(equity_series, result.trades)

            market_features = self._calculate_market_features(data)

            data_quality = self._assess_data_quality(data)

            return BacktestRecord(
                ticker=ticker,
                strategy_name=strategy_name,
                strategy_params=strategy.get_parameters(),
                start_date=start_date,
                end_date=end_date,
                metrics=metrics,
                market_features=market_features,
                data_quality=data_quality,
                backtest_timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.debug(f"Failed backtest for {ticker}-{strategy_name}: {e}")
            return None

    def _calculate_market_features(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            features = {}

            returns = data['Close'].pct_change().dropna()
            features['volatility'] = returns.std() * np.sqrt(252)
            features['mean_return'] = returns.mean() * 252
            features['skewness'] = returns.skew()
            features['kurtosis'] = returns.kurtosis()

            features['total_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

            features['avg_volume'] = data['Volume'].mean()
            features['volume_volatility'] = data['Volume'].std() / data['Volume'].mean()

            features['price_level'] = data['Close'].iloc[-1]
            features['price_range'] = (data['High'].max() - data['Low'].min()) / data['Close'].mean()

            ma_20 = data['Close'].rolling(20).mean()
            features['ma_trend'] = (ma_20.iloc[-1] / ma_20.iloc[-21] - 1) if len(ma_20) > 21 else 0

            rolling_vol = returns.rolling(20).std()
            features['vol_regime'] = rolling_vol.iloc[-1] / rolling_vol.mean()

            return features

        except Exception as e:
            self.logger.warning(f"Failed to calculate market features: {e}")
            return {}

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            quality = {}

            quality['data_points'] = len(data)
            quality['missing_values'] = data.isnull().sum().sum()
            quality['zero_volume_days'] = (data['Volume'] == 0).sum()
            quality['extreme_moves'] = (data['Close'].pct_change().abs() > 0.2).sum()
            quality['date_range_days'] = (data.index[-1] - data.index[0]).days

            return quality

        except Exception:
            return {}

    def save_training_data(self, records: List[BacktestRecord], filename: str = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.pkl"

        filepath = os.path.join(self.cache_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(records, f)

        self.logger.info(f"Saved {len(records)} records to {filepath}")
        return filepath

    def load_training_data(self, filename: str) -> List[BacktestRecord]:
        filepath = os.path.join(self.cache_dir, filename)

        with open(filepath, 'rb') as f:
            records = pickle.load(f)

        self.logger.info(f"Loaded {len(records)} records from {filepath}")
        return records

    def get_latest_training_data(self) -> Optional[List[BacktestRecord]]:
        try:
            files = [f for f in os.listdir(self.cache_dir) if f.startswith("training_data_") and f.endswith(".pkl")]
            if not files:
                return None

            latest_file = max(files)
            return self.load_training_data(latest_file)

        except Exception as e:
            self.logger.error(f"Failed to load latest training data: {e}")
            return None