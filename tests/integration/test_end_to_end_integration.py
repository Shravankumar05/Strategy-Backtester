import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import date
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from backtester.data.yfinance_fetcher import YFinanceDataFetcher
from backtester.data.cache_manager import CacheManager
from backtester.strategy.ma_crossover import MovingAverageCrossoverStrategy
from backtester.strategy.rsi_strategy import RSIStrategy
from backtester.simulation.engine import SimulationEngine
from backtester.simulation.config import SimulationConfig
from backtester.metrics.performance import PerformanceMetrics

class TestEndToEndIntegration:
    @pytest.fixture
    def sample_ohlcv_data(self):
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        np.random.seed(42)
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = initial_price * (1 + returns).cumprod()
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0.005, 0.002, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0.005, 0.002, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        })
        data.set_index('Date', inplace=True)
        return data
    
    @pytest.fixture
    def simulation_config(self):
        return SimulationConfig(
            initial_capital=10000.0,
            leverage=1.0,
            transaction_cost=0.001,
            slippage=0.0005,
            position_sizing="fixed_fraction",
            position_size=0.1
        )
    
    def test_complete_ma_crossover_workflow(self, sample_ohlcv_data, simulation_config):
        strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=20)
        signals = strategy.generate_signals(sample_ohlcv_data)
        assert not signals.empty
        assert 'signal' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()
        engine = SimulationEngine(simulation_config)
        result = engine.run_simulation(sample_ohlcv_data, signals)
        assert result is not None
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'trades')
        assert not result.equity_curve.empty
        metrics = PerformanceMetrics.calculate_all_metrics(result.equity_curve, result.trades)
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'total_return' in metrics
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert 0 <= metrics['max_drawdown'] <= 1
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_complete_rsi_workflow(self, sample_ohlcv_data, simulation_config):
        strategy = RSIStrategy(period=14, overbought=70, oversold=30)
        signals = strategy.generate_signals(sample_ohlcv_data)
        assert not signals.empty
        assert 'signal' in signals.columns
        engine = SimulationEngine(simulation_config)
        result = engine.run_simulation(sample_ohlcv_data, signals)
        assert result is not None
        assert not result.equity_curve.empty
        metrics = PerformanceMetrics.calculate_all_metrics(result.equity_curve, result.trades)
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in ['sharpe_ratio', 'max_drawdown', 'win_rate'])
    
    @patch('backtester.data.yfinance_fetcher.yf.download')
    def test_data_fetching_integration(self, mock_yf_download, sample_ohlcv_data):
        mock_yf_download.return_value = sample_ohlcv_data
        cache_manager = CacheManager()
        data_fetcher = YFinanceDataFetcher(cache_manager)
        data = data_fetcher.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 3, 31))
        assert not data.empty
        assert len(data) > 0
        data2 = data_fetcher.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 3, 31))
        pd.testing.assert_frame_equal(data, data2)
        assert mock_yf_download.call_count == 1
    
    def test_error_handling_integration(self, simulation_config):
        empty_data = pd.DataFrame()
        strategy = MovingAverageCrossoverStrategy(10, 20)
        with pytest.raises(Exception):
            signals = strategy.generate_signals(empty_data)
        
        invalid_config = SimulationConfig(initial_capital=-1000, leverage=1.0, transaction_cost=0.001, slippage=0.0005, position_sizing="fixed_fraction", position_size=0.1)
        engine = SimulationEngine(invalid_config)
    
    def test_performance_consistency(self, sample_ohlcv_data, simulation_config):
        strategy = MovingAverageCrossoverStrategy(10, 20)
        signals = strategy.generate_signals(sample_ohlcv_data)
        engine = SimulationEngine(simulation_config)
        result1 = engine.run_simulation(sample_ohlcv_data, signals)
        result2 = engine.run_simulation(sample_ohlcv_data, signals)
        pd.testing.assert_series_equal(result1.equity_curve, result2.equity_curve)
        metrics1 = PerformanceMetrics.calculate_all_metrics(result1.equity_curve, result1.trades)
        metrics2 = PerformanceMetrics.calculate_all_metrics(result2.equity_curve, result2.trades)
        
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-10, f"Inconsistent {key}: {metrics1[key]} vs {metrics2[key]}"
    
    def test_leverage_integration(self, sample_ohlcv_data):
        for leverage in [1.0, 1.5, 2.0]:
            config = SimulationConfig(initial_capital=10000.0, leverage=leverage, transaction_cost=0.001, slippage=0.0005, position_sizing="fixed_fraction", position_size=0.1)
            strategy = MovingAverageCrossoverStrategy(10, 20)
            signals = strategy.generate_signals(sample_ohlcv_data)
            engine = SimulationEngine(config)
            result = engine.run_simulation(sample_ohlcv_data, signals)
            assert not result.equity_curve.empty
            returns = result.equity_curve.pct_change().dropna()
            volatility = returns.std()
            assert 0 < volatility < 1.0
    
    def test_different_position_sizing_methods(self, sample_ohlcv_data):
        strategy = MovingAverageCrossoverStrategy(10, 20)
        signals = strategy.generate_signals(sample_ohlcv_data)
        config_fraction = SimulationConfig(initial_capital=10000.0, leverage=1.0, transaction_cost=0.001, slippage=0.0005, position_sizing="fixed_fraction", position_size=0.1)
        engine_fraction = SimulationEngine(config_fraction)
        result_fraction = engine_fraction.run_simulation(sample_ohlcv_data, signals)
        config_size = SimulationConfig(initial_capital=10000.0, leverage=1.0, transaction_cost=0.001, slippage=0.0005, position_sizing="fixed_size", position_size=1000.0)
        engine_size = SimulationEngine(config_size)
        result_size = engine_size.run_simulation(sample_ohlcv_data, signals)
        assert not result_fraction.equity_curve.empty
        assert not result_size.equity_curve.empty
        assert not result_fraction.equity_curve.equals(result_size.equity_curve)
    
    def test_metrics_calculation_integration(self, sample_ohlcv_data, simulation_config):
        strategy = MovingAverageCrossoverStrategy(10, 20)
        signals = strategy.generate_signals(sample_ohlcv_data)
        engine = SimulationEngine(simulation_config)
        result = engine.run_simulation(sample_ohlcv_data, signals)
        metrics = PerformanceMetrics.calculate_all_metrics(result.equity_curve, result.trades)
        expected_metrics = ['total_return', 'cagr', 'volatility', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'max_drawdown_duration', 'win_rate', 'profit_factor', 'avg_trade_return', 'max_consecutive_losses']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Invalid type for {metric}: {type(metrics[metric])}"
            assert not np.isnan(metrics[metric]), f"NaN value for {metric}"
            assert not np.isinf(metrics[metric]), f"Infinite value for {metric}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])