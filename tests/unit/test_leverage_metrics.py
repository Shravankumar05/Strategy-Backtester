import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester.metrics.performance import PerformanceMetrics

class TestLeverageMetrics:
    @pytest.fixture
    def no_leverage_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=101, freq='D')
        equity_values = np.linspace(10000, 12000, 101)
        position_values = equity_values.copy()
        cash_values = np.zeros(101)
        position_sizes = position_values / 100
        
        return pd.DataFrame({
            'equity': equity_values,
            'cash': cash_values,
            'position': position_sizes,
            'position_value': position_values
        }, index=dates)
    
    @pytest.fixture
    def moderate_leverage_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=101, freq='D')
        equity_values = np.linspace(10000, 12000, 101)
        position_values = equity_values * 1.5
        cash_values = equity_values - (position_values / 1.5)
        position_sizes = position_values / 100
        
        return pd.DataFrame({
            'equity': equity_values,
            'cash': cash_values,
            'position': position_sizes,
            'position_value': position_values
        }, index=dates)
    
    @pytest.fixture
    def high_leverage_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=101, freq='D')
        equity_values = np.linspace(10000, 12000, 101)
        leverage_ratios = []
        for i in range(101):
            if i < 50:
                leverage = 1.0 + (2.0 * i / 50)
            else:
                leverage = 3.0 - (1.0 * (i - 50) / 50)
            leverage_ratios.append(leverage)
        
        position_values = equity_values * np.array(leverage_ratios)
        cash_values = equity_values - (position_values / np.array(leverage_ratios))
        position_sizes = position_values / 100
        
        return pd.DataFrame({
            'equity': equity_values,
            'cash': cash_values,
            'position': position_sizes,
            'position_value': position_values
        }, index=dates)
    
    @pytest.fixture
    def no_position_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=101, freq='D')
        equity_values = np.full(101, 10000)
        
        position_values = []
        for i in range(101):
            if i < 25 or i >= 75:
                position_values.append(0.0)
            else:
                position_values.append(15000.0)
        
        cash_values = equity_values - np.array(position_values) / 1.5
        position_sizes = np.array(position_values) / 100
        
        return pd.DataFrame({
            'equity': equity_values,
            'cash': cash_values,
            'position': position_sizes,
            'position_value': position_values
        }, index=dates)
    
    @pytest.fixture
    def volatile_leverage_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=201, freq='D')
        np.random.seed(42)
        base_equity = np.linspace(10000, 15000, 201)
        equity_volatility = np.random.normal(0, 200, 201)
        equity_values = base_equity + equity_volatility
        equity_values = np.maximum(equity_values, 5000)
        leverage_ratios = np.random.uniform(0.5, 2.5, 201)
        position_values = equity_values * leverage_ratios
        cash_values = equity_values - (position_values / leverage_ratios)
        position_sizes = position_values / 100
        return pd.DataFrame({
            'equity': equity_values,
            'cash': cash_values,
            'position': position_sizes,
            'position_value': position_values
        }, index=dates)
    
    def test_calculate_average_leverage_no_leverage(self, no_leverage_curve):
        avg_leverage = PerformanceMetrics.calculate_average_leverage(no_leverage_curve, 10000)
        assert abs(avg_leverage - 1.0) < 1e-10
    
    def test_calculate_average_leverage_moderate(self, moderate_leverage_curve):
        avg_leverage = PerformanceMetrics.calculate_average_leverage(moderate_leverage_curve, 10000)
        assert abs(avg_leverage - 1.5) < 1e-10
    
    def test_calculate_average_leverage_high_variable(self, high_leverage_curve):
        avg_leverage = PerformanceMetrics.calculate_average_leverage(high_leverage_curve, 10000)
        assert 1.8 <= avg_leverage <= 2.3
    
    def test_calculate_average_leverage_with_no_position(self, no_position_curve):
        avg_leverage = PerformanceMetrics.calculate_average_leverage(no_position_curve, 10000)
        expected_avg = (50 * 1.5) / 101
        assert abs(avg_leverage - expected_avg) < 0.1
    
    def test_calculate_maximum_leverage_no_leverage(self, no_leverage_curve):
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(no_leverage_curve, 10000)
        assert abs(max_leverage - 1.0) < 1e-10
    
    def test_calculate_maximum_leverage_high_variable(self, high_leverage_curve):
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(high_leverage_curve, 10000)
        assert abs(max_leverage - 3.0) < 0.1
    
    def test_calculate_maximum_leverage_with_no_position(self, no_position_curve):
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(no_position_curve, 10000)
        assert abs(max_leverage - 1.5) < 1e-10
    
    def test_calculate_leverage_utilization_basic(self, moderate_leverage_curve):
        utilization = PerformanceMetrics.calculate_leverage_utilization(moderate_leverage_curve, 2.0)
        assert abs(utilization - 0.75) < 1e-10
    
    def test_calculate_leverage_utilization_over_limit(self, high_leverage_curve):
        utilization = PerformanceMetrics.calculate_leverage_utilization(high_leverage_curve, 1.5)
        assert utilization == 1.0
    
    def test_calculate_leverage_efficiency_basic(self, moderate_leverage_curve):
        returns = PerformanceMetrics.calculate_returns(moderate_leverage_curve['equity'])
        efficiency = PerformanceMetrics.calculate_leverage_efficiency(moderate_leverage_curve, returns)
        assert efficiency > 0.0
        assert efficiency < 1.0
    
    def test_calculate_leverage_efficiency_no_leverage(self, no_leverage_curve):
        returns = PerformanceMetrics.calculate_returns(no_leverage_curve['equity'])
        efficiency = PerformanceMetrics.calculate_leverage_efficiency(no_leverage_curve, returns)
        expected_efficiency = 0.2 / 1.0
        assert abs(efficiency - expected_efficiency) < 0.01
    
    def test_calculate_all_leverage_metrics_comprehensive(self, high_leverage_curve):
        metrics = PerformanceMetrics.calculate_all_leverage_metrics(
            high_leverage_curve, 10000, max_allowed_leverage=2.5
        )
        
        expected_metrics = [
            'average_leverage', 'maximum_leverage', 'leverage_utilization',
            'leverage_efficiency', 'periods_with_leverage', 'periods_without_position'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        assert metrics['maximum_leverage'] >= metrics['average_leverage']
        assert 0.0 <= metrics['leverage_utilization'] <= 1.0
        assert metrics['periods_with_leverage'] >= 0
        assert metrics['periods_without_position'] >= 0
        assert metrics['periods_with_leverage'] >= 95  # Allow some tolerance
        assert metrics['periods_without_position'] == 0
    
    def test_calculate_all_leverage_metrics_with_no_position(self, no_position_curve):
        metrics = PerformanceMetrics.calculate_all_leverage_metrics(no_position_curve, 10000)
        assert metrics['periods_without_position'] == 51
        assert metrics['periods_with_leverage'] >= 0
        assert metrics['average_leverage'] < 1.0
    
    def test_calculate_leverage_metrics_edge_cases(self):
        empty_df = pd.DataFrame()
        metrics = PerformanceMetrics.calculate_all_leverage_metrics(empty_df, 10000)
        
        expected_empty = {
            'average_leverage': 0.0,
            'maximum_leverage': 0.0,
            'leverage_utilization': 0.0,
            'leverage_efficiency': 0.0,
            'periods_with_leverage': 0,
            'periods_without_position': 0
        }
        
        assert metrics == expected_empty
        incomplete_df = pd.DataFrame({
            'equity': [10000, 11000],
            'cash': [5000, 5500]
        })
        
        metrics = PerformanceMetrics.calculate_all_leverage_metrics(incomplete_df, 10000)
        assert metrics['average_leverage'] == 0.0
        assert metrics['maximum_leverage'] == 0.0
    
    def test_leverage_utilization_edge_cases(self):
        test_df = pd.DataFrame({
            'equity': [10000],
            'position_value': [15000]
        })
        
        assert PerformanceMetrics.calculate_leverage_utilization(test_df, 0.0) == 0.0
        assert PerformanceMetrics.calculate_leverage_utilization(test_df, -1.0) == 0.0
    
    def test_leverage_efficiency_edge_cases(self):
        test_df = pd.DataFrame({
            'equity': [10000],
            'position_value': [15000]
        })
        empty_returns = pd.Series(dtype=float)
        
        efficiency = PerformanceMetrics.calculate_leverage_efficiency(test_df, empty_returns)
        assert efficiency == 0.0
        zero_leverage_df = pd.DataFrame({
            'equity': [10000, 11000],
            'position_value': [0, 0]
        })
        returns = pd.Series([0.1])
        
        efficiency = PerformanceMetrics.calculate_leverage_efficiency(zero_leverage_df, returns)
        assert efficiency == 0.0
    
    def test_mathematical_relationships(self, volatile_leverage_curve):
        metrics = PerformanceMetrics.calculate_all_leverage_metrics(volatile_leverage_curve, 10000)
        assert metrics['maximum_leverage'] >= metrics['average_leverage']
        assert 0.0 <= metrics['leverage_utilization'] <= 1.0
        total_periods = len(volatile_leverage_curve)
        assert metrics['periods_with_leverage'] >= 0
        assert metrics['periods_without_position'] >= 0
        assert metrics['periods_with_leverage'] + metrics['periods_without_position'] <= total_periods
        assert not np.isnan(metrics['leverage_efficiency'])
        assert np.isfinite(metrics['leverage_efficiency'])
    
    def test_known_scenario_calculations(self):
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        constant_2x_df = pd.DataFrame({
            'equity': [10000, 10000, 10000, 10000, 10000],
            'position_value': [20000, 20000, 20000, 20000, 20000],
            'cash': [0, 0, 0, 0, 0],
            'position': [200, 200, 200, 200, 200]
        }, index=dates)
        
        avg_leverage = PerformanceMetrics.calculate_average_leverage(constant_2x_df, 10000)
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(constant_2x_df, 10000)
        assert abs(avg_leverage - 2.0) < 1e-10
        assert abs(max_leverage - 2.0) < 1e-10
        no_position_df = pd.DataFrame({
            'equity': [10000, 10000, 10000],
            'position_value': [0, 0, 0],
            'cash': [10000, 10000, 10000],
            'position': [0, 0, 0]
        }, index=dates[:3])
        
        avg_leverage = PerformanceMetrics.calculate_average_leverage(no_position_df, 10000)
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(no_position_df, 10000)
        assert avg_leverage == 0.0
        assert max_leverage == 0.0
    
    def test_integration_with_existing_metrics(self, moderate_leverage_curve):
        leverage_metrics = PerformanceMetrics.calculate_all_leverage_metrics(moderate_leverage_curve, 10000)
        absolute_metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(moderate_leverage_curve['equity'])
        assert len(leverage_metrics) > 0
        assert len(absolute_metrics) > 0
        leverage_keys = set(leverage_metrics.keys())
        absolute_keys = set(absolute_metrics.keys())
        assert len(leverage_keys & absolute_keys) == 0
    
    def test_performance_with_large_datasets(self):
        dates = pd.date_range(start='2022-01-01', periods=731, freq='D')  # ~2 years
        np.random.seed(123)
        equity_values = np.linspace(10000, 15000, 731)  # Growing equity
        leverage_ratios = np.random.uniform(0.8, 2.2, 731)  # Variable leverage
        position_values = equity_values * leverage_ratios
        
        large_df = pd.DataFrame({
            'equity': equity_values,
            'position_value': position_values,
            'cash': equity_values - (position_values / leverage_ratios),
            'position': position_values / 100
        }, index=dates)
        
        import time
        start_time = time.time()
        metrics = PerformanceMetrics.calculate_all_leverage_metrics(large_df, 10000, max_allowed_leverage=2.0)
        end_time = time.time()
        assert (end_time - start_time) < 1.0
        assert 0.0 <= metrics['average_leverage'] <= 3.0
        assert metrics['maximum_leverage'] >= metrics['average_leverage']
        assert 0.0 <= metrics['leverage_utilization'] <= 1.0
        assert metrics['periods_with_leverage'] >= 0
        assert metrics['periods_without_position'] >= 0
    
    def test_leverage_metrics_with_negative_equity(self):
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        negative_equity_df = pd.DataFrame({
            'equity': [10000, -5000, 8000],
            'position_value': [15000, 10000, 12000],
            'cash': [5000, -15000, 4000],
            'position': [150, 100, 120]
        }, index=dates)
        
        avg_leverage = PerformanceMetrics.calculate_average_leverage(negative_equity_df, 10000)
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(negative_equity_df, 10000)
        assert avg_leverage > 0.0
        assert max_leverage > 0.0
        assert np.isfinite(avg_leverage)
        assert np.isfinite(max_leverage)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])