import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester.metrics.performance import PerformanceMetrics

class TestAbsoluteReturnMetrics:
    @pytest.fixture
    def simple_equity_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=505, freq='D')
        values = np.linspace(10000, 20000, 505)
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def declining_equity_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=253, freq='D')
        values = np.linspace(10000, 5000, 253)
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def volatile_equity_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=253, freq='D')
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 252)
        equity_values = [10000]
        for ret in daily_returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        return pd.Series(equity_values, index=dates)
    
    @pytest.fixture
    def short_period_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=22, freq='D')
        values = np.linspace(10000, 11000, 22)
        return pd.Series(values, index=dates)
    
    def test_calculate_total_return_basic(self, simple_equity_curve):
        total_return = PerformanceMetrics.calculate_total_return(simple_equity_curve)
        expected_return = (20000 / 10000) - 1
        assert abs(total_return - expected_return) < 1e-10
        assert abs(total_return - 1.0) < 1e-10
    
    def test_calculate_total_return_loss(self, declining_equity_curve):
        total_return = PerformanceMetrics.calculate_total_return(declining_equity_curve)
        expected_return = (5000 / 10000) - 1
        assert abs(total_return - expected_return) < 1e-10
        assert abs(total_return - (-0.5)) < 1e-10  # -50% return
    
    def test_calculate_total_return_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        assert PerformanceMetrics.calculate_total_return(empty_series) == 0.0
        single_value = pd.Series([1000.0])
        assert PerformanceMetrics.calculate_total_return(single_value) == 0.0
        zero_start = pd.Series([0.0, 1000.0])
        assert PerformanceMetrics.calculate_total_return(zero_start) == 0.0
        negative_start = pd.Series([-1000.0, 500.0])
        assert PerformanceMetrics.calculate_total_return(negative_start) == 0.0
    
    def test_calculate_cagr_basic(self, simple_equity_curve):
        cagr = PerformanceMetrics.calculate_cagr(simple_equity_curve, periods_per_year=252)
        expected_cagr = (2.0) ** (1/2) - 1
        assert abs(cagr - expected_cagr) < 1e-4  # Allow small numerical error
        assert abs(cagr - 0.4142135623730951) < 1e-10
    
    def test_calculate_cagr_loss(self, declining_equity_curve):
        cagr = PerformanceMetrics.calculate_cagr(declining_equity_curve, periods_per_year=252)
        expected_cagr = (0.5) ** (1/1) - 1
        assert abs(cagr - expected_cagr) < 1e-10
        assert abs(cagr - (-0.5)) < 1e-10
    
    def test_calculate_cagr_short_period(self, short_period_curve):
        cagr = PerformanceMetrics.calculate_cagr(short_period_curve, periods_per_year=252)
        years = 21 / 252
        expected_cagr = (1.1) ** (1/years) - 1
        assert abs(cagr - expected_cagr) < 1e-4
        assert cagr > 1.0  # More than 100% annualized
    
    def test_calculate_cagr_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        assert PerformanceMetrics.calculate_cagr(empty_series) == 0.0
        single_value = pd.Series([1000.0])
        assert PerformanceMetrics.calculate_cagr(single_value) == 0.0
        zero_start = pd.Series([0.0, 1000.0])
        assert PerformanceMetrics.calculate_cagr(zero_start) == 0.0
        complete_loss = pd.Series([1000.0, 0.0])
        assert PerformanceMetrics.calculate_cagr(complete_loss) == -1.0
        negative_end = pd.Series([1000.0, -500.0])
        assert PerformanceMetrics.calculate_cagr(negative_end) == -1.0
    
    def test_calculate_annualized_return_alias(self, simple_equity_curve):
        cagr = PerformanceMetrics.calculate_cagr(simple_equity_curve)
        annualized_return = PerformanceMetrics.calculate_annualized_return(simple_equity_curve)
        assert cagr == annualized_return
    
    def test_different_periods_per_year(self, simple_equity_curve):
        cagr_daily = PerformanceMetrics.calculate_cagr(simple_equity_curve, periods_per_year=252)
        monthly_curve = simple_equity_curve.iloc[::21]  # Approximate monthly sampling
        cagr_monthly = PerformanceMetrics.calculate_cagr(monthly_curve, periods_per_year=12)
        assert abs(cagr_daily - cagr_monthly) < 0.1  # Within 10 percentage points
    
    def test_calculate_all_absolute_return_metrics(self, simple_equity_curve):
        metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(simple_equity_curve)
        expected_metrics = ['total_return', 'cagr', 'annualized_return', 'periods_elapsed', 'years_elapsed']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        assert abs(metrics['total_return'] - 1.0) < 1e-10
        assert abs(metrics['cagr'] - metrics['annualized_return']) < 1e-10
        assert metrics['periods_elapsed'] == 504
        assert abs(metrics['years_elapsed'] - 2.0) < 1e-10
    
    def test_calculate_all_absolute_return_metrics_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(empty_series)
        
        expected_empty = {
            'total_return': 0.0,
            'cagr': 0.0,
            'annualized_return': 0.0,
            'periods_elapsed': 0,
            'years_elapsed': 0.0
        }
        
        assert metrics == expected_empty
        single_value = pd.Series([1000.0])
        metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(single_value)
        assert metrics == expected_empty
    
    def test_mathematical_relationships(self, volatile_equity_curve):
        metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(volatile_equity_curve)
        assert metrics['cagr'] == metrics['annualized_return']
        if abs(metrics['years_elapsed'] - 1.0) < 0.1:  # Approximately 1 year
            assert abs(metrics['cagr'] - metrics['total_return']) < 0.1
        
        assert metrics['periods_elapsed'] >= 0
        assert metrics['years_elapsed'] >= 0
    
    def test_known_scenario_calculations(self):
        dates = pd.date_range(start='2024-01-01', periods=253, freq='D')
        equity_curve = pd.Series([10000.0] + [12000.0] * 252, index=dates)
        total_return = PerformanceMetrics.calculate_total_return(equity_curve)
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        assert abs(total_return - 0.2) < 1e-10
        assert abs(cagr - 0.2) < 1e-10
        dates = pd.date_range(start='2024-01-01', periods=505, freq='D')
        equity_curve = pd.Series([10000.0] + [14400.0] * 504, index=dates)
        total_return = PerformanceMetrics.calculate_total_return(equity_curve)
        cagr = PerformanceMetrics.calculate_cagr(equity_curve)
        assert abs(total_return - 0.44) < 1e-10
        assert abs(cagr - 0.2) < 1e-4
    
    def test_integration_with_existing_metrics(self, volatile_equity_curve):
        absolute_metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(volatile_equity_curve)
        risk_metrics = PerformanceMetrics.calculate_all_risk_metrics(volatile_equity_curve)
        assert len(absolute_metrics) > 0
        assert len(risk_metrics) > 0
        absolute_keys = set(absolute_metrics.keys())
        risk_keys = set(risk_metrics.keys())
        assert len(absolute_keys) > 0
        assert len(risk_keys) > 0
    
    def test_performance_with_large_datasets(self):
        dates = pd.date_range(start='2014-01-01', periods=2521, freq='D')  # ~10 years
        np.random.seed(123)
        daily_returns = np.random.normal(0.0008, 0.015, 2520)  # Slightly positive with volatility
        equity_values = [10000.0]
        for ret in daily_returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        equity_curve = pd.Series(equity_values, index=dates)
        import time
        start_time = time.time()
        metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(equity_curve)
        end_time = time.time()
        assert (end_time - start_time) < 1.0
        assert metrics['years_elapsed'] > 9.5
        assert metrics['periods_elapsed'] > 2500
        assert -1.0 <= metrics['total_return'] <= 10.0
        assert -1.0 <= metrics['cagr'] <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])