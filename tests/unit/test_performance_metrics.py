import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from src.backtester.metrics.performance import PerformanceMetrics

class TestPerformanceMetrics:
    @pytest.fixture
    def simple_returns(self):
        return pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012])
    
    @pytest.fixture
    def positive_returns(self):
        return pd.Series([0.01, 0.02, 0.015, 0.008, 0.012, 0.005])
    
    @pytest.fixture
    def negative_returns(self):
        return pd.Series([-0.01, -0.02, -0.015, -0.008, -0.012, -0.005])
    
    @pytest.fixture
    def zero_returns(self):
        return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
    
    @pytest.fixture
    def sample_equity_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        equity_values = 10000 * (1 + returns).cumprod()
        return pd.Series(equity_values, index=dates)
    
    def test_calculate_returns_basic(self, sample_equity_curve):
        returns = PerformanceMetrics.calculate_returns(sample_equity_curve)
        assert len(returns) == len(sample_equity_curve) - 1
        expected_first_return = (sample_equity_curve.iloc[1] / sample_equity_curve.iloc[0]) - 1
        assert abs(returns.iloc[0] - expected_first_return) < 1e-10
        assert not returns.isna().any()
    
    def test_calculate_returns_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        returns = PerformanceMetrics.calculate_returns(empty_series)
        assert len(returns) == 0
        single_value = pd.Series([1000.0])
        returns = PerformanceMetrics.calculate_returns(single_value)
        assert len(returns) == 0
        two_values = pd.Series([1000.0, 1100.0])
        returns = PerformanceMetrics.calculate_returns(two_values)
        assert len(returns) == 1
        assert abs(returns.iloc[0] - 0.1) < 1e-10
    
    def test_sharpe_ratio_basic_calculation(self, simple_returns):
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(simple_returns, risk_free_rate=0.0)
        assert np.isfinite(sharpe)
        mean_return = simple_returns.mean()
        std_return = simple_returns.std()
        expected_sharpe = (mean_return / std_return) * np.sqrt(252)
        assert abs(sharpe - expected_sharpe) < 1e-10
    
    def test_sharpe_ratio_with_risk_free_rate(self, simple_returns):
        risk_free_rate = 0.03  # 3% annual
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(simple_returns, risk_free_rate=risk_free_rate)
        sharpe_zero_rf = PerformanceMetrics.calculate_sharpe_ratio(simple_returns, risk_free_rate=0.0)
        
        if simple_returns.mean() > 0:
            assert sharpe < sharpe_zero_rf
    
    def test_sharpe_ratio_edge_cases(self):
        empty_returns = pd.Series(dtype=float)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(empty_returns)
        assert sharpe == 0.0
        constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(constant_returns)
        assert sharpe == np.inf
        zero_returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(zero_returns)
        assert sharpe == 0.0
    
    def test_sortino_ratio_basic_calculation(self, simple_returns):
        sortino = PerformanceMetrics.calculate_sortino_ratio(simple_returns, risk_free_rate=0.0)
        assert np.isfinite(sortino)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(simple_returns, risk_free_rate=0.0)
        assert sortino >= sharpe
    
    def test_sortino_ratio_positive_returns_only(self, positive_returns):
        sortino = PerformanceMetrics.calculate_sortino_ratio(positive_returns, risk_free_rate=0.0)
        assert sortino == np.inf
    
    def test_sortino_ratio_negative_returns_only(self, negative_returns):
        sortino = PerformanceMetrics.calculate_sortino_ratio(negative_returns, risk_free_rate=0.0)
        assert sortino < 0
        assert np.isfinite(sortino)
    
    def test_sortino_ratio_edge_cases(self):
        empty_returns = pd.Series(dtype=float)
        sortino = PerformanceMetrics.calculate_sortino_ratio(empty_returns)
        assert sortino == 0.0
        zero_returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        sortino = PerformanceMetrics.calculate_sortino_ratio(zero_returns)
        assert sortino == 0.0
    
    def test_volatility_calculation(self, simple_returns):
        volatility = PerformanceMetrics.calculate_volatility(simple_returns)
        assert volatility > 0
        expected_vol = simple_returns.std() * np.sqrt(252)
        assert abs(volatility - expected_vol) < 1e-10
    
    def test_downside_volatility_calculation(self, simple_returns):
        downside_vol = PerformanceMetrics.calculate_downside_volatility(simple_returns)
        assert downside_vol >= 0
        total_vol = PerformanceMetrics.calculate_volatility(simple_returns)
        assert downside_vol <= total_vol
    
    def test_downside_volatility_positive_returns(self, positive_returns):
        downside_vol = PerformanceMetrics.calculate_downside_volatility(positive_returns)
        assert downside_vol == 0.0
    
    def test_calculate_all_risk_metrics(self, sample_equity_curve):
        metrics = PerformanceMetrics.calculate_all_risk_metrics(sample_equity_curve)
        expected_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'volatility', 'downside_volatility',
            'total_return_annualized', 'excess_return_annualized'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert np.isfinite(metrics[metric])
    
    def test_risk_metrics_mathematical_relationships(self, sample_equity_curve):
        metrics = PerformanceMetrics.calculate_all_risk_metrics(sample_equity_curve)
        assert metrics['sortino_ratio'] >= metrics['sharpe_ratio']
        assert metrics['downside_volatility'] <= metrics['volatility']
        assert metrics['volatility'] >= 0
        risk_free_rate = 0.0
        expected_excess = metrics['total_return_annualized'] - risk_free_rate
        assert abs(metrics['excess_return_annualized'] - expected_excess) < 1e-8
    
    def test_annualization_periods(self, simple_returns):
        sharpe_daily = PerformanceMetrics.calculate_sharpe_ratio(simple_returns, periods_per_year=252)
        sharpe_monthly = PerformanceMetrics.calculate_sharpe_ratio(simple_returns, periods_per_year=12)
        assert sharpe_daily != sharpe_monthly
        expected_ratio = np.sqrt(252 / 12)
        actual_ratio = sharpe_daily / sharpe_monthly if sharpe_monthly != 0 else 0
        if sharpe_monthly != 0:
            assert abs(actual_ratio - expected_ratio) < 0.1
    
    def test_known_scenario_calculations(self):
        np.random.seed(123)
        returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015, -0.008, 0.01, 0.005])
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert np.isfinite(sharpe)
        assert np.isfinite(sortino)
        assert sortino >= sharpe
        assert returns.mean() > 0
        assert sharpe > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])