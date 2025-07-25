import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester.metrics.performance import PerformanceMetrics

class TestDrawdownMetrics:
    @pytest.fixture
    def simple_drawdown_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=101, freq='D')
        values = [10000] * 25 + [15000] * 25 + list(np.linspace(15000, 9000, 25)) + list(np.linspace(9000, 16000, 26))
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def multiple_drawdowns_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=151, freq='D')
        values = []
        values.extend([10000] * 10)
        values.extend(list(np.linspace(10000, 12000, 20)))
        values.extend(list(np.linspace(12000, 8000, 30)))
        values.extend(list(np.linspace(8000, 13000, 20)))
        values.extend([13000] * 10)
        values.extend(list(np.linspace(13000, 11000, 20)))
        values.extend(list(np.linspace(11000, 14000, 15)))
        values.extend([14000] * 6)  # 6 values
        values.extend(list(np.linspace(14000, 10500, 10)))
        values.extend(list(np.linspace(10500, 15000, 10)))
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def no_drawdown_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=51, freq='D')
        values = np.linspace(10000, 20000, 51)
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def continuous_decline_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=51, freq='D')
        values = np.linspace(10000, 5000, 51)
        return pd.Series(values, index=dates)
    
    @pytest.fixture
    def volatile_recovery_curve(self):
        dates = pd.date_range(start='2024-01-01', periods=301, freq='D')
        values = []
        values.extend(list(np.linspace(10000, 15000, 50)))
        values.extend(list(np.linspace(15000, 7500, 25)))  # 50% drawdown
        recovery_base = np.linspace(7500, 15000, 226)
        np.random.seed(42)
        volatility = np.random.normal(0, 200, 226)
        recovery_values = recovery_base + volatility
        recovery_values = np.maximum(recovery_values, 7500)
        values.extend(recovery_values.tolist())
        return pd.Series(values, index=dates)
    
    def test_calculate_max_drawdown_basic(self, simple_drawdown_curve):
        max_dd = PerformanceMetrics.calculate_max_drawdown(simple_drawdown_curve)
        expected_dd = (15000 - 9000) / 15000
        assert abs(max_dd - expected_dd) < 1e-10
        assert abs(max_dd - 0.4) < 1e-10
    
    def test_calculate_max_drawdown_multiple(self, multiple_drawdowns_curve):
        max_dd = PerformanceMetrics.calculate_max_drawdown(multiple_drawdowns_curve)
        expected_dd = (12000 - 8000) / 12000
        assert abs(max_dd - expected_dd) < 1e-6
        assert abs(max_dd - (1/3)) < 1e-6  # 33.33%
    
    def test_calculate_max_drawdown_no_drawdown(self, no_drawdown_curve):
        max_dd = PerformanceMetrics.calculate_max_drawdown(no_drawdown_curve)
        assert max_dd == 0.0
    
    def test_calculate_max_drawdown_continuous_decline(self, continuous_decline_curve):
        max_dd = PerformanceMetrics.calculate_max_drawdown(continuous_decline_curve)
        expected_dd = (10000 - 5000) / 10000
        assert abs(max_dd - expected_dd) < 1e-10
        assert abs(max_dd - 0.5) < 1e-10
    
    def test_calculate_max_drawdown_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        assert PerformanceMetrics.calculate_max_drawdown(empty_series) == 0.0
        single_value = pd.Series([1000.0])
        assert PerformanceMetrics.calculate_max_drawdown(single_value) == 0.0
        identical_values = pd.Series([1000.0, 1000.0])
        assert PerformanceMetrics.calculate_max_drawdown(identical_values) == 0.0
        gain_values = pd.Series([1000.0, 1200.0])
        assert PerformanceMetrics.calculate_max_drawdown(gain_values) == 0.0
    
    def test_calculate_max_drawdown_duration_basic(self, simple_drawdown_curve):
        max_duration = PerformanceMetrics.calculate_max_drawdown_duration(simple_drawdown_curve)
        assert 40 <= max_duration <= 55
    
    def test_calculate_max_drawdown_duration_multiple(self, multiple_drawdowns_curve):
        max_duration = PerformanceMetrics.calculate_max_drawdown_duration(multiple_drawdowns_curve)
        assert max_duration >= 45
        assert max_duration <= 55
    
    def test_calculate_max_drawdown_duration_no_drawdown(self, no_drawdown_curve):
        max_duration = PerformanceMetrics.calculate_max_drawdown_duration(no_drawdown_curve)
        assert max_duration == 0
    
    def test_calculate_max_drawdown_duration_long_recovery(self, volatile_recovery_curve):
        max_duration = PerformanceMetrics.calculate_max_drawdown_duration(volatile_recovery_curve)
        assert max_duration >= 220
        assert max_duration <= 270
    
    def test_calculate_max_drawdown_duration_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        assert PerformanceMetrics.calculate_max_drawdown_duration(empty_series) == 0
        single_value = pd.Series([1000.0])
        assert PerformanceMetrics.calculate_max_drawdown_duration(single_value) == 0
        decline = pd.Series([1000.0, 900.0, 800.0, 700.0])
        duration = PerformanceMetrics.calculate_max_drawdown_duration(decline)
        assert duration == 3
    
    def test_calculate_drawdown_series_basic(self, simple_drawdown_curve):
        dd_series = PerformanceMetrics.calculate_drawdown_series(simple_drawdown_curve)
        assert len(dd_series) == len(simple_drawdown_curve)
        assert dd_series.index.equals(simple_drawdown_curve.index)
        assert dd_series.iloc[49] == 0.0
        trough_idx = 74
        assert dd_series.iloc[trough_idx] > 0.35
        assert dd_series.iloc[-1] == 0.0  # At new high
    
    def test_calculate_drawdown_series_no_drawdown(self, no_drawdown_curve):
        dd_series = PerformanceMetrics.calculate_drawdown_series(no_drawdown_curve)
        assert (dd_series == 0.0).all()
    
    def test_calculate_drawdown_series_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        dd_series = PerformanceMetrics.calculate_drawdown_series(empty_series)
        assert len(dd_series) == 0
        single_value = pd.Series([1000.0], index=[pd.Timestamp('2024-01-01')])
        dd_series = PerformanceMetrics.calculate_drawdown_series(single_value)
        assert len(dd_series) == 1
        assert dd_series.iloc[0] == 0.0
    
    def test_calculate_average_drawdown_basic(self, multiple_drawdowns_curve):
        avg_dd = PerformanceMetrics.calculate_average_drawdown(multiple_drawdowns_curve)
        assert avg_dd > 0.0
        max_dd = PerformanceMetrics.calculate_max_drawdown(multiple_drawdowns_curve)
        assert avg_dd < max_dd
        assert avg_dd < 0.2
    
    def test_calculate_average_drawdown_no_drawdown(self, no_drawdown_curve):
        avg_dd = PerformanceMetrics.calculate_average_drawdown(no_drawdown_curve)
        assert avg_dd == 0.0
    
    def test_calculate_all_drawdown_metrics_comprehensive(self, multiple_drawdowns_curve):
        metrics = PerformanceMetrics.calculate_all_drawdown_metrics(multiple_drawdowns_curve)
        expected_metrics = [
            'max_drawdown', 'max_drawdown_pct', 'max_drawdown_duration',
            'average_drawdown', 'drawdown_periods', 'recovery_periods'
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        assert metrics['max_drawdown'] > 0.0
        assert metrics['max_drawdown_pct'] == metrics['max_drawdown'] * 100
        assert metrics['max_drawdown_duration'] > 0
        assert metrics['average_drawdown'] > 0.0
        assert metrics['drawdown_periods'] > 0
        assert metrics['recovery_periods'] >= 0
        assert metrics['max_drawdown'] > metrics['average_drawdown']
    
    def test_calculate_all_drawdown_metrics_no_drawdown(self, no_drawdown_curve):
        metrics = PerformanceMetrics.calculate_all_drawdown_metrics(no_drawdown_curve)
        
        expected_zeros = {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration': 0,
            'average_drawdown': 0.0,
            'drawdown_periods': 0,
            'recovery_periods': 0
        }
        assert metrics == expected_zeros
    
    def test_calculate_all_drawdown_metrics_edge_cases(self):
        empty_series = pd.Series(dtype=float)
        metrics = PerformanceMetrics.calculate_all_drawdown_metrics(empty_series)
        
        expected_empty = {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration': 0,
            'average_drawdown': 0.0,
            'drawdown_periods': 0,
            'recovery_periods': 0
        }
        
        assert metrics == expected_empty
        single_value = pd.Series([1000.0])
        metrics = PerformanceMetrics.calculate_all_drawdown_metrics(single_value)
        assert metrics == expected_empty
    
    def test_mathematical_relationships(self, simple_drawdown_curve):
        metrics = PerformanceMetrics.calculate_all_drawdown_metrics(simple_drawdown_curve)
        assert metrics['max_drawdown'] >= metrics['average_drawdown']
        assert abs(metrics['max_drawdown_pct'] - metrics['max_drawdown'] * 100) < 1e-10
        assert metrics['max_drawdown_duration'] >= 0
        assert metrics['drawdown_periods'] >= 0
        assert metrics['recovery_periods'] >= 0
    
    def test_known_scenario_calculations(self):
        dates = pd.date_range(start='2024-01-01', periods=4, freq='D')
        equity_curve = pd.Series([10000, 10000, 5000, 10000], index=dates)
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        max_duration = PerformanceMetrics.calculate_max_drawdown_duration(equity_curve)
        assert abs(max_dd - 0.5) < 1e-10  # Exactly 50%
        assert max_duration == 1
        equity_curve = pd.Series([10000, 8000, 6000, 4000], index=dates)
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        max_duration = PerformanceMetrics.calculate_max_drawdown_duration(equity_curve)
        assert abs(max_dd - 0.6) < 1e-10
        assert max_duration == 3
    
    def test_integration_with_existing_metrics(self, simple_drawdown_curve):
        drawdown_metrics = PerformanceMetrics.calculate_all_drawdown_metrics(simple_drawdown_curve)
        absolute_metrics = PerformanceMetrics.calculate_all_absolute_return_metrics(simple_drawdown_curve)
        risk_metrics = PerformanceMetrics.calculate_all_risk_metrics(simple_drawdown_curve)
        assert len(drawdown_metrics) > 0
        assert len(absolute_metrics) > 0
        assert len(risk_metrics) > 0
        drawdown_keys = set(drawdown_metrics.keys())
        absolute_keys = set(absolute_metrics.keys())
        risk_keys = set(risk_metrics.keys())
        assert len(drawdown_keys & absolute_keys) == 0
        assert len(drawdown_keys & risk_keys) == 0
    
    def test_performance_with_large_datasets(self):
        dates = pd.date_range(start='2019-01-01', periods=1261, freq='D')
        np.random.seed(123)
        
        equity_values = [10000.0]
        for i in range(1260):
            if i % 200 == 0 and i > 0:
                change = np.random.uniform(-0.15, -0.05)
            else:
                change = np.random.normal(0.0005, 0.015)
            
            new_value = equity_values[-1] * (1 + change)
            equity_values.append(max(new_value, 1000))
        
        equity_curve = pd.Series(equity_values, index=dates)
        import time
        start_time = time.time()
        metrics = PerformanceMetrics.calculate_all_drawdown_metrics(equity_curve)
        end_time = time.time()
        
        assert (end_time - start_time) < 1.0
        assert 0.0 <= metrics['max_drawdown'] <= 1.0
        assert metrics['max_drawdown_duration'] >= 0
        assert metrics['average_drawdown'] >= 0.0
        assert metrics['drawdown_periods'] >= 0
        assert metrics['recovery_periods'] >= 0
    
    def test_drawdown_series_for_visualization(self, simple_drawdown_curve):
        dd_series = PerformanceMetrics.calculate_drawdown_series(simple_drawdown_curve)
        assert (dd_series >= 0).all()
        assert dd_series.index.equals(simple_drawdown_curve.index)
        assert (dd_series <= 1.0).all()
        assert dd_series.std() > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])