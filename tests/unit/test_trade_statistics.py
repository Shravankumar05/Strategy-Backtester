import pytest
from datetime import datetime, timedelta
from src.backtester.metrics.performance import PerformanceMetrics

# Mock Trade class for testing
class MockTrade:
    def __init__(self, timestamp, trade_type, price, size, value, commission, slippage, pnl=None):
        self.timestamp = timestamp
        self.type = trade_type
        self.price = price
        self.size = size
        self.value = value
        self.commission = commission
        self.slippage = slippage
        self.pnl = pnl

class TestTradeStatistics:
    @pytest.fixture
    def profitable_trades(self):
        base_date = datetime(2024, 1, 1)
        return [
            MockTrade(base_date, "buy", 100, 10, 1000, 5, 2),
            MockTrade(base_date + timedelta(days=1), "sell", 110, 10, 1100, 5, 2, pnl=93),  # Profit after costs
            MockTrade(base_date + timedelta(days=2), "buy", 105, 10, 1050, 5, 2),
            MockTrade(base_date + timedelta(days=3), "sell", 120, 10, 1200, 5, 2, pnl=143),  # Profit after costs
            MockTrade(base_date + timedelta(days=4), "buy", 115, 10, 1150, 5, 2),
            MockTrade(base_date + timedelta(days=5), "sell", 125, 10, 1250, 5, 2, pnl=93),   # Profit after costs
        ]
    
    @pytest.fixture
    def mixed_trades(self):
        base_date = datetime(2024, 1, 1)
        return [
            MockTrade(base_date, "buy", 100, 10, 1000, 5, 2),
            MockTrade(base_date + timedelta(days=1), "sell", 110, 10, 1100, 5, 2, pnl=93),   # Win
            MockTrade(base_date + timedelta(days=2), "buy", 105, 10, 1050, 5, 2),
            MockTrade(base_date + timedelta(days=3), "sell", 95, 10, 950, 5, 2, pnl=-107),   # Loss
            MockTrade(base_date + timedelta(days=4), "buy", 110, 10, 1100, 5, 2),
            MockTrade(base_date + timedelta(days=5), "sell", 120, 10, 1200, 5, 2, pnl=93),   # Win
            MockTrade(base_date + timedelta(days=6), "buy", 115, 10, 1150, 5, 2),
            MockTrade(base_date + timedelta(days=7), "sell", 105, 10, 1050, 5, 2, pnl=-107), # Loss
        ]
    
    @pytest.fixture
    def losing_trades(self):
        base_date = datetime(2024, 1, 1)
        return [
            MockTrade(base_date, "buy", 100, 10, 1000, 5, 2),
            MockTrade(base_date + timedelta(days=1), "sell", 90, 10, 900, 5, 2, pnl=-107),
            MockTrade(base_date + timedelta(days=2), "buy", 105, 10, 1050, 5, 2),
            MockTrade(base_date + timedelta(days=3), "sell", 95, 10, 950, 5, 2, pnl=-107),
        ]
    
    @pytest.fixture
    def consecutive_pattern_trades(self):
        base_date = datetime(2024, 1, 1)
        return [
            # 3 consecutive wins
            MockTrade(base_date, "sell", 110, 10, 1100, 5, 2, pnl=100),     # Win 1
            MockTrade(base_date + timedelta(days=1), "sell", 120, 10, 1200, 5, 2, pnl=150),  # Win 2
            MockTrade(base_date + timedelta(days=2), "sell", 115, 10, 1150, 5, 2, pnl=75),   # Win 3
            # 2 consecutive losses
            MockTrade(base_date + timedelta(days=3), "sell", 95, 10, 950, 5, 2, pnl=-100),   # Loss 1
            MockTrade(base_date + timedelta(days=4), "sell", 90, 10, 900, 5, 2, pnl=-150),   # Loss 2
            # 1 win
            MockTrade(base_date + timedelta(days=5), "sell", 125, 10, 1250, 5, 2, pnl=200),  # Win 4
            # 4 consecutive losses
            MockTrade(base_date + timedelta(days=6), "sell", 85, 10, 850, 5, 2, pnl=-200),   # Loss 3
            MockTrade(base_date + timedelta(days=7), "sell", 80, 10, 800, 5, 2, pnl=-250),   # Loss 4
            MockTrade(base_date + timedelta(days=8), "sell", 75, 10, 750, 5, 2, pnl=-300),   # Loss 5
            MockTrade(base_date + timedelta(days=9), "sell", 70, 10, 700, 5, 2, pnl=-350),   # Loss 6
        ]
    
    def test_calculate_win_rate_all_wins(self, profitable_trades):
        win_rate = PerformanceMetrics.calculate_win_rate(profitable_trades)
        assert win_rate == 1.0  # 100% win rate
    
    def test_calculate_win_rate_mixed(self, mixed_trades):
        win_rate = PerformanceMetrics.calculate_win_rate(mixed_trades)
        assert abs(win_rate - 0.5) < 1e-10
    
    def test_calculate_win_rate_all_losses(self, losing_trades):
        win_rate = PerformanceMetrics.calculate_win_rate(losing_trades)
        assert win_rate == 0.0  # 0% win rate
    
    def test_calculate_win_rate_edge_cases(self):
        assert PerformanceMetrics.calculate_win_rate([]) == 0.0
        buy_only_trades = [
            MockTrade(datetime(2024, 1, 1), "buy", 100, 10, 1000, 5, 2),
            MockTrade(datetime(2024, 1, 2), "buy", 105, 10, 1050, 5, 2),
        ]
        assert PerformanceMetrics.calculate_win_rate(buy_only_trades) == 0.0
        single_win = [MockTrade(datetime(2024, 1, 1), "sell", 110, 10, 1100, 5, 2, pnl=100)]
        assert PerformanceMetrics.calculate_win_rate(single_win) == 1.0
        single_loss = [MockTrade(datetime(2024, 1, 1), "sell", 90, 10, 900, 5, 2, pnl=-100)]
        assert PerformanceMetrics.calculate_win_rate(single_loss) == 0.0
    
    def test_calculate_profit_factor_profitable(self, profitable_trades):
        profit_factor = PerformanceMetrics.calculate_profit_factor(profitable_trades)
        assert profit_factor == float('inf')
    
    def test_calculate_profit_factor_mixed(self, mixed_trades):
        profit_factor = PerformanceMetrics.calculate_profit_factor(mixed_trades)
        expected_pf = 186 / 214
        assert abs(profit_factor - expected_pf) < 1e-6
    
    def test_calculate_profit_factor_losing(self, losing_trades):
        profit_factor = PerformanceMetrics.calculate_profit_factor(losing_trades)
        assert profit_factor == 0.0  # No profits
    
    def test_calculate_profit_factor_edge_cases(self):
        assert PerformanceMetrics.calculate_profit_factor([]) == 0.0
        buy_only_trades = [MockTrade(datetime(2024, 1, 1), "buy", 100, 10, 1000, 5, 2)]
        assert PerformanceMetrics.calculate_profit_factor(buy_only_trades) == 0.0
        breakeven_trades = [
            MockTrade(datetime(2024, 1, 1), "sell", 100, 10, 1000, 5, 2, pnl=0),
            MockTrade(datetime(2024, 1, 2), "sell", 100, 10, 1000, 5, 2, pnl=0),
        ]
        assert PerformanceMetrics.calculate_profit_factor(breakeven_trades) == 0.0
    
    def test_calculate_trade_count_basic(self, mixed_trades):
        count = PerformanceMetrics.calculate_trade_count(mixed_trades)
        assert count == 8
    
    def test_calculate_trade_count_edge_cases(self):
        assert PerformanceMetrics.calculate_trade_count([]) == 0
        assert PerformanceMetrics.calculate_trade_count(None) == 0
        single_trade = [MockTrade(datetime(2024, 1, 1), "buy", 100, 10, 1000, 5, 2)]
        assert PerformanceMetrics.calculate_trade_count(single_trade) == 1
    
    def test_calculate_average_trade_pnl_mixed(self, mixed_trades):
        avg_pnl = PerformanceMetrics.calculate_average_trade_pnl(mixed_trades)
        expected_avg = (93 - 107 + 93 - 107) / 4
        assert abs(avg_pnl - expected_avg) < 1e-10
    
    def test_calculate_average_trade_pnl_edge_cases(self):
        assert PerformanceMetrics.calculate_average_trade_pnl([]) == 0.0
        buy_only_trades = [MockTrade(datetime(2024, 1, 1), "buy", 100, 10, 1000, 5, 2)]
        assert PerformanceMetrics.calculate_average_trade_pnl(buy_only_trades) == 0.0
    
    def test_calculate_largest_win_and_loss(self, mixed_trades):
        largest_win = PerformanceMetrics.calculate_largest_win(mixed_trades)
        largest_loss = PerformanceMetrics.calculate_largest_loss(mixed_trades)
        assert largest_win == 93.0
        assert largest_loss == 107.0
    
    def test_calculate_largest_win_loss_edge_cases(self):
        assert PerformanceMetrics.calculate_largest_win([]) == 0.0
        assert PerformanceMetrics.calculate_largest_loss([]) == 0.0
        win_only = [MockTrade(datetime(2024, 1, 1), "sell", 110, 10, 1100, 5, 2, pnl=100)]
        assert PerformanceMetrics.calculate_largest_win(win_only) == 100.0
        assert PerformanceMetrics.calculate_largest_loss(win_only) == 0.0
        loss_only = [MockTrade(datetime(2024, 1, 1), "sell", 90, 10, 900, 5, 2, pnl=-150)]
        assert PerformanceMetrics.calculate_largest_win(loss_only) == 0.0
        assert PerformanceMetrics.calculate_largest_loss(loss_only) == 150.0
    
    def test_calculate_consecutive_wins_losses(self, consecutive_pattern_trades):
        consecutive_wins = PerformanceMetrics.calculate_consecutive_wins(consecutive_pattern_trades)
        consecutive_losses = PerformanceMetrics.calculate_consecutive_losses(consecutive_pattern_trades)
        assert consecutive_wins == 3
        assert consecutive_losses == 4
    
    def test_calculate_consecutive_edge_cases(self):
        assert PerformanceMetrics.calculate_consecutive_wins([]) == 0
        assert PerformanceMetrics.calculate_consecutive_losses([]) == 0
        single_win = [MockTrade(datetime(2024, 1, 1), "sell", 110, 10, 1100, 5, 2, pnl=100)]
        assert PerformanceMetrics.calculate_consecutive_wins(single_win) == 1
        assert PerformanceMetrics.calculate_consecutive_losses(single_win) == 0
        single_loss = [MockTrade(datetime(2024, 1, 1), "sell", 90, 10, 900, 5, 2, pnl=-100)]
        assert PerformanceMetrics.calculate_consecutive_wins(single_loss) == 0
        assert PerformanceMetrics.calculate_consecutive_losses(single_loss) == 1
        alternating = [
            MockTrade(datetime(2024, 1, 1), "sell", 110, 10, 1100, 5, 2, pnl=100),   # Win
            MockTrade(datetime(2024, 1, 2), "sell", 90, 10, 900, 5, 2, pnl=-100),    # Loss
            MockTrade(datetime(2024, 1, 3), "sell", 110, 10, 1100, 5, 2, pnl=100),   # Win
            MockTrade(datetime(2024, 1, 4), "sell", 90, 10, 900, 5, 2, pnl=-100),    # Loss
        ]
        assert PerformanceMetrics.calculate_consecutive_wins(alternating) == 1
        assert PerformanceMetrics.calculate_consecutive_losses(alternating) == 1
    
    def test_calculate_all_trade_statistics_comprehensive(self, mixed_trades):
        stats = PerformanceMetrics.calculate_all_trade_statistics(mixed_trades)
        expected_metrics = [
            'trade_count', 'win_rate', 'profit_factor', 'average_trade_pnl',
            'largest_win', 'largest_loss', 'consecutive_wins', 'consecutive_losses',
            'total_pnl', 'gross_profit', 'gross_loss'
        ]
        for metric in expected_metrics:
            assert metric in stats
            assert isinstance(stats[metric], (int, float))
        
        assert stats['trade_count'] == 8
        assert abs(stats['win_rate'] - 0.5) < 1e-10  # 50% win rate
        assert stats['largest_win'] == 93.0
        assert stats['largest_loss'] == 107.0
        assert stats['total_pnl'] == (93 - 107 + 93 - 107)  # -28
        assert stats['gross_profit'] == (93 + 93)  # 186
        assert stats['gross_loss'] == (107 + 107)  # 214
    
    def test_calculate_all_trade_statistics_edge_cases(self):
        empty_stats = PerformanceMetrics.calculate_all_trade_statistics([])
        expected_empty = {
            'trade_count': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_trade_pnl': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0
        }
        
        assert empty_stats == expected_empty
        buy_only_trades = [
            MockTrade(datetime(2024, 1, 1), "buy", 100, 10, 1000, 5, 2),
            MockTrade(datetime(2024, 1, 2), "buy", 105, 10, 1050, 5, 2),
        ]
        buy_only_stats = PerformanceMetrics.calculate_all_trade_statistics(buy_only_trades)
        assert buy_only_stats['trade_count'] == 2
        assert buy_only_stats['win_rate'] == 0.0
        assert buy_only_stats['profit_factor'] == 0.0
        assert buy_only_stats['total_pnl'] == 0.0
    
    def test_mathematical_relationships(self, mixed_trades):
        stats = PerformanceMetrics.calculate_all_trade_statistics(mixed_trades)
        expected_total_pnl = stats['gross_profit'] - stats['gross_loss']
        assert abs(stats['total_pnl'] - expected_total_pnl) < 1e-10
        if stats['gross_loss'] > 0:
            expected_pf = stats['gross_profit'] / stats['gross_loss']
            assert abs(stats['profit_factor'] - expected_pf) < 1e-10
        
        assert 0.0 <= stats['win_rate'] <= 1.0
        assert stats['consecutive_wins'] >= 0
        assert stats['consecutive_losses'] >= 0
        assert stats['largest_win'] >= 0.0
        assert stats['largest_loss'] >= 0.0
    
    def test_known_scenario_calculations(self):
        perfect_trades = [
            MockTrade(datetime(2024, 1, 1), "sell", 110, 10, 1100, 5, 2, pnl=100),
            MockTrade(datetime(2024, 1, 2), "sell", 120, 10, 1200, 5, 2, pnl=200),
            MockTrade(datetime(2024, 1, 3), "sell", 115, 10, 1150, 5, 2, pnl=150),
        ]
        stats = PerformanceMetrics.calculate_all_trade_statistics(perfect_trades)
        assert stats['win_rate'] == 1.0
        assert stats['profit_factor'] == float('inf')
        assert stats['consecutive_wins'] == 3
        assert stats['consecutive_losses'] == 0
        assert stats['total_pnl'] == 450
        assert stats['gross_profit'] == 450
        assert stats['gross_loss'] == 0
        terrible_trades = [
            MockTrade(datetime(2024, 1, 1), "sell", 90, 10, 900, 5, 2, pnl=-100),
            MockTrade(datetime(2024, 1, 2), "sell", 80, 10, 800, 5, 2, pnl=-200),
        ]
        stats = PerformanceMetrics.calculate_all_trade_statistics(terrible_trades)
        assert stats['win_rate'] == 0.0
        assert stats['profit_factor'] == 0.0
        assert stats['consecutive_wins'] == 0
        assert stats['consecutive_losses'] == 2
        assert stats['total_pnl'] == -300
        assert stats['gross_profit'] == 0
        assert stats['gross_loss'] == 300
    
    def test_integration_with_existing_metrics(self, mixed_trades):
        trade_stats = PerformanceMetrics.calculate_all_trade_statistics(mixed_trades)
        assert len(trade_stats) > 0
        for key, value in trade_stats.items():
            if key != 'profit_factor':
                assert isinstance(value, (int, float))
                if isinstance(value, float):
                    assert not (value != value)  # Check for NaN
    
    def test_performance_with_large_datasets(self):
        large_trades = []
        base_date = datetime(2024, 1, 1)
        for i in range(1000):
            if i % 3 == 0:
                pnl = -50 - (i % 100)
            else:
                pnl = 75 + (i % 50)
            
            trade = MockTrade(
                base_date + timedelta(days=i),
                "sell", 100 + (i % 20), 10, 1000, 5, 2, pnl=pnl
            )
            large_trades.append(trade)
        
        import time
        start_time = time.time()
        stats = PerformanceMetrics.calculate_all_trade_statistics(large_trades)
        end_time = time.time()
        assert (end_time - start_time) < 1.0
        assert stats['trade_count'] == 1000
        assert 0.0 <= stats['win_rate'] <= 1.0
        assert stats['profit_factor'] >= 0.0
        assert stats['consecutive_wins'] >= 0
        assert stats['consecutive_losses'] >= 0
    
    def test_trade_statistics_with_zero_pnl_trades(self):
        breakeven_trades = [
            MockTrade(datetime(2024, 1, 1), "sell", 100, 10, 1000, 5, 2, pnl=0),    # Break-even
            MockTrade(datetime(2024, 1, 2), "sell", 110, 10, 1100, 5, 2, pnl=100),  # Win
            MockTrade(datetime(2024, 1, 3), "sell", 90, 10, 900, 5, 2, pnl=-100),   # Loss
            MockTrade(datetime(2024, 1, 4), "sell", 100, 10, 1000, 5, 2, pnl=0),    # Break-even
        ]
        
        stats = PerformanceMetrics.calculate_all_trade_statistics(breakeven_trades)
        assert stats['win_rate'] == 0.25
        assert stats['profit_factor'] == 1.0
        assert stats['total_pnl'] == 0.0
        assert stats['gross_profit'] == 100.0
        assert stats['gross_loss'] == 100.0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])