# test_engine.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester.simulation.config import SimulationConfig, PositionSizing
from src.backtester.simulation.engine import SimulationEngine, SimulationError, TradeType
from src.backtester.strategy.strategy import SignalType

class TestSimulationEngine:
    @pytest.fixture
    def config(self):
        return SimulationConfig(
            initial_capital=10000.0,
            leverage=1.0,
            transaction_cost=0.001,
            slippage=0.0005,
            position_sizing=PositionSizing.FIXED_FRACTION,
            position_size=0.1
        )
    
    @pytest.fixture
    def engine(self, config):
        return SimulationEngine(config)
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = {
            'Open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            'High': [102, 103, 104, 105, 106, 107, 106, 105, 104, 103],
            'Low': [99, 100, 101, 102, 103, 104, 103, 102, 101, 100],
            'Close': [101, 102, 103, 104, 105, 106, 105, 104, 103, 102],
            'Volume': [1000] * 10
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_signals(self):
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        signals = [SignalType.BUY] + [SignalType.HOLD] * 3 + [SignalType.SELL] + [SignalType.HOLD] * 5
        return pd.DataFrame({'signal': signals}, index=dates)
    
    def test_initialization(self, engine, config):
        assert engine.config == config
        assert engine.cash == config.initial_capital
        assert engine.position == 0.0
        assert engine.equity == config.initial_capital
        assert len(engine.trades) == 0
        assert len(engine.equity_curve) == 0
        assert engine.position_cost_basis == 0.0
    
    def test_run_simulation_basic(self, engine, sample_data, sample_signals):
        result = engine.run_simulation(sample_data, sample_signals)
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'metrics')
        assert len(result.equity_curve) == len(sample_data)
        assert len(result.trades) == 2  # A Buy and a sell
        
        buy_trade = result.trades[0]
        assert buy_trade.type == TradeType.BUY
        assert buy_trade.price == 101.0 
        assert hasattr(buy_trade, 'commission')
        assert hasattr(buy_trade, 'slippage')
        sell_trade = result.trades[1]
        assert sell_trade.type == TradeType.SELL
        assert sell_trade.price == 105.0
        assert sell_trade.pnl is not None
        assert sell_trade.pnl > 0  # Should be profitable
        assert 'total_return' in result.metrics
        assert 'trade_count' in result.metrics
        assert 'win_rate' in result.metrics
        assert 'profit_factor' in result.metrics
        assert 'max_drawdown' in result.metrics
        assert result.metrics['trade_count'] == 2
        assert result.metrics['win_rate'] == 1.0  # 100% win rate
    
    def test_position_sizing_fixed_fraction(self, config, sample_data, sample_signals):
        config.position_sizing = PositionSizing.FIXED_FRACTION
        config.position_size = 0.2
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        buy_trade = result.trades[0]
        expected_value = config.initial_capital * config.position_size
        expected_size = expected_value / buy_trade.price
        assert abs(buy_trade.size - expected_size) < 0.001
        assert abs(buy_trade.value - expected_value) < 0.01
    
    def test_position_sizing_fixed_size(self, config, sample_data, sample_signals):
        config.position_sizing = PositionSizing.FIXED_SIZE
        config.position_size = 10.0  # 10 units
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        buy_trade = result.trades[0]
        assert buy_trade.size == 10.0
        assert buy_trade.value == 10.0 * buy_trade.price
    
    def test_transaction_costs_calculation(self, config, sample_data, sample_signals):
        config.transaction_cost = 0.01  # 1%
        config.slippage = 0.005  # 0.5%
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        buy_trade = result.trades[0]
        expected_commission = buy_trade.value * config.transaction_cost
        expected_slippage = buy_trade.value * config.slippage
        assert abs(buy_trade.commission - expected_commission) < 0.001
        assert abs(buy_trade.slippage - expected_slippage) < 0.001
    
    def test_leverage_functionality(self, config, sample_data, sample_signals):
        config.leverage = 2.0
        config.position_size = 0.5  # 50% of capital
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        buy_trade = result.trades[0]
        expected_value = config.initial_capital * config.leverage * config.position_size
        expected_size = expected_value / buy_trade.price
        assert abs(buy_trade.size - expected_size) < 0.001
    
    def test_margin_call_scenario(self, config, sample_data):
        config.leverage = 2.0
        config.position_size = 0.9  # 90% of capital with 2x leverage = high risk
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        signals = [SignalType.BUY] + [SignalType.HOLD] * 9
        signals_df = pd.DataFrame({'signal': signals}, index=dates)
        data = sample_data.copy()
        data.loc[dates[5], 'Close'] = 10.0  # 90% price drop on day 6 - this should definitely trigger margin call
        engine = SimulationEngine(config)
        result = engine.run_simulation(data, signals_df)
        assert len(result.trades) >= 2
        assert result.trades[0].type == TradeType.BUY
        margin_call_trades = [t for t in result.trades if t.price == 10.0]
        assert len(margin_call_trades) > 0
        margin_call_trade = margin_call_trades[0]
        assert margin_call_trade.type == TradeType.SELL
        assert margin_call_trade.timestamp == dates[5]
    
    def test_pnl_calculation_accuracy(self, config, sample_data):
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'Open': [100, 110, 120],
            'High': [102, 112, 122],
            'Low': [98, 108, 118],
            'Close': [100, 110, 120],
            'Volume': [1000] * 3
        }, index=dates)
        
        signals = pd.DataFrame({
            'signal': [SignalType.BUY, SignalType.HOLD, SignalType.SELL]
        }, index=dates)
        
        config.position_sizing = PositionSizing.FIXED_SIZE
        config.position_size = 10.0 
        config.transaction_cost = 0.0
        config.slippage = 0.0
        engine = SimulationEngine(config)
        result = engine.run_simulation(data, signals)
        buy_trade = result.trades[0]
        sell_trade = result.trades[1]
        expected_pnl = (sell_trade.price - buy_trade.price) * sell_trade.size
        assert abs(sell_trade.pnl - expected_pnl) < 0.001
    
    def test_partial_position_selling(self, config, sample_data):
        dates = pd.date_range(start='2024-01-01', periods=4, freq='D')
        data = pd.DataFrame({
            'Open': [100, 110, 120, 130],
            'High': [102, 112, 122, 132],
            'Low': [98, 108, 118, 128],
            'Close': [100, 110, 120, 130],
            'Volume': [1000] * 4
        }, index=dates)
        signals = pd.DataFrame({
            'signal': [SignalType.BUY, SignalType.HOLD, SignalType.SELL, SignalType.SELL]
        }, index=dates)
        config.position_sizing = PositionSizing.FIXED_SIZE
        config.position_size = 10.0  # Buy 10 shares initially
        config.transaction_cost = 0.0
        config.slippage = 0.0
        engine = SimulationEngine(config)
        original_execute_sell = engine._execute_sell_trade
        def modified_execute_sell(timestamp, price, margin_call=False):
            if not margin_call and engine.position > 0:
                old_position_size = engine.config.position_size
                engine.config.position_size = 5.0
                original_execute_sell(timestamp, price, margin_call)
                engine.config.position_size = old_position_size
            else:
                original_execute_sell(timestamp, price, margin_call)
        
        engine._execute_sell_trade = modified_execute_sell
        result = engine.run_simulation(data, signals)
        assert len(result.trades) >= 2  # At least buy and one sell
        buy_trade = result.trades[0]
        assert buy_trade.type == TradeType.BUY
        assert buy_trade.size == 10.0
        sell_trades = [t for t in result.trades if t.type == TradeType.SELL]
        assert len(sell_trades) >= 1
        for sell_trade in sell_trades:
            assert sell_trade.pnl > 0
    
    def test_insufficient_funds_handling(self, config, sample_data, sample_signals):
        config.initial_capital = 100.0
        config.position_size = 1.0  # Buy more than we can afford - do not try at home
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        assert hasattr(result, 'trades')
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'metrics')
    
    def test_no_position_sell_handling(self, config, sample_data):
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        signals = pd.DataFrame({
            'signal': [SignalType.SELL, SignalType.HOLD, SignalType.HOLD]  # Try to sell first
        }, index=dates[:3])
        
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data.iloc[:3], signals)
        assert len(result.trades) == 0
    
    def test_empty_data_error(self, engine):
        empty_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
        empty_signals = pd.DataFrame(columns=['signal'])
        with pytest.raises(SimulationError, match="Data cannot be empty"):
            engine.run_simulation(empty_data, empty_signals)
    
    def test_mismatched_data_length_error(self, engine):
        data = pd.DataFrame({'Open': [100], 'High': [101], 'Low': [99], 'Close': [100]})
        signals = pd.DataFrame({'signal': [SignalType.BUY, SignalType.SELL]})
        with pytest.raises(SimulationError, match="Data and signals must have the same length"):
            engine.run_simulation(data, signals)
    
    def test_missing_ohlc_columns_error(self, engine):
        data = pd.DataFrame({'Price': [100]})
        signals = pd.DataFrame({'signal': [SignalType.BUY]})
        with pytest.raises(SimulationError, match="Data must contain OHLC columns"):
            engine.run_simulation(data, signals)
    
    def test_missing_signal_column_error(self, engine):
        data = pd.DataFrame({'Open': [100], 'High': [101], 'Low': [99], 'Close': [100]})
        signals = pd.DataFrame({'action': [SignalType.BUY]})  # Wrong column name
        with pytest.raises(SimulationError, match="Signals DataFrame must contain a 'signal' column"):
            engine.run_simulation(data, signals)
    
    def test_equity_curve_completeness(self, engine, sample_data, sample_signals):
        result = engine.run_simulation(sample_data, sample_signals)
        required_columns = ['equity', 'cash', 'position', 'position_value']
        for col in required_columns:
            assert col in result.equity_curve.columns
        
        assert len(result.equity_curve) == len(sample_data)
        assert all(result.equity_curve['equity'] > 0)
        assert all(result.equity_curve['cash'] >= 0)
    
    def test_metrics_completeness(self, engine, sample_data, sample_signals):
        result = engine.run_simulation(sample_data, sample_signals)
        required_metrics = [
            'total_return', 'trade_count', 'win_rate', 'profit_factor',
            'max_drawdown', 'max_drawdown_pct'
        ]
        for metric in required_metrics:
            assert metric in result.metrics
            assert isinstance(result.metrics[metric], (int, float))
            assert not np.isnan(result.metrics[metric])

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])