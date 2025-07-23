# test_engine.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtester.simulation.config import SimulationConfig, PositionSizing
from src.backtester.simulation.engine import SimulationEngine, SimulationError, TradeType
from src.backtester.strategy.strategy import SignalType

class TestSimulationEngine:
    """Test suite for the SimulationEngine class."""
    
    @pytest.fixture
    def config(self):
        """Create a simulation configuration for testing."""
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
        """Create a simulation engine for testing."""
        return SimulationEngine(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
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
        """Create sample signals for testing."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        
        # Create a simple strategy: buy on day 1, hold, sell on day 5
        signals = [SignalType.BUY] + [SignalType.HOLD] * 3 + [SignalType.SELL] + [SignalType.HOLD] * 4
        
        return pd.DataFrame({'signal': signals}, index=dates)
    
    def test_initialization(self, engine, config):
        """Test engine initialization."""
        assert engine.config == config
        assert engine.cash == config.initial_capital
        assert engine.position == 0.0
        assert engine.equity == config.initial_capital
        assert len(engine.trades) == 0
        assert len(engine.equity_curve) == 0
    
    def test_run_simulation_basic(self, engine, sample_data, sample_signals):
        """Test running a basic simulation."""
        result = engine.run_simulation(sample_data, sample_signals)
        
        # Check that the result has the expected attributes
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'metrics')
        
        # Check that the equity curve has the expected length
        assert len(result.equity_curve) == len(sample_data)
        
        # Check that trades were executed
        assert len(result.trades) == 2  # Buy and sell
        
        # Check trade details
        buy_trade = result.trades[0]
        assert buy_trade.type == TradeType.BUY
        assert buy_trade.price == 101.0  # Close price on day 1
        
        sell_trade = result.trades[1]
        assert sell_trade.type == TradeType.SELL
        assert sell_trade.price == 106.0  # Close price on day 5
        
        # Check that metrics were calculated
        assert 'total_return' in result.metrics
        assert 'trade_count' in result.metrics
        assert 'win_rate' in result.metrics
        
        # Check that the trade was profitable
        assert sell_trade.pnl > 0
    
    def test_position_sizing_fixed_fraction(self, config, sample_data, sample_signals):
        """Test position sizing with fixed fraction."""
        config.position_sizing = PositionSizing.FIXED_FRACTION
        config.position_size = 0.2  # 20% of capital
        
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        
        # Check that the position size is correct
        buy_trade = result.trades[0]
        expected_size = (config.initial_capital * config.position_size) / buy_trade.price
        assert abs(buy_trade.size - expected_size) < 0.001
    
    def test_position_sizing_fixed_size(self, config, sample_data, sample_signals):
        """Test position sizing with fixed size."""
        config.position_sizing = PositionSizing.FIXED_SIZE
        config.position_size = 10.0  # 10 units
        
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        
        # Check that the position size is correct
        buy_trade = result.trades[0]
        assert buy_trade.size == 10.0
    
    def test_transaction_costs(self, config, sample_data, sample_signals):
        """Test transaction costs."""
        config.transaction_cost = 0.01  # 1%
        config.slippage = 0.005  # 0.5%
        
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        
        # Check that transaction costs were applied
        buy_trade = result.trades[0]
        expected_commission = buy_trade.value * config.transaction_cost
        expected_slippage = buy_trade.value * config.slippage
        
        assert abs(buy_trade.commission - expected_commission) < 0.001
        assert abs(buy_trade.slippage - expected_slippage) < 0.001
    
    def test_leverage(self, config, sample_data, sample_signals):
        """Test leverage."""
        config.leverage = 2.0
        config.position_size = 0.5  # 50% of capital
        
        engine = SimulationEngine(config)
        result = engine.run_simulation(sample_data, sample_signals)
        
        # Check that leverage was applied correctly
        buy_trade = result.trades[0]
        expected_size = (config.initial_capital * config.leverage * config.position_size) / buy_trade.price
        assert abs(buy_trade.size - expected_size) < 0.001
    
    def test_margin_call(self, config, sample_data):
        """Test margin call."""
        config.leverage = 2.0
        config.position_size = 0.9  # 90% of capital with 2x leverage
        
        # Create signals that will trigger a margin call
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        signals = [SignalType.BUY] + [SignalType.HOLD] * 8
        signals_df = pd.DataFrame({'signal': signals}, index=dates)
        
        # Modify data to create a price drop that will trigger a margin call
        data = sample_data.copy()
        data.loc[dates[5], 'Close'] = 50.0  # Big price drop on day 6
        
        engine = SimulationEngine(config)
        result = engine.run_simulation(data, signals_df)
        
        # Check that a margin call was triggered
        assert len(result.trades) == 2
        assert result.trades[0].type == TradeType.BUY
        assert result.trades[1].type == TradeType.SELL
        
        # The second trade should be a margin call liquidation
        assert result.trades[1].timestamp == dates[5]
        assert result.trades[1].price == 50.0
    
    def test_invalid_inputs(self, engine):
        """Test handling of invalid inputs."""
        # Test with mismatched data and signals lengths
        data = pd.DataFrame({'Close': [100, 101]})
        signals = pd.DataFrame({'signal': [SignalType.BUY]})
        
        with pytest.raises(SimulationError, match="Data and signals must have the same length"):
            engine.run_simulation(data, signals)
        
        # Test with missing OHLC columns
        data = pd.DataFrame({'Price': [100]})
        signals = pd.DataFrame({'signal': [SignalType.BUY]})
        
        with pytest.raises(SimulationError, match="Data must contain OHLC columns"):
            engine.run_simulation(data, signals)
        
        # Test with missing signal column
        data = pd.DataFrame({'Open': [100], 'High': [101], 'Low': [99], 'Close': [100]})
        signals = pd.DataFrame({'action': [SignalType.BUY]})
        
        with pytest.raises(SimulationError, match="Signals DataFrame must contain a 'signal' column"):
            engine.run_simulation(data, signals)
