import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.backtester.strategy.strategy import Strategy, StrategyError, SignalType
from src.backtester.strategy.ma_crossover import MovingAverageCrossoverStrategy

class TestMovingAverageCrossoverStrategy:
    @pytest.fixture
    def strategy(self):
        return MovingAverageCrossoverStrategy(short_window=5, long_window=10)
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # First 15 days: downtrend
        # Last 15 days: uptrend
        close_prices = np.concatenate([
            np.linspace(100, 80, 15),
            np.linspace(80, 120, 15)
        ])
        
        noise = np.random.normal(0, 1, 30) # Noise for fun :)
        close_prices = close_prices + noise
        
        data = {
            'Open': close_prices - 1,
            'High': close_prices + 2,
            'Low': close_prices - 2,
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 30)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_initialization(self):
        print("Testing strategy initialization...")
        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)
        assert strategy._short_window == 5
        assert strategy._long_window == 10
        print("✓ Strategy initialization works correctly")
    
    def test_parameter_management(self):
        print("Testing parameter management...")
        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)
        params = strategy.get_parameters()
        assert params == {'short_window': 5, 'long_window': 10}
        strategy.set_parameters({'short_window': 3})
        assert strategy.get_parameters() == {'short_window': 3, 'long_window': 10}
        strategy.set_parameters({'long_window': 15})
        assert strategy.get_parameters() == {'short_window': 3, 'long_window': 15}
        strategy.set_parameters({'short_window': 7, 'long_window': 20})
        assert strategy.get_parameters() == {'short_window': 7, 'long_window': 20}
        
        # set_parameters with invalid values
        with pytest.raises(StrategyError, match="Short window .* must be less than long window"):
            strategy.set_parameters({'short_window': 20})
        
        with pytest.raises(StrategyError, match="Short window .* must be less than long window"):
            strategy.set_parameters({'long_window': 7})
        
        with pytest.raises(StrategyError, match="Parameter .* must be >= 2"):
            strategy.set_parameters({'short_window': 1})
        
        with pytest.raises(StrategyError, match="Parameter .* must be of type int"):
            strategy.set_parameters({'short_window': 5.5})
        
        print("✓ Parameter management works correctly")
    
    def test_signal_generation(self, strategy, sample_data):
        print("Testing signal generation...")
        signals = strategy.generate_signals(sample_data)
        assert signals.index.equals(sample_data.index)
        assert 'short_ma' in signals.columns
        assert 'long_ma' in signals.columns
        assert 'signal' in signals.columns
        assert np.isclose(signals['short_ma'].iloc[4], sample_data['Close'].iloc[0:5].mean())
        assert np.isclose(signals['long_ma'].iloc[9], sample_data['Close'].iloc[0:10].mean())
        assert signals['signal'].iloc[0] == SignalType.HOLD
        buy_count = (signals['signal'] == SignalType.BUY).sum()
        sell_count = (signals['signal'] == SignalType.SELL).sum()
        hold_count = (signals['signal'] == SignalType.HOLD).sum()
        
        print(f"  - Generated {len(signals)} signals")
        print(f"  - Buy signals: {buy_count}")
        print(f"  - Sell signals: {sell_count}")
        print(f"  - Hold signals: {hold_count}")
        
        assert buy_count > 0
        assert buy_count + sell_count + hold_count == len(sample_data)
        print("✓ Signal generation works correctly")
    
    def test_crossover_detection(self):
        print("Testing crossover detection...")
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        data = {
            'Open': [100] * 15,
            'High': [110] * 15,
            'Low': [90] * 15,
            'Close': [100] * 15,
            'Volume': [1000] * 15
        }
        df = pd.DataFrame(data, index=dates)
        strategy = MovingAverageCrossoverStrategy(short_window=2, long_window=3)
        
        df['Close'] = [95, 95, 95, 95, 95, 105, 105, 105, 105, 105, 95, 95, 95, 95, 95]
        signals = strategy.generate_signals(df)
        
        print("Debug info:")
        debug_df = pd.DataFrame({
            'Close': df['Close'],
            'short_ma': signals['short_ma'],
            'long_ma': signals['long_ma'],
            'signal': signals['signal']
        })
        print(debug_df)
        
        buy_index = signals.index[5]  # Day 6
        assert signals.loc[buy_index, 'signal'] == SignalType.BUY, f"Expected BUY signal at index {buy_index}"

        sell_index = signals.index[7]  # Day 8
        assert signals.loc[sell_index, 'signal'] == SignalType.SELL, f"Expected SELL signal at index {sell_index}"
        
        buy_count = (signals['signal'] == SignalType.BUY).sum()
        sell_count = (signals['signal'] == SignalType.SELL).sum()
        hold_count = (signals['signal'] == SignalType.HOLD).sum()
        
        print(f"Buy signals: {buy_count}, Sell signals: {sell_count}, Hold signals: {hold_count}")
        
        assert buy_count == 1, f"Expected 1 buy signal, got {buy_count}"
        assert sell_count == 1, f"Expected 1 sell signal, got {sell_count}"
        
        print("✓ Crossover detection works correctly")
    
    def test_not_enough_data(self, strategy):
        print("Testing insufficient data handling...")
        
        # Dataset with fewer points than required for the long window
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = {
            'Open': [100] * 5,
            'High': [110] * 5,
            'Low': [90] * 5,
            'Close': [100] * 5,
            'Volume': [1000] * 5
        }
        df = pd.DataFrame(data, index=dates)
        
        with pytest.raises(StrategyError, match="Not enough data for long window"):
            strategy.generate_signals(df)
        
        print("✓ Insufficient data handling works correctly")
    
    def test_metadata(self, strategy):
        print("Testing strategy metadata...")
        
        assert isinstance(strategy.name, str)
        assert len(strategy.name) > 0
        assert isinstance(strategy.description, str)
        assert len(strategy.description) > 0
        param_info = strategy.parameter_info
        assert 'short_window' in param_info
        assert 'long_window' in param_info
        assert param_info['short_window']['type'] == int
        assert param_info['long_window']['type'] == int
        assert 'min' in param_info['short_window']
        assert 'max' in param_info['short_window']
        assert 'default' in param_info['short_window']
        assert 'description' in param_info['short_window']
        print("✓ Strategy metadata is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])