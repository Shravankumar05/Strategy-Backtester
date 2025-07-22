import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.backtester.strategy.strategy import Strategy, StrategyError, SignalType
from src.backtester.strategy.rsi_strategy import RSIStrategy

class TestRSIStrategy:
    @pytest.fixture
    def strategy(self):
        return RSIStrategy(period=5, overbought=70, oversold=30)
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Create a price series with a clear trend change
        # First 10 days: uptrend (to generate overbought condition)
        # Next 10 days: downtrend (to generate oversold condition)
        # Last 10 days: recovery
        close_prices = np.concatenate([
            np.linspace(100, 150, 10),
            np.linspace(150, 80, 10),
            np.linspace(80, 110, 10)
        ])
        
        # Noise :)
        np.random.seed(42)
        noise = np.random.normal(0, 2, 30)
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
        strategy = RSIStrategy(period=14, overbought=70, oversold=30)
        assert strategy._period == 14
        assert strategy._overbought == 70
        assert strategy._oversold == 30
        
        with pytest.raises(StrategyError, match="Period must be >= 2"):
            RSIStrategy(period=1)

        with pytest.raises(StrategyError, match="Overbought threshold .* must be > oversold threshold"):
            RSIStrategy(overbought=30, oversold=30)
        
        with pytest.raises(StrategyError, match="Overbought threshold .* must be > oversold threshold"):
            RSIStrategy(overbought=20, oversold=30)
        
        with pytest.raises(StrategyError, match="Overbought threshold must be <= 100"):
            RSIStrategy(overbought=110)
        
        with pytest.raises(StrategyError, match="Oversold threshold must be >= 0"):
            RSIStrategy(oversold=-10)
        
        print("✓ Strategy initialization works correctly")
    
    def test_parameter_management(self):
        print("Testing parameter management...")
        strategy = RSIStrategy(period=14, overbought=70, oversold=30)
        params = strategy.get_parameters()
        assert params == {'period': 14, 'overbought': 70, 'oversold': 30}
        strategy.set_parameters({'period': 7})
        assert strategy.get_parameters() == {'period': 7, 'overbought': 70, 'oversold': 30}
        strategy.set_parameters({'overbought': 80})
        assert strategy.get_parameters() == {'period': 7, 'overbought': 80, 'oversold': 30}
        strategy.set_parameters({'oversold': 20})
        assert strategy.get_parameters() == {'period': 7, 'overbought': 80, 'oversold': 20}
        strategy.set_parameters({'period': 10, 'overbought': 75, 'oversold': 25})
        assert strategy.get_parameters() == {'period': 10, 'overbought': 75, 'oversold': 25}
        
        # Invalid values
        with pytest.raises(StrategyError, match=".*period.*>= 2"):
            strategy.set_parameters({'period': 1})
        
        with pytest.raises(StrategyError, match="Overbought threshold .* must be > oversold threshold"):
            strategy.set_parameters({'overbought': 20})
        
        with pytest.raises(StrategyError, match="Overbought threshold .* must be > oversold threshold"):
            strategy.set_parameters({'oversold': 80})
        
        with pytest.raises(StrategyError, match="Parameter .* must be of type int"):
            strategy.set_parameters({'period': 5.5})
        
        print("✓ Parameter management works correctly")
    
    def test_rsi_calculation(self, strategy, sample_data):
        print("Testing RSI calculation...")
        rsi = strategy._calculate_rsi(sample_data['Close'])
        assert isinstance(rsi, pd.Series)
        assert rsi.index.equals(sample_data.index)
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        assert rsi.iloc[:strategy._period-1].isna().all()
        assert not rsi.iloc[strategy._period:].isna().any()
        print("✓ RSI calculation works correctly")
    
    def test_signal_generation(self, strategy, sample_data):
        print("Testing signal generation...")
        signals = strategy.generate_signals(sample_data)
        assert signals.index.equals(sample_data.index)
        assert 'rsi' in signals.columns
        assert 'signal' in signals.columns
        assert signals['rsi'].min() >= 0
        assert signals['rsi'].max() <= 100
        buy_count = (signals['signal'] == SignalType.BUY).sum()
        sell_count = (signals['signal'] == SignalType.SELL).sum()
        hold_count = (signals['signal'] == SignalType.HOLD).sum()
        print(f"  - Generated {len(signals)} signals")
        print(f"  - Buy signals: {buy_count}")
        print(f"  - Sell signals: {sell_count}")
        print(f"  - Hold signals: {hold_count}")
        assert buy_count > 0
        assert sell_count > 0
        assert buy_count + sell_count + hold_count == len(sample_data)
        print("✓ Signal generation works correctly")
    
    def test_overbought_oversold_detection(self):
        print("Testing overbought and oversold detection...")
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        strategy = RSIStrategy(period=2, overbought=70, oversold=30)
        
        changes = [
            5, -3, 4, -2,
            # Strong gains to push RSI above 70
            10, 8, 6, 4,
            # Strong losses to push RSI below 30
            -12, -10, -8, -6,
            6, -4, 5, -3, 4, -2, 3, -1
        ]
        
        prices = [100]
        for change in changes:
            prices.append(prices[-1] + change)
        
        data = {
            'Open': prices[:-1],
            'High': [p + 1 for p in prices[:-1]],
            'Low': [p - 1 for p in prices[:-1]],
            'Close': prices[1:],
            'Volume': [1000] * len(prices[1:])
        }
        df = pd.DataFrame(data, index=dates)
        signals = strategy.generate_signals(df)
        print("Debug info:")
        debug_df = pd.DataFrame({
            'Close': df['Close'],
            'RSI': signals['rsi'],
            'Signal': signals['signal']
        })
        print(debug_df)
        overbought_indices = signals.index[
            (signals['rsi'] > strategy._overbought) & 
            (signals['rsi'].shift(1) <= strategy._overbought)
        ]
        oversold_indices = signals.index[
            (signals['rsi'] < strategy._oversold) & 
            (signals['rsi'].shift(1) >= strategy._oversold)
        ]
        
        assert len(overbought_indices) > 0, "No overbought condition detected"
        assert len(oversold_indices) > 0, "No oversold condition detected"
        
        for idx in overbought_indices:
            assert signals.loc[idx, 'signal'] == SignalType.SELL, f"Expected SELL signal at index {idx}"
        
        for idx in oversold_indices:
            assert signals.loc[idx, 'signal'] == SignalType.BUY, f"Expected BUY signal at index {idx}"
        
        buy_count = (signals['signal'] == SignalType.BUY).sum()
        sell_count = (signals['signal'] == SignalType.SELL).sum()
        hold_count = (signals['signal'] == SignalType.HOLD).sum()
        print(f"Buy signals: {buy_count}, Sell signals: {sell_count}, Hold signals: {hold_count}")
        print("✓ Overbought and oversold detection works correctly")
    
    def test_not_enough_data(self, strategy):
        print("Testing insufficient data handling...")
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        data = {
            'Open': [100, 102, 104],
            'High': [105, 107, 109],
            'Low': [98, 100, 102],
            'Close': [102, 104, 106],
            'Volume': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data, index=dates)
        with pytest.raises(StrategyError, match="Not enough data for RSI calculation"):
            strategy.generate_signals(df)

        print("✓ Insufficient data handling works correctly")
    
    def test_metadata(self, strategy):
        print("Testing strategy metadata...")
        assert isinstance(strategy.name, str)
        assert len(strategy.name) > 0
        assert isinstance(strategy.description, str)
        assert len(strategy.description) > 0
        param_info = strategy.parameter_info
        assert 'period' in param_info
        assert 'overbought' in param_info
        assert 'oversold' in param_info
        assert param_info['period']['type'] == int
        assert param_info['overbought']['type'] == int
        assert param_info['oversold']['type'] == int
        assert 'min' in param_info['period']
        assert 'max' in param_info['period']
        assert 'default' in param_info['period']
        assert 'description' in param_info['period']
        print("✓ Strategy metadata is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])