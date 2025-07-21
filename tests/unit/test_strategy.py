import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.backtester.strategy.strategy import Strategy, StrategyError, SignalType, StrategyRegistry

class MockStrategy(Strategy):
    def __init__(self, window: int = 10):
        self._window = window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        signals = pd.DataFrame(index=data.index)
        np.random.seed(42)  # For reproducibility
        random_signals = np.random.choice([SignalType.BUY, SignalType.HOLD, SignalType.SELL], size=len(data))
        signals['signal'] = random_signals
        return signals
    
    def get_parameters(self) -> dict:
        return {'window': self._window}
    
    def set_parameters(self, parameters: dict) -> None:
        self.validate_parameters(parameters)
        
        if 'window' in parameters:
            self._window = parameters['window']
    
    @property
    def name(self) -> str:
        return "MockStrategy"
    
    @property
    def description(self) -> str:
        return "A mock strategy for testing purposes."
    
    @property
    def parameter_info(self) -> dict:
        return {
            'window': {
                'type': int,
                'default': 10,
                'min': 1,
                'max': 100,
                'description': 'Window size for calculations'
            }
        }


class TestStrategy:
    @pytest.fixture
    def mock_strategy(self):
        return MockStrategy()
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        data = {
            'Open': np.random.normal(100, 5, 20),
            'High': np.random.normal(105, 5, 20),
            'Low': np.random.normal(95, 5, 20),
            'Close': np.random.normal(102, 5, 20),
            'Volume': np.random.randint(1000, 10000, 20)
        }
        return pd.DataFrame(data, index=dates)
    
    def test_strategy_interface(self):
        print("Testing Strategy interface...")
        with pytest.raises(TypeError):
            Strategy()
        print("✓ Strategy is correctly defined as abstract")
    
    def test_signal_generation(self, mock_strategy, sample_data):
        print("Testing signal generation...")
        signals = mock_strategy.generate_signals(sample_data)
        assert signals.index.equals(sample_data.index)
        assert 'signal' in signals.columns
        assert all(signal in [SignalType.BUY, SignalType.HOLD, SignalType.SELL] for signal in signals['signal'])
        print("✓ Signal generation works correctly")
        print(f"  - Generated {len(signals)} signals")
        print(f"  - Buy signals: {(signals['signal'] == SignalType.BUY).sum()}")
        print(f"  - Sell signals: {(signals['signal'] == SignalType.SELL).sum()}")
        print(f"  - Hold signals: {(signals['signal'] == SignalType.HOLD).sum()}")
    
    def test_parameter_management(self, mock_strategy):
        print("Testing parameter management...")
        params = mock_strategy.get_parameters()
        assert params == {'window': 10}
        mock_strategy.set_parameters({'window': 20})
        params = mock_strategy.get_parameters()
        assert params == {'window': 20}
        print("✓ Parameter management works correctly")
    
    def test_parameter_validation(self, mock_strategy):
        print("Testing parameter validation...")

        with pytest.raises(StrategyError, match="Unknown parameter"):
            mock_strategy.set_parameters({'unknown': 10})
        
        with pytest.raises(StrategyError, match="must be of type int"):
            mock_strategy.set_parameters({'window': 'invalid'})
        
        with pytest.raises(StrategyError, match="must be >= 1"):
            mock_strategy.set_parameters({'window': 0})
        
        with pytest.raises(StrategyError, match="must be <= 100"):
            mock_strategy.set_parameters({'window': 101})
        
        print("✓ Parameter validation works correctly")
    
    def test_data_validation(self, mock_strategy):
        print("Testing data validation...")
        
        with pytest.raises(StrategyError, match="must be a pandas DataFrame"):
            mock_strategy.validate_data("not a dataframe")
        
        with pytest.raises(StrategyError, match="is empty"):
            mock_strategy.validate_data(pd.DataFrame())
        
        df = pd.DataFrame({'Open': [1, 2], 'Close': [3, 4]})
        with pytest.raises(StrategyError, match="missing required columns"):
            mock_strategy.validate_data(df)
        
        df = pd.DataFrame({
            'Open': [1, 2],
            'High': [3, 4],
            'Low': [0.5, 1.5],
            'Close': [2, 3],
            'Volume': [100, 200]
        })
        with pytest.raises(StrategyError, match="must be a DatetimeIndex"):
            mock_strategy.validate_data(df)
        print("✓ Data validation works correctly")
    
    def test_strategy_registry(self):
        print("Testing strategy registry...")
        StrategyRegistry.register(MockStrategy)
        strategies = StrategyRegistry.list_strategies()
        assert "MockStrategy" in strategies
        strategy_class = StrategyRegistry.get_strategy_class("MockStrategy")
        assert strategy_class == MockStrategy
        strategy = StrategyRegistry.create_strategy("MockStrategy", {'window': 15})
        assert isinstance(strategy, MockStrategy)
        assert strategy.get_parameters()['window'] == 15
        info = StrategyRegistry.get_strategy_info()
        assert "MockStrategy" in info
        assert "description" in info["MockStrategy"]
        assert "parameters" in info["MockStrategy"]
        print("✓ Strategy registry works correctly")
        print(f"  - Registered strategies: {strategies}")
    
    def test_strategy_registry_errors(self):
        print("Testing strategy registry error handling...")
        with pytest.raises(StrategyError, match="Strategy not found"):
            StrategyRegistry.get_strategy_class("NonExistentStrategy")
        
        class NotAStrategy:
            pass
        
        with pytest.raises(StrategyError, match="is not a subclass of Strategy"):
            StrategyRegistry.register(NotAStrategy)
        
        print("✓ Strategy registry error handling works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])