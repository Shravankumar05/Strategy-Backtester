import pytest
import pandas as pd
import numpy as np
import operator
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.backtester.strategy.strategy import Strategy, StrategyError, SignalType
from src.backtester.strategy.custom_strategy import CustomStrategy, Rule, Condition, Operator

class TestCustomStrategy:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Create a price series with a clear trend
        close_prices = np.concatenate([
            np.linspace(100, 120, 10),  # Uptrend
            np.linspace(120, 100, 10),  # Downtrend
            np.linspace(100, 110, 10)   # Recovery
        ])
        
        np.random.seed(42)  # Noise :)
        noise = np.random.normal(0, 1, 30)
        close_prices = close_prices + noise
        
        data = {
            'Open': close_prices - 1,
            'High': close_prices + 2,
            'Low': close_prices - 2,
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, 30)
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def simple_strategy(self):
        rules = [
            {
                'conditions': [
                    {
                        'indicator': 'sma_fast',
                        'operator': '>',
                        'value': 'sma_slow'
                    }
                ],
                'action': 'buy'
            },
            {
                'conditions': [
                    {
                        'indicator': 'sma_fast',
                        'operator': '<',
                        'value': 'sma_slow'
                    }
                ],
                'action': 'sell'
            }
        ]
        
        indicators = {
            'sma_fast': {
                'type': 'sma',
                'window': 5,
                'source': 'Close'
            },
            'sma_slow': {
                'type': 'sma',
                'window': 10,
                'source': 'Close'
            }
        }
        
        return CustomStrategy(rules=rules, indicators=indicators)
    
    @pytest.fixture
    def complex_strategy(self):
        # Multi-rule strat
        rules = [
            {
                'conditions': [
                    {
                        'indicator': 'rsi_14',
                        'operator': '<',
                        'value': 30
                    },
                    {
                        'indicator': 'Close',
                        'operator': '<',
                        'value': 'bb_lower'
                    }
                ],
                'action': 'buy'
            },
            {
                'conditions': [
                    {
                        'indicator': 'rsi_14',
                        'operator': '>',
                        'value': 70
                    },
                    {
                        'indicator': 'Close',
                        'operator': '>',
                        'value': 'bb_upper'
                    }
                ],
                'action': 'sell'
            },
            {
                'conditions': [
                    {
                        'indicator': 'macd_hist',
                        'operator': '>',
                        'value': 0
                    }
                ],
                'action': 'buy'
            },
            {
                'conditions': [
                    {
                        'indicator': 'macd_hist',
                        'operator': '<',
                        'value': 0
                    }
                ],
                'action': 'sell'
            }
        ]
        
        indicators = {
            'rsi_14': {
                'type': 'rsi',
                'window': 14,
                'source': 'Close'
            },
            'bb': {
                'type': 'bollinger',
                'window': 20,
                'num_std': 2,
                'source': 'Close'
            },
            'macd': {
                'type': 'macd',
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'source': 'Close'
            }
        }
        return CustomStrategy(rules=rules, indicators=indicators)
    
    def test_operator_enum(self):
        print("Testing Operator enum...")
        assert Operator.get_operator_func(">") == operator.gt
        assert Operator.get_operator_func("<") == operator.lt
        assert Operator.get_operator_func(">=") == operator.ge
        assert Operator.get_operator_func("<=") == operator.le
        assert Operator.get_operator_func("==") == operator.eq
        assert Operator.get_operator_func("!=") == operator.ne
        with pytest.raises(StrategyError, match="Unsupported operator"):
            Operator.get_operator_func("invalid")
        
        print("✓ Operator enum works correctly")
    
    def test_condition_class(self):
        print("Testing Condition class...")
        df = pd.DataFrame({
            'Close': [100, 110, 120],
            'sma_10': [105, 108, 112]
        }, index=[0, 1, 2])
        condition1 = Condition('Close', '>', 110)
        assert condition1.evaluate(df, 0) == False
        assert condition1.evaluate(df, 1) == False
        assert condition1.evaluate(df, 2) == True
        condition2 = Condition('Close', '>', 'sma_10')
        assert condition2.evaluate(df, 0) == False
        assert condition2.evaluate(df, 1) == True
        assert condition2.evaluate(df, 2) == True
        condition3 = Condition('Invalid', '>', 100)
        with pytest.raises(StrategyError, match="Indicator not found"):
            condition3.evaluate(df, 0)
        
        print("✓ Condition class works correctly")
    
    def test_rule_class(self):
        print("Testing Rule class...")
        df = pd.DataFrame({
            'Close': [100, 110, 120],
            'sma_10': [105, 108, 112],
            'rsi': [40, 60, 80]
        }, index=[0, 1, 2])
        rule1 = Rule([{'indicator': 'Close', 'operator': '>', 'value': 110}], 'buy')
        assert rule1.evaluate(df, 0) == SignalType.HOLD
        assert rule1.evaluate(df, 1) == SignalType.HOLD
        assert rule1.evaluate(df, 2) == SignalType.BUY
        rule2 = Rule([
            {'indicator': 'Close', 'operator': '>', 'value': 100},
            {'indicator': 'rsi', 'operator': '>', 'value': 70}
        ], 'sell')
        assert rule2.evaluate(df, 0) == SignalType.HOLD
        assert rule2.evaluate(df, 1) == SignalType.HOLD
        assert rule2.evaluate(df, 2) == SignalType.SELL
        with pytest.raises(StrategyError, match="at least one condition"):
            Rule([], 'buy')
        with pytest.raises(StrategyError, match="Invalid action"):
            Rule([{'indicator': 'Close', 'operator': '>', 'value': 100}], 'invalid')
        
        print("✓ Rule class works correctly")
    
    def test_initialization(self):
        print("Testing strategy initialization...")
        strategy = CustomStrategy()
        assert strategy._rules == []
        assert strategy._indicators == {}
        rules = [
            {
                'conditions': [
                    {
                        'indicator': 'Close',
                        'operator': '>',
                        'value': 100
                    }
                ],
                'action': 'buy'
            }
        ]
        indicators = {
            'sma_10': {
                'type': 'sma',
                'window': 10,
                'source': 'Close'
            }
        }
        strategy = CustomStrategy(rules=rules, indicators=indicators)
        assert len(strategy._rules) == 1
        assert strategy._indicators == indicators
        with pytest.raises(StrategyError, match="Each rule must have"):
            CustomStrategy(rules=[{'invalid': 'rule'}])
        
        print("✓ Strategy initialization works correctly")
    
    def test_indicator_calculation(self, simple_strategy, sample_data):
        print("Testing indicator calculation...")
        data_with_indicators = simple_strategy._calculate_indicators(sample_data)
        assert 'sma_fast' in data_with_indicators.columns
        assert 'sma_slow' in data_with_indicators.columns
        assert data_with_indicators['sma_fast'].iloc[:4].isna().all()
        assert data_with_indicators['sma_slow'].iloc[:9].isna().all()
        close_values = sample_data['Close'].iloc[0:5]
        expected_sma = close_values.mean()
        assert np.isclose(data_with_indicators['sma_fast'].iloc[4], expected_sma)
        print("✓ Indicator calculation works correctly")
    
    def test_signal_generation_simple(self, simple_strategy, sample_data):
        print("Testing signal generation with simple strategy...")
        signals = simple_strategy.generate_signals(sample_data)
        assert signals.index.equals(sample_data.index)
        assert 'signal' in signals.columns
        assert 'sma_fast' in signals.columns
        assert 'sma_slow' in signals.columns
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
        print("✓ Signal generation with simple strategy works correctly")
    
    def test_signal_generation_complex(self, complex_strategy, sample_data):
        print("Testing signal generation with complex strategy...")
        signals = complex_strategy.generate_signals(sample_data)
        assert signals.index.equals(sample_data.index)
        assert 'signal' in signals.columns
        assert 'rsi_14' in signals.columns
        assert 'bb_upper' in signals.columns
        assert 'bb_lower' in signals.columns
        assert 'macd_line' in signals.columns
        assert 'macd_signal' in signals.columns
        assert 'macd_hist' in signals.columns
        buy_count = (signals['signal'] == SignalType.BUY).sum()
        sell_count = (signals['signal'] == SignalType.SELL).sum()
        hold_count = (signals['signal'] == SignalType.HOLD).sum()
        print(f"  - Generated {len(signals)} signals")
        print(f"  - Buy signals: {buy_count}")
        print(f"  - Sell signals: {sell_count}")
        print(f"  - Hold signals: {hold_count}")
        assert buy_count + sell_count + hold_count == len(sample_data)
        print("✓ Signal generation with complex strategy works correctly")
    
    def test_parameter_management(self, simple_strategy):
        print("Testing parameter management...")
        params = simple_strategy.get_parameters()
        assert 'rules' in params
        assert 'indicators' in params
        assert len(params['rules']) == 2
        assert len(params['indicators']) == 2
        new_rules = [
            {
                'conditions': [
                    {
                        'indicator': 'Close',
                        'operator': '>',
                        'value': 100
                    }
                ],
                'action': 'buy'
            }
        ]
        simple_strategy.set_parameters({'rules': new_rules})
        assert len(simple_strategy._rules) == 1
        new_indicators = {
            'rsi_14': {
                'type': 'rsi',
                'window': 14,
                'source': 'Close'
            }
        }
        simple_strategy.set_parameters({'indicators': new_indicators})
        assert simple_strategy._indicators == new_indicators
        print("✓ Parameter management works correctly")
    
    def test_metadata(self, simple_strategy):
        print("Testing strategy metadata...")
        assert isinstance(simple_strategy.name, str)
        assert len(simple_strategy.name) > 0
        assert isinstance(simple_strategy.description, str)
        assert len(simple_strategy.description) > 0
        param_info = simple_strategy.parameter_info
        assert 'rules' in param_info
        assert 'indicators' in param_info
        assert param_info['rules']['type'] == list
        assert param_info['indicators']['type'] == dict
        assert 'description' in param_info['rules']
        assert 'description' in param_info['indicators']
        print("✓ Strategy metadata is correct")
    
    def test_error_handling(self, simple_strategy, sample_data):
        print("Testing error handling...")
        strategy = CustomStrategy()
        with pytest.raises(StrategyError, match="No rules defined"):
            strategy.generate_signals(sample_data)
        
        indicators = {
            'invalid': {
                'type': 'invalid_type',
                'window': 10
            }
        }
        strategy = CustomStrategy(
            rules=[{'conditions': [{'indicator': 'Close', 'operator': '>', 'value': 100}], 'action': 'buy'}],
            indicators=indicators
        )
        
        with pytest.raises(StrategyError, match="Unsupported indicator type"):
            strategy.generate_signals(sample_data)

        print("✓ Error handling works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])