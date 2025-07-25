import pytest
import pandas as pd
import sys
import os
from datetime import date
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.modules['streamlit'] = MagicMock()
from backtester.ui.utils.session_state import (initialize_session_state, get_session_state, set_session_state,get_backtest_config, is_config_valid)

class TestUIIntegration:
    def setup_method(self):
        self.mock_session_state = {}
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.session_state_patcher.start()
    
    def teardown_method(self):
        self.session_state_patcher.stop()
    
    def test_session_state_initialization(self):
        initialize_session_state()
        required_keys = ['ticker', 'start_date', 'end_date', 'initial_capital', 'leverage', 'strategy_type', 'transaction_cost', 'slippage', 'position_sizing']
        
        for key in required_keys:
            assert key in self.mock_session_state
        
        assert self.mock_session_state['ticker'] == 'AAPL'
        assert self.mock_session_state['initial_capital'] == 10000.0
        assert self.mock_session_state['leverage'] == 1.0
        assert self.mock_session_state['strategy_type'] == 'Moving Average Crossover'
    
    def test_config_validation_valid_config(self):
        initialize_session_state()
        set_session_state('ticker', 'AAPL')
        set_session_state('start_date', date(2024, 1, 1))
        set_session_state('end_date', date(2024, 6, 30))
        set_session_state('initial_capital', 10000.0)
        set_session_state('leverage', 1.5)
        set_session_state('strategy_type', 'Moving Average Crossover')
        set_session_state('short_window', 20)
        set_session_state('long_window', 50)
        is_valid, error = is_config_valid()
        assert is_valid
        assert error == ""
    
    def test_config_validation_invalid_dates(self):
        initialize_session_state()
        set_session_state('start_date', date(2024, 6, 30))
        set_session_state('end_date', date(2024, 1, 1))
        is_valid, error = is_config_valid()
        assert not is_valid
        assert "Start date must be before end date" in error
    
    def test_config_validation_invalid_ma_windows(self):
        initialize_session_state()
        set_session_state('strategy_type', 'Moving Average Crossover')
        set_session_state('short_window', 50)
        set_session_state('long_window', 20)
        is_valid, error = is_config_valid()
        assert not is_valid
        assert "Short MA period must be less than Long MA period" in error
    
    def test_config_validation_invalid_capital(self):
        initialize_session_state()
        set_session_state('initial_capital', -1000.0)
        is_valid, error = is_config_valid()
        assert not is_valid
        assert "Initial capital must be positive" in error
    
    def test_backtest_config_generation(self):
        initialize_session_state()
        set_session_state('ticker', 'MSFT')
        set_session_state('initial_capital', 25000.0)
        set_session_state('leverage', 2.0)
        set_session_state('strategy_type', 'RSI Strategy')
        set_session_state('rsi_period', 21)
        set_session_state('rsi_overbought', 75)
        set_session_state('rsi_oversold', 25)
        config = get_backtest_config()
        assert config['ticker'] == 'MSFT'
        assert config['initial_capital'] == 25000.0
        assert config['leverage'] == 2.0
        assert config['strategy_type'] == 'RSI Strategy'
        strategy_params = config['strategy_params']
        assert strategy_params['period'] == 21
        assert strategy_params['overbought'] == 75
        assert strategy_params['oversold'] == 25
    
    def test_session_state_persistence(self):
        initialize_session_state()
        test_values = {
            'ticker': 'GOOGL',
            'initial_capital': 50000.0,
            'leverage': 1.8,
            'strategy_type': 'Moving Average Crossover',
            'short_window': 15,
            'long_window': 45
        }
        for key, value in test_values.items():
            set_session_state(key, value)
        for key, expected_value in test_values.items():
            actual_value = get_session_state(key)
            assert actual_value == expected_value
    
    def test_strategy_parameter_handling(self):
        initialize_session_state()
        set_session_state('strategy_type', 'Moving Average Crossover')
        set_session_state('short_window', 10)
        set_session_state('long_window', 30)
        config = get_backtest_config()
        ma_params = config['strategy_params']
        assert ma_params['short_window'] == 10
        assert ma_params['long_window'] == 30
        set_session_state('strategy_type', 'RSI Strategy')
        set_session_state('rsi_period', 14)
        set_session_state('rsi_overbought', 70)
        set_session_state('rsi_oversold', 30)
        config = get_backtest_config()
        rsi_params = config['strategy_params']
        assert rsi_params['period'] == 14
        assert rsi_params['overbought'] == 70
        assert rsi_params['oversold'] == 30
        set_session_state('strategy_type', 'Buy and Hold')
        config = get_backtest_config()
        bh_params = config['strategy_params']
        assert bh_params == {}
    
    def test_position_sizing_validation(self):
        initialize_session_state()
        set_session_state('position_sizing', 'Fixed Fraction')
        set_session_state('position_size', 0.2)  # 20%
        is_valid, error = is_config_valid()
        assert is_valid
        set_session_state('position_size', 1.5)  # 150%
        is_valid, error = is_config_valid()
        assert not is_valid
        assert "Position size fraction must be between 0 and 1" in error
        set_session_state('position_sizing', 'Fixed Size')
        set_session_state('position_size', 5000.0)
        set_session_state('initial_capital', 10000.0)
        is_valid, error = is_config_valid()
        assert is_valid
        set_session_state('position_size', 15000.0)
        is_valid, error = is_config_valid()
        assert not is_valid
        assert "Position size must be positive and less than initial capital" in error


class TestUIWorkflowIntegration:
    def setup_method(self):
        self.mock_session_state = {}
        self.session_state_patcher = patch('streamlit.session_state', self.mock_session_state)
        self.session_state_patcher.start()
    
    def teardown_method(self):
        self.session_state_patcher.stop()
    
    def test_complete_configuration_workflow(self):
        initialize_session_state()
        set_session_state('ticker', 'TSLA')
        set_session_state('start_date', date(2024, 2, 1))
        set_session_state('end_date', date(2024, 5, 31))
        set_session_state('initial_capital', 20000.0)
        set_session_state('leverage', 1.2)
        set_session_state('strategy_type', 'RSI Strategy')
        set_session_state('rsi_period', 10)
        set_session_state('rsi_overbought', 80)
        set_session_state('rsi_oversold', 20)
        set_session_state('transaction_cost', 0.002)  # 0.2%
        set_session_state('slippage', 0.001)  # 0.1%
        set_session_state('position_sizing', 'Fixed Fraction')
        set_session_state('position_size', 0.15)  # 15%
        is_valid, error = is_config_valid()
        assert is_valid, f"Configuration should be valid but got error: {error}"
        config = get_backtest_config()
        assert config['ticker'] == 'TSLA'
        assert config['start_date'] == date(2024, 2, 1)
        assert config['end_date'] == date(2024, 5, 31)
        assert config['initial_capital'] == 20000.0
        assert config['leverage'] == 1.2
        assert config['strategy_type'] == 'RSI Strategy'
        assert config['transaction_cost'] == 0.002
        assert config['slippage'] == 0.001
        assert config['position_sizing'] == 'Fixed Fraction'
        assert config['position_size'] == 0.15
        strategy_params = config['strategy_params']
        assert strategy_params['period'] == 10
        assert strategy_params['overbought'] == 80
        assert strategy_params['oversold'] == 20
    
    def test_configuration_error_recovery(self):
        initialize_session_state()
        set_session_state('initial_capital', -5000.0)  # Invalid
        set_session_state('start_date', date(2024, 6, 1))
        set_session_state('end_date', date(2024, 1, 1))  # Invalid (before start)
        is_valid, error = is_config_valid()
        assert not is_valid
        set_session_state('initial_capital', 15000.0)  # Fix capital
        set_session_state('end_date', date(2024, 12, 31))  # Fix date
        is_valid, error = is_config_valid()
        assert is_valid
    
    def test_strategy_switching_workflow(self):
        initialize_session_state()
        set_session_state('strategy_type', 'Moving Average Crossover')
        set_session_state('short_window', 12)
        set_session_state('long_window', 26)
        config1 = get_backtest_config()
        assert config1['strategy_type'] == 'Moving Average Crossover'
        assert config1['strategy_params']['short_window'] == 12
        assert config1['strategy_params']['long_window'] == 26
        set_session_state('strategy_type', 'RSI Strategy')
        set_session_state('rsi_period', 21)
        set_session_state('rsi_overbought', 75)
        set_session_state('rsi_oversold', 25)
        config2 = get_backtest_config()
        assert config2['strategy_type'] == 'RSI Strategy'
        assert config2['strategy_params']['period'] == 21
        assert config2['strategy_params']['overbought'] == 75
        assert config2['strategy_params']['oversold'] == 25
        set_session_state('strategy_type', 'Buy and Hold')
        config3 = get_backtest_config()
        assert config3['strategy_type'] == 'Buy and Hold'
        assert config3['strategy_params'] == {}
    
    def test_results_storage_workflow(self):
        initialize_session_state()
        mock_results = {
            'equity_curve': pd.Series([10000, 10100, 10050, 10200], 
                                    index=pd.date_range('2024-01-01', periods=4)),
            'trades': [
                {'timestamp': '2024-01-02', 'type': 'buy', 'price': 100.0, 'size': 100},
                {'timestamp': '2024-01-03', 'type': 'sell', 'price': 102.0, 'size': 100}
            ],
            'metrics': {
                'total_return': 0.02,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.05,
                'win_rate': 0.6
            },
            'status': 'completed'
        }
        
        set_session_state('backtest_results', mock_results)
        stored_results = get_session_state('backtest_results')
        assert stored_results is not None
        assert stored_results['status'] == 'completed'
        assert 'equity_curve' in stored_results
        assert 'trades' in stored_results
        assert 'metrics' in stored_results
        metrics = stored_results['metrics']
        assert metrics['total_return'] == 0.02
        assert metrics['sharpe_ratio'] == 1.5
        assert metrics['max_drawdown'] == 0.05
        assert metrics['win_rate'] == 0.6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])