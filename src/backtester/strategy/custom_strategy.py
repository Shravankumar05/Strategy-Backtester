import pandas as pd
import numpy as np
import operator
from typing import Dict, Optional, Any, List, Tuple, Callable
from enum import Enum
from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry

class Operator(str, Enum):
    GREAT_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="

    @classmethod
    def get_operator_func(cls, op_str: str) -> Callable:
        op_map = {
            cls.GREAT_THAN: operator.gt,
            cls.LESS_THAN: operator.lt,
            cls.GREATER_EQUAL: operator.ge,
            cls.LESS_EQUAL: operator.le,
            cls.EQUAL: operator.eq,
            cls.NOT_EQUAL: operator.ne
        }

        if op_str not in op_map:
            raise StrategyError(f"Unsupported operator: {op_str}")
        
        return op_map[op_str]
    
class Condition:
    def __init__(self, indicator: str, operator_str: str, value: Any):
        self.indicator = indicator # Which indicator to trade on
        self.operator_str = operator_str # Which condition to trade it on
        self.value = value # What value in that condition to trade it on

        try:
            self.operator_func = Operator.get_operator_func(operator_str)
        except StrategyError as e:
            raise StrategyError(f"Invalid condition: {str(e)}")
        
    def evaluate(self, data: pd.DataFrame, current_idx: int) -> bool:
        try:
            if self.indicator not in data.columns:
                raise StrategyError(f"Indicator not found in data: {self.indicator}")
            
            indicator_value = data.loc[current_idx, self.indicator]
            
            if isinstance(self.value, str) and self.value in data.columns:
                comparison_value = data.loc[current_idx, self.value]
            else:
                comparison_value = self.value
            
            return self.operator_func(indicator_value, comparison_value)
        
        except StrategyError as e:
            raise
        except Exception as e:
            raise StrategyError(f"Error evaluating condition: {str(e)}")

class Rule:
    def __init__(self, conditions: List[Dict[str, Any]], action: str):
        if not conditions:
            raise StrategyError("Rule must have at least one condition")
        
        self.conditions = []

        for condition_dict in conditions:
            if not all(k in condition_dict for k in ['indicator', 'operator', 'value']):
                raise StrategyError("Each condition must have 'indicator', 'operator', and 'value' keys")
            
            condition = Condition(condition_dict['indicator'], condition_dict['operator'], condition_dict['value'])
            self.conditions.append(condition)
        
        action = action.lower()
        if action not in ['buy', 'sell', 'hold']:
            raise StrategyError(f"Invalid action: {action}. Must be 'buy', 'sell', or 'hold'")
            
        self.action = action
        self.signal_map = {
            'hold': SignalType.HOLD,
            'buy': SignalType.BUY,
            'sell': SignalType.SELL
        }
    
    def evaluate(self, data: pd.DataFrame, current_idx: int) -> SignalType:
        try:
            all_conditions_met = all(condition.evaluate(data, current_idx) for condition in self.conditions)

            if all_conditions_met:
                return self.signal_map[self.action]
            
            return SignalType.HOLD
        
        except Exception as e:
            raise StrategyError(f"Error evaluating rule: {str(e)}")
    
    def __str__(self) -> str:
        conditions_str = " AND ".join(str(condition) for condition in self.conditions)
        return f"IF {conditions_str} THEN {self.action.upper()}"

@StrategyRegistry.register
class CustomStrategy(Strategy):
    def __init__(self, rules: List[Dict[str, Any]] = None, indicators: Dict[str, Dict[str, Any]] = None):
        self._rules = []
        self._indicators = indicators or {}

        if rules:
            self.set_rules(rules)
    
    def set_rules(self, rules: List[Dict[str, Any]]) -> None:
        if not rules:
            raise StrategyError("At least one rule must be provided")
        
        parsed_rules = []
        for rule_dict in rules:
            if not all(k in rule_dict for k in ['conditions', 'action']):
                raise StrategyError("Each rule must have 'conditions' and 'action' keys")
            rule = Rule(rule_dict['conditions'], rule_dict['action'])
            parsed_rules.append(rule)
        self._rules = parsed_rules
    
    def set_indicators(self, indicators: Dict[str, Dict[str, Any]]) -> None:
        self._indicators = indicators or {}
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        try:
            for name, params in self._indicators.items():
                indicator_type = params.get('type', '').lower()

                # Simple moving average we meet again
                if indicator_type == 'sma':
                    window = params.get('window', 20)
                    source = params.get('source', 'Close')
                    result[name] = result[source].rolling(window=window).mean()
                
                # Exponential moving average
                elif indicator_type == 'ema':
                    window = params.get('window', 20)
                    source = params.get('source', 'Close')
                    result[name] = result[source].ewm(span=window, adjust=False).mean()
                
                # RSI we meet again
                elif indicator_type == 'rsi':
                    window = params.get('window', 14)
                    source = params.get('source', 'Close')
                    delta = result[source].diff()
                    
                    gain = delta.copy()
                    gain[gain<0] = 0
                    avg_gain = gain.rolling(window=window).mean()

                    loss = delta.copy()
                    loss[loss>0] = 0
                    loss = -loss
                    avg_loss = loss.rolling(window=window).mean()

                    rs = avg_gain/avg_loss
                    result[name] = 100 - ((100)/(rs+1))
                
                # Moving average convergence divergence
                elif indicator_type == 'macd':
                    fast_period = params.get('fast_period', 12)
                    slow_period = params.get('slow_period', 26)
                    signal_period = params.get('signal_period', 9)
                    source = params.get('source', 'Close')
                    fast_ema = result[source].ewm(span=fast_period, adjust=False).mean()
                    slow_ema = result[source].ewm(span=slow_period, adjust=False).mean()

                    macd_line = fast_ema - slow_ema
                    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                    histogram = macd_line - signal_line

                    result[f"{name}_line"] = macd_line
                    result[f"{name}_signal"] = signal_line
                    result[f"{name}_hist"] = histogram
                
                # Bollinger bands
                elif indicator_type == 'bollinger':
                    window = params.get('window', 20)
                    num_std = params.get('num_std', 2)
                    source = params.get('source', 'Close')

                    x = result[source].rolling(window=window)
                    sma = x.mean()
                    std = x.std()

                    result[f"{name}_middle"] = sma
                    result[f"{name}_upper"] = sma + (std * num_std)
                    result[f"{name}_lower"] = sma - (std * num_std)
                
                # Average true range
                elif indicator_type == 'atr':
                    window = params.get('window', 14)

                    high = result['High']
                    low = result['Low']
                    close = result['Close'].shift(1)

                    tr1 = high - low
                    tr2 = (high-close).abs()
                    tr3 = (low-close).abs()

                    tr = pd.concat([tr1, tr2, tr3], axis=1).mean(axis=1)
                    result[name] = tr.rolling(window=window).mean()
                
                else:
                    raise StrategyError(f"Unsupported indicator type: {indicator_type}")
            
            return result
        
        except Exception as e:
            raise StrategyError(f"Failed to calculate indicators: {str(e)}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        if not self._rules:
            raise StrategyError("No rules defined")
        
        try:
            data_with_indicators = self._calculate_indicators(data=data)
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = SignalType.HOLD

            for i, idx in enumerate(data_with_indicators.index):
                if data_with_indicators.iloc[i].isna().any():
                    continue

                for rule in self._rules:
                    signal = rule.evaluate(data_with_indicators, idx)
                    if signal != SignalType.HOLD:
                        signals.loc[idx, 'signal'] = signal
                        break
            
            for col in data_with_indicators.columns:
                if col not in data.columns:
                    signals[col] = data_with_indicators[col]
            
            return signals

        except Exception as e:
            raise StrategyError(f"Failed to generate signals: {str(e)}")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'rules': [
                {
                    'conditions': [
                        {
                            'indicator': condition.indicator,
                            'operator': condition.operator_str,
                            'value': condition.value
                        }
                        for condition in rule.conditions
                    ],
                    'action': rule.action
                }
                for rule in self._rules
            ],
            'indicators': self._indicators
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        if 'rules' in parameters:
            self.set_rules(parameters['rules'])
        
        if 'indicators' in parameters:
            self.set_indicators(parameters['indicators'])
    
    @property
    def name(self) -> str:
        return "Custom Strategy"
    
    @property
    def description(self) -> str:
        return( "Customizable strategy for users to define their own strats."
                "Users can do this through multiple rules based on predefined indicators."
                "Indicators available: sma, ema, rsi, macd, bollinger, atr")
    
    @property
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            'rules': {
                'type': list,
                'description': 'List of trading rules to evaluate'
            },
            'indicators': {
                'type': dict,
                'description': 'Dictionary of technical indicators to calculate'
            }
        }
    
    def plot(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            indicator_groups = {}
            for col in signals.columns:
                if col in ['signal', 'Open', 'High', 'Low', 'Close', 'Volume']:
                    continue
                    
                if col.startswith('macd_'):
                    base_name = 'MACD'
                    if base_name not in indicator_groups:
                        indicator_groups[base_name] = []
                    indicator_groups[base_name].append(col)
                    
                elif col.endswith('_upper') or col.endswith('_middle') or col.endswith('_lower'):
                    base_name = col.split('_')[0]
                    if base_name not in indicator_groups:
                        indicator_groups[base_name] = []
                    indicator_groups[base_name].append(col)
                    
                else:
                    indicator_groups[col] = [col]
            
            num_indicators = len(indicator_groups)
            fig = plt.figure(figsize=(12, 8 + 2 * num_indicators))
            gs = GridSpec(1 + num_indicators, 1, height_ratios=[2] + [1] * num_indicators)
            
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
            for col in signals.columns:
                if col.startswith('sma_') or col.startswith('ema_'):
                    ax1.plot(signals.index, signals[col], label=col, alpha=0.7)
            
            buy_signals = signals[signals['signal'] == SignalType.BUY]
            ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], marker='^', color='green', s=100, label='Buy Signal')
            sell_signals = signals[signals['signal'] == SignalType.SELL]
            ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], marker='v', color='red', s=100, label='Sell Signal')
            
            ax1.set_title('Custom Strategy')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            i = 0
            for group_name, columns in indicator_groups.items():
                if any(col.startswith('sma_') or col.startswith('ema_') for col in columns):
                    continue
                    
                if all(col in ax1.get_legend_handles_labels()[1] for col in columns):
                    continue
                    
                ax = fig.add_subplot(gs[i+1], sharex=ax1)
                
                if group_name == 'MACD':
                    macd_line = next((col for col in columns if col.endswith('_line')), None)
                    signal_line = next((col for col in columns if col.endswith('_signal')), None)
                    histogram = next((col for col in columns if col.endswith('_hist')), None)
                    
                    if macd_line and signal_line:
                        ax.plot(signals.index, signals[macd_line], label='MACD Line')
                        ax.plot(signals.index, signals[signal_line], label='Signal Line')
                    
                    if histogram:
                        ax.bar(signals.index, signals[histogram], label='Histogram', color=['green' if x > 0 else 'red' for x in signals[histogram]])
                    
                elif any(col.endswith('_upper') or col.endswith('_lower') for col in columns):
                    upper = next((col for col in columns if col.endswith('_upper')), None)
                    middle = next((col for col in columns if col.endswith('_middle')), None)
                    lower = next((col for col in columns if col.endswith('_lower')), None)
                    
                    if upper and middle and lower:
                        ax.plot(signals.index, signals[middle], label='Middle Band')
                        ax.plot(signals.index, signals[upper], label='Upper Band')
                        ax.plot(signals.index, signals[lower], label='Lower Band')
                        ax.fill_between(signals.index, signals[upper], signals[lower], alpha=0.1, color='blue')
                        ax.plot(data.index, data['Close'], label='Close', alpha=0.5)
                    
                else:
                    for col in columns:
                        ax.plot(signals.index, signals[col], label=col)
                
                ax.set_ylabel(group_name)
                ax.grid(True)
                ax.legend()
                i += 1
            
            if i > 0:
                ax.set_xlabel('Date')
            else:
                ax1.set_xlabel('Date')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'.")