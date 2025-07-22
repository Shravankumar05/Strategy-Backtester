import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry

@StrategyRegistry.register
class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise StrategyError(f"Short window ({short_window}) must be less than the long window ({long_window})")
        
        self._short_window = short_window
        self._long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        
        if len(data) < self._long_window:
            raise StrategyError(f"Not enough data for long window ({self._long_window}) need at least {self._long_window} points, got {len(data)}.")
        
        try:
            signals = pd.DataFrame(index=data.index)
            signals['short_ma'] = data['Close'].rolling(window=self._short_window).mean()
            signals['long_ma'] = data['Close'].rolling(window=self._long_window).mean()
            signals['signal'] = SignalType.HOLD # Default is hold

            signals['prev_short_above_long'] = signals['short_ma'].shift(1) > signals['long_ma'].shift(1)
            signals['short_above_long'] = signals['short_ma'] > signals['long_ma']
            
            # Buy signal - previous day short was below or equal to long, but today it's above
            buy_signals = (~signals['prev_short_above_long']) & signals['short_above_long']
            signals.loc[buy_signals, 'signal'] = SignalType.BUY
            
            # Sell signal - previous day short was above or equal to long, but today it's below
            sell_signals = signals['prev_short_above_long'] & (~signals['short_above_long'])
            signals.loc[sell_signals, 'signal'] = SignalType.SELL
            
            signals = signals.drop(['prev_short_above_long', 'short_above_long'], axis=1)
            signals['signal'] = signals['signal'].fillna(SignalType.HOLD) # Nan is hold
            return signals
            
        except Exception as e:
            raise StrategyError(f"Failed to generate signals: {str(e)}")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'short_window': self._short_window,
            'long_window': self._long_window
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.validate_parameters(parameters)
        
        short_window = parameters.get('short_window', self._short_window)
        long_window = parameters.get('long_window', self._long_window)
        
        if short_window >= long_window:
            raise StrategyError(
                f"Short window ({short_window}) must be less than "
                f"long window ({long_window})"
            )
        
        if 'short_window' in parameters:
            self._short_window = parameters['short_window']
        
        if 'long_window' in parameters:
            self._long_window = parameters['long_window']
    
    @property
    def name(self) -> str:
        return "Moving Average Crossover"
    
    @property
    def description(self) -> str:
        return ("Generates buy and sell signals based on the crossover of short-term and long-term moving averages."
                "A buy signal is generated when the short-term MA crosses above the long-term MA."
                "sell signal is generated when the short-term MA crosses below the long-term MA.")
    
    @property
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            'short_window': {
                'type': int,
                'default': 20,
                'min': 2,
                'max': 200,
                'description': 'Window size for the short-term moving average'
            },
            'long_window': {
                'type': int,
                'default': 50,
                'min': 5,
                'max': 500,
                'description': 'Window size for the long-term moving average'
            }
        }
    
    def plot(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
            
            # Moving averages
            ax.plot(signals.index, signals['short_ma'], label=f'{self._short_window}-day MA', alpha=0.8)
            ax.plot(signals.index, signals['long_ma'], label=f'{self._long_window}-day MA', alpha=0.8)
            
            # Buy signals
            buy_signals = signals[signals['signal'] == SignalType.BUY]
            ax.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], marker='^', color='green', s=100, label='Buy Signal')
            
            # Sell signals
            sell_signals = signals[signals['signal'] == SignalType.SELL]
            ax.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], marker='v', color='red', s=100, label='Sell Signal')
            
            ax.set_title(f'Moving Average Crossover Strategy ({self._short_window}/{self._long_window})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'.")