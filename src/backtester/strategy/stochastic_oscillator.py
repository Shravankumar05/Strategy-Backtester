import pandas as pd
import numpy as np
from typing import Dict, Any
from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry

@StrategyRegistry.register
class StochasticOscillatorStrategy(Strategy):
    def __init__(self, k_period: int = 14, d_period: int = 3, oversold_level: float = 20.0, overbought_level: float = 80.0):
        if k_period < 2:
            raise StrategyError(f"K period must be >= 2, got {k_period}")
        
        if d_period < 1:
            raise StrategyError(f"D period must be >= 1, got {d_period}")
        
        if oversold_level <= 0 or oversold_level >= 100:
            raise StrategyError(f"Oversold level must be between 0 and 100, got {oversold_level}")
        
        if overbought_level <= 0 or overbought_level >= 100:
            raise StrategyError(f"Overbought level must be between 0 and 100, got {overbought_level}")
        
        if oversold_level >= overbought_level:
            raise StrategyError(f"Oversold level ({oversold_level}) must be less than overbought level ({overbought_level})")
        
        self._k_period = k_period
        self._d_period = d_period
        self._oversold_level = oversold_level
        self._overbought_level = overbought_level
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        
        min_required_data = max(self._k_period, self._d_period) + 1
        if len(data) < min_required_data:
            raise StrategyError(f"Not enough data for Stochastic Oscillator calculation. "
                              f"Need at least {min_required_data} data points, got {len(data)}.")
        
        try:
            signals = pd.DataFrame(index=data.index)
            
            # Calculate %K (Fast Stochastic)
            lowest_low = data['Low'].rolling(window=self._k_period).min()
            highest_high = data['High'].rolling(window=self._k_period).max()
            
            signals['%K'] = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
            
            # Calculate %D (Slow Stochastic) - moving average of %K
            signals['%D'] = signals['%K'].rolling(window=self._d_period).mean()
            
            # Initialize signals
            signals['signal'] = SignalType.HOLD
            signals['prev_%K'] = signals['%K'].shift(1)
            
            # Buy signal: %K crosses above oversold level from below
            buy_condition = (
                (signals['%K'] > self._oversold_level) & 
                (signals['prev_%K'] <= self._oversold_level)
            )
            signals.loc[buy_condition, 'signal'] = SignalType.BUY
            
            # Sell signal: %K crosses below overbought level from above
            sell_condition = (
                (signals['%K'] < self._overbought_level) &
                (signals['prev_%K'] >= self._overbought_level)
            )
            signals.loc[sell_condition, 'signal'] = SignalType.SELL
            
            # Clean up temporary columns
            signals = signals.drop(['prev_%K'], axis=1)
            signals['signal'] = signals['signal'].fillna(SignalType.HOLD)
            
            return signals
            
        except Exception as e:
            raise StrategyError(f"Failed to generate signals: {str(e)}")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'k_period': self._k_period,
            'd_period': self._d_period,
            'oversold_level': self._oversold_level,
            'overbought_level': self._overbought_level
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.validate_parameters(parameters)
        
        if 'k_period' in parameters:
            self._k_period = parameters['k_period']
        if 'd_period' in parameters:
            self._d_period = parameters['d_period']
        if 'oversold_level' in parameters:
            self._oversold_level = parameters['oversold_level']
        if 'overbought_level' in parameters:
            self._overbought_level = parameters['overbought_level']
    
    @property
    def name(self) -> str:
        return "Stochastic Oscillator"
    
    @property
    def description(self) -> str:
        return ("Generates buy and sell signals based on Stochastic Oscillator crossovers. "
                "Buy when %K crosses above oversold level (momentum building from oversold). "
                "Sell when %K crosses below overbought level (momentum weakening from overbought). "
                "The oscillator compares closing price to recent high-low range.")
    
    @property
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            'k_period': {
                'type': int,
                'default': 14,
                'min': 5,
                'max': 50,
                'description': 'Period for %K calculation (fast stochastic)'
            },
            'd_period': {
                'type': int,
                'default': 3,
                'min': 1,
                'max': 10,
                'description': 'Period for %D smoothing (slow stochastic)'
            },
            'oversold_level': {
                'type': float,
                'default': 20.0,
                'min': 5.0,
                'max': 40.0,
                'description': 'Level below which asset is considered oversold'
            },
            'overbought_level': {
                'type': float,
                'default': 80.0,
                'min': 60.0,
                'max': 95.0,
                'description': 'Level above which asset is considered overbought'
            }
        }
    
    def plot(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
            
            # Price chart
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1)
            
            # Buy/sell signals
            buy_signals = signals[signals['signal'] == SignalType.BUY]
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], 
                          marker='^', color='green', s=100, label='Buy Signal', zorder=5)
            
            sell_signals = signals[signals['signal'] == SignalType.SELL]
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], 
                          marker='v', color='red', s=100, label='Sell Signal', zorder=5)
            
            ax1.set_title(f'Stochastic Oscillator Strategy (K: {self._k_period}, D: {self._d_period})')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # %K chart
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(signals.index, signals['%K'], label='%K', color='blue', linewidth=1)
            ax2.axhline(y=self._overbought_level, color='r', linestyle='--', alpha=0.7, label=f'Overbought ({self._overbought_level})')
            ax2.axhline(y=self._oversold_level, color='g', linestyle='--', alpha=0.7, label=f'Oversold ({self._oversold_level})')
            ax2.fill_between(signals.index, 0, 100, alpha=0.1, color='gray')
            ax2.set_ylabel('%K')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # %D chart
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(signals.index, signals['%D'], label='%D', color='orange', linewidth=1)
            ax3.axhline(y=self._overbought_level, color='r', linestyle='--', alpha=0.7)
            ax3.axhline(y=self._oversold_level, color='g', linestyle='--', alpha=0.7)
            ax3.fill_between(signals.index, 0, 100, alpha=0.1, color='gray')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('%D')
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'.")