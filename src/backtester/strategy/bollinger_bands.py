import pandas as pd
import numpy as np
from typing import Dict, Any
from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry

@StrategyRegistry.register
class BollingerBandsStrategy(Strategy):
    def __init__(self, period: int = 20, std_multiplier: float = 2.0, buy_threshold: float = 0.0, sell_threshold: float = 0.0):
        if period < 2:
            raise StrategyError(f"Period must be >= 2, got {period}")
        
        if std_multiplier <= 0:
            raise StrategyError(f"Standard deviation multiplier must be positive, got {std_multiplier}")
        
        if buy_threshold < 0 or buy_threshold > 1:
            raise StrategyError(f"Buy threshold must be between 0 and 1, got {buy_threshold}")
        
        if sell_threshold < 0 or sell_threshold > 1:
            raise StrategyError(f"Sell threshold must be between 0 and 1, got {sell_threshold}")
        
        self._period = period
        self._std_multiplier = std_multiplier
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        
        if len(data) < self._period:
            raise StrategyError(f"Not enough data for Bollinger Bands calculation. "
                              f"Need at least {self._period} data points, got {len(data)}.")
        
        try:
            signals = pd.DataFrame(index=data.index)
            
            signals['middle_band'] = data['Close'].rolling(window=self._period).mean()
            rolling_std = data['Close'].rolling(window=self._period).std()
            signals['upper_band'] = signals['middle_band'] + (rolling_std * self._std_multiplier)
            signals['lower_band'] = signals['middle_band'] - (rolling_std * self._std_multiplier)
            band_width = signals['upper_band'] - signals['lower_band']
            signals['band_position'] = (data['Close'] - signals['lower_band']) / band_width
            signals['signal'] = SignalType.HOLD
            signals['prev_band_position'] = signals['band_position'].shift(1)
            
            # Buy signal- teh price touches or moves below lower band + threshold
            buy_condition = (
                (signals['band_position'] <= self._buy_threshold) & 
                (signals['prev_band_position'] > self._buy_threshold)
            )
            signals.loc[buy_condition, 'signal'] = SignalType.BUY
            
            # Sell signal - the price touches or moves above upper band - threshold  
            sell_condition = (
                (signals['band_position'] >= (1 - self._sell_threshold)) &
                (signals['prev_band_position'] < (1 - self._sell_threshold))
            )
            signals.loc[sell_condition, 'signal'] = SignalType.SELL
            
            signals = signals.drop(['prev_band_position'], axis=1)
            signals['signal'] = signals['signal'].fillna(SignalType.HOLD)
            return signals
            
        except Exception as e:
            raise StrategyError(f"Failed to generate signals: {str(e)}")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self._period,
            'std_multiplier': self._std_multiplier,
            'buy_threshold': self._buy_threshold,
            'sell_threshold': self._sell_threshold
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.validate_parameters(parameters)
        
        if 'period' in parameters:
            self._period = parameters['period']
        if 'std_multiplier' in parameters:
            self._std_multiplier = parameters['std_multiplier']
        if 'buy_threshold' in parameters:
            self._buy_threshold = parameters['buy_threshold']
        if 'sell_threshold' in parameters:
            self._sell_threshold = parameters['sell_threshold']
    
    @property
    def name(self) -> str:
        return "Bollinger Bands"
    
    @property
    def description(self) -> str:
        return ("Generates buy and sell signals based on Bollinger Bands. "
                "Buy when price touches or moves below the lower band (oversold). "
                "Sell when price touches or moves above the upper band (overbought). "
                "Bands are calculated using a moving average with standard deviation envelope.")
    
    @property
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            'period': {
                'type': int,
                'default': 20,
                'min': 5,
                'max': 100,
                'description': 'Period for moving average and standard deviation calculation'
            },
            'std_multiplier': {
                'type': float,
                'default': 2.0,
                'min': 0.5,
                'max': 4.0,
                'description': 'Standard deviation multiplier for band width'
            },
            'buy_threshold': {
                'type': float,
                'default': 0.0,
                'min': 0.0,
                'max': 0.3,
                'description': 'Distance from lower band to trigger buy (0 = touch band)'
            },
            'sell_threshold': {
                'type': float,
                'default': 0.0,
                'min': 0.0,
                'max': 0.3,
                'description': 'Distance from upper band to trigger sell (0 = touch band)'
            }
        }
    
    def plot(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            ax.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1)
            ax.plot(signals.index, signals['middle_band'], label=f'{self._period}-day MA', color='blue', alpha=0.7)
            ax.plot(signals.index, signals['upper_band'], label='Upper Band', color='red', alpha=0.7)
            ax.plot(signals.index, signals['lower_band'], label='Lower Band', color='green', alpha=0.7)
            ax.fill_between(signals.index, signals['lower_band'], signals['upper_band'], 
                          alpha=0.1, color='gray', label='Bollinger Bands')
            buy_signals = signals[signals['signal'] == SignalType.BUY]
            if not buy_signals.empty:
                ax.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], 
                          marker='^', color='green', s=100, label='Buy Signal', zorder=5)
            sell_signals = signals[signals['signal'] == SignalType.SELL]
            if not sell_signals.empty:
                ax.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], 
                          marker='v', color='red', s=100, label='Sell Signal', zorder=5)
            ax.set_title(f'Bollinger Bands Strategy (Period: {self._period}, Std: {self._std_multiplier})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'.")