import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry

@StrategyRegistry.register
class RSIStrategy(Strategy):
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        # period = time for calculating RSI
        # overbought - predicting price will fall so sell
        # oversold - predicting price will rise so buy
        if period < 2:
            raise StrategyError(f"Period must be >= 2, got {period}")
        
        if overbought <= oversold:
            raise StrategyError(f"Overbought threshold ({overbought}) must be > oversold threshold ({oversold})")
        
        if overbought > 100:
            raise StrategyError(f"Overbought threshold must be <= 100, got {overbought}")
        
        if oversold < 0:
            raise StrategyError(f"Oversold threshold must be >= 0, got {oversold}")
        
        self._period = period
        self._oversold = oversold
        self._overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)

        if len(data) < self._period + 1:
            raise StrategyError(f"Not enough data for RSI calculation. "
                                f"Need at least {self._period + 1} data points, got {len(data)}.")
        
        try:
            signals = pd.DataFrame(index=data.index)
            signals['rsi'] = self._calculate_rsi(data['Close'])
            signals['signal'] = SignalType.HOLD # again default to hold
            signals['prev_rsi'] = signals['rsi'].shift(1)
            # Buy
            buy_signals = (signals['rsi'] < self._oversold) & (signals['prev_rsi'] >= self._oversold)
            signals.loc[buy_signals, 'signal'] = SignalType.BUY
            #Sell
            sell_signals = (signals['rsi'] > self._overbought) & (signals['prev_rsi'] <= self._overbought)
            signals.loc[sell_signals, 'signal'] = SignalType.SELL

            signals = signals.drop(['prev_rsi'], axis=1)
            signals['signal'] = signals['signal'].fillna(SignalType.HOLD) # default hold for NaNs
            return signals

        except Exception as e:
            raise StrategyError(f"Failed to generate signals: {str(e)}")
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()

        gain = delta.copy()
        gain[gain < 0] = 0
        avg_gain = gain.rolling(window=self._period).mean()

        loss = delta.copy()
        loss[loss > 0] = 0
        loss = -loss
        avg_loss = loss.rolling(window=self._period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self._period,
            'overbought': self._overbought,
            'oversold': self._oversold
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        # Get current values for parameters not being updated
        period = parameters.get('period', self._period)
        overbought = parameters.get('overbought', self._overbought)
        oversold = parameters.get('oversold', self._oversold)

        if overbought <= oversold:
            raise StrategyError(f"Overbought threshold ({overbought}) must be > oversold threshold ({oversold})")
        
        self.validate_parameters(parameters)
        
        if 'period' in parameters:
            self._period = parameters['period']
        if 'oversold' in parameters:
            self._oversold = parameters['oversold']
        if 'overbought' in parameters:
            self._overbought = parameters['overbought']
        
    @property
    def name(self) -> str:
        return "RSI Strategy"
    
    @property
    def description(self) -> str:
        return( "Generates buy/sell signals based on RSI."
                "Buys when RSI goes below oversold threshold."
                "Sell when RSI goes above overbought threshold")
    
    @property
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            'period': {
                'type': int,
                'default': 14,
                'min': 2,
                'max': 100,
                'description': 'Period for RSI calculation'
            },
            'overbought': {
                'type': int,
                'default': 70,
                'min': 50,
                'max': 100,
                'description': 'Threshold above which you will sell'
            },
            'oversold': {
                'type': int,
                'default': 30,
                'min': 0,
                'max': 50,
                'description': 'Threshold below which you will then buy'
            }
        }
    
    def plot(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 1, height_ratios=[2, 1])
            
            # Price chart
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
            
            # Buy/sell signals
            buy_signals = signals[signals['signal'] == SignalType.BUY]
            ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], marker='^', color='green', s=100, label='Buy Signal')
            sell_signals = signals[signals['signal'] == SignalType.SELL]
            ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], marker='v', color='red', s=100, label='Sell Signal')
            
            ax1.set_title(f'RSI Strategy ({self._period}, {self._overbought}/{self._oversold})')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # RSI chart
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(signals.index, signals['rsi'], label='RSI', color='purple')
            ax2.axhline(y=self._overbought, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=self._oversold, color='g', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'.")