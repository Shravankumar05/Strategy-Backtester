# Provides the interface and implementations for trading strategies.

from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry
from .ma_crossover import MovingAverageCrossoverStrategy
from .rsi_strategy import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .stochastic_oscillator import StochasticOscillatorStrategy
from .custom_strategy import CustomStrategy

__all__ = ['Strategy', 'StrategyError', 'SignalType', 'StrategyRegistry', 'MovingAverageCrossoverStrategy', 'RSIStrategy', 'BollingerBandsStrategy', 'StochasticOscillatorStrategy', 'CustomStrategy']