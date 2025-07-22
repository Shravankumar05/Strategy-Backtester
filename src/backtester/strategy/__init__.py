# Provides the interface and implementations for trading strategies.

from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry
from .ma_crossover import MovingAverageCrossoverStrategy

__all__ = ['Strategy', 'StrategyError', 'SignalType', 'StrategyRegistry', 'MovingAverageCrossoverStrategy']