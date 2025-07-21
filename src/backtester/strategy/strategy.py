import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, Any, List, Optional, Tuple

class SignalType(IntEnum):
    # Enum the trade signals
    SELL = -1
    HOLD = 0
    BUY = 1

class StrategyError(Exception):
    pass

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # User puts the technique to generate signals in here
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        # Define strategy params here
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        # Set params of the strategy here
        pass
    

    # Info so name and short description of strategy
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def parameter_info(self) -> Dict[str, Dict[str, Any]]:
        # Info on each param added here
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        # Check params have been fed as declared
        param_info = self.parameter_info
        
        for name, value in parameters.items():
            if name not in param_info:
                raise StrategyError(f"Unknown parameter: {name}")
            
            info = param_info[name]
            expected_type = info.get('type')
            
            # Type validation
            if expected_type and not isinstance(value, expected_type):
                raise StrategyError(f"Parameter '{name}' must be of type {expected_type.__name__}, "
                    f"got {type(value).__name__}")
            
            # Range validation
            if 'min' in info and value < info['min']:
                raise StrategyError(f"Parameter '{name}' must be >= {info['min']}, got {value}")
            if 'max' in info and value > info['max']:
                raise StrategyError(f"Parameter '{name}' must be <= {info['max']}, got {value}")
            
            # Options validation
            if 'options' in info and value not in info['options']:
                raise StrategyError(f"Parameter '{name}' must be one of {info['options']}, got {value}")
    
    def validate_data(self, data: pd.DataFrame) -> None:
        # One more check on the data we use
        if not isinstance(data, pd.DataFrame):
            raise StrategyError("Input data must be a pandas DataFrame")
        
        if data.empty:
            raise StrategyError("Input data is empty")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise StrategyError(f"Input data is missing required columns: {missing_columns}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise StrategyError("Input data index must be a DatetimeIndex")

class StrategyRegistry:
    _strategies: Dict[str, type] = {}
    
    @classmethod
    def register(cls, strategy_class: type) -> type:
        if not issubclass(strategy_class, Strategy):
            raise StrategyError(f"{strategy_class.__name__} is in Strategy")
        
        cls._strategies[strategy_class.__name__] = strategy_class
        return strategy_class
    
    @classmethod
    def get_strategy_class(cls, name: str) -> type:
        # Check if strategy exists
        if name not in cls._strategies:
            raise StrategyError(f"Strategy not found: {name}")
        
        return cls._strategies[name]
    
    @classmethod
    def create_strategy(cls, name: str, parameters: Optional[Dict[str, Any]] = None) -> Strategy:
        # Create a strategy by name and params
        strategy_class = cls.get_strategy_class(name)
        
        try:
            strategy = strategy_class()
            if parameters:
                strategy.set_parameters(parameters)
            return strategy

        except Exception as e:
            raise StrategyError(f"Failed to create strategy {name}: {str(e)}")
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, Dict[str, Any]]:
        # Like a /all gives a list of all strategies info
        info = {}
        
        for name, strategy_class in cls._strategies.items(): # Iterate through all strategies
            try:
                strategy = strategy_class()
                info[name] = {
                    'description': strategy.description,
                    'parameters': strategy.parameter_info
                }
            except Exception as e:
                continue
        
        return info