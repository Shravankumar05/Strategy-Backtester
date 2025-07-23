import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from .config import SimulationConfig, PositionSizing, SimulationError
from ..strategy.strategy import SignalType

class TradeType(str, Enum):
    SELL = "sell"
    BUY = "buy"

class Trade(BaseModel):
    timestamp: datetime
    type: TradeType
    size: float
    value: float
    comission: float
    slippage: float
    pnl: Optional[float] = None

class SimulationResult(BaseModel):
    equity_curve: pd.DataFrame
    trades: List[Trade]
    metrics: Dict[str, float]

    class Config:
        arbitrary_types_allowed = True

class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._reset()
    
    def _reset(self):
        self.cash = self.config.initial_capital
        self.position = 0.0
        self.equity = self.config.initial_capital
        self.trades = []
        self.equity_curve = []
        self.current_price = None
    
    def run_simulation(self, data: pd.DataFrame, signals: pd.DataFrame) -> SimulationResult:
        pass