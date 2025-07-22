import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from .strategy import Strategy, StrategyError, SignalType, StrategyRegistry

@StrategyRegistry.register
class RSIStrategy(Strategy):
    pass