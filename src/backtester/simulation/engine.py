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
    price: float
    size: float
    value: float
    commission: float
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
        self.position_cost_basis = 0.0
    
    def run_simulation(self, data: pd.DataFrame, signals: pd.DataFrame) -> SimulationResult:
        if len(data) != len(signals):
            raise SimulationError("Data and signals must have the same length")
        
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            raise SimulationError("Data must contain OHLC columns")
        
        if 'signal' not in signals.columns:
            raise SimulationError("Signals DataFrame must contain a 'signal' column")
        
        if len(data) == 0:
            raise SimulationError("Data cannot be empty")

        self._reset()
        try:
            for idx, row in data.iterrows():
                self.current_price = row['Close']
                signal = signals.loc[idx, 'signal']
                if signal != SignalType.HOLD:
                    self._execute_trade(timestamp=idx, price=self.current_price, signal=signal)
                self.equity = self.cash + (self.position * self.current_price)
                if self._check_margin_call():
                    if self.position != 0:
                        close_signal = SignalType.SELL if self.position > 0 else SignalType.BUY
                        self._execute_trade(timestamp=idx, price=self.current_price, signal=close_signal, margin_call=True)
                        self.equity = self.cash + (self.position * self.current_price)
                
                self.equity_curve.append({
                    'timestamp': idx,
                    'equity': self.equity,
                    'cash': self.cash,
                    'position': self.position,
                    'position_value': self.position * self.current_price
                })
            
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            metrics = self._calculate_metrics(equity_df)
            return SimulationResult(equity_curve=equity_df, trades=self.trades, metrics=metrics)
        
        except Exception as e:
            raise SimulationError(f"Simulation failed: {str(e)}")
    
    def _execute_trade(self, timestamp: datetime, price: float, signal: SignalType, margin_call: bool = False) -> None:
        if signal == SignalType.BUY:
            self._execute_buy_trade(timestamp, price, margin_call)
        
        elif signal == SignalType.SELL:
            self._execute_sell_trade(timestamp, price, margin_call)
    
    def _execute_buy_trade(self, timestamp: datetime, price: float, margin_call: bool = False) -> None:
        if self.config.position_sizing == PositionSizing.FIXED_FRACTION:
            available_cash = self.cash * self.config.leverage
            target_value = available_cash * self.config.position_size
            size = target_value / price
        else:
            size = self.config.position_size
        
        if self.config.max_position_size is not None:
            size = min(size, self.config.max_position_size)
        
        max_affordable_value = self.cash * self.config.leverage
        max_affordable_size = max_affordable_value / price
        size = min(size, max_affordable_size)
        size = round(size, 6)
        if size <= 0:
            return  # Cant afford any shares
        
        value = price * size
        commission = self._calculate_commission(value)
        slippage_cost = self._calculate_slippage(price, size)
        total_cost = value + commission + slippage_cost
        required_cash = total_cost / self.config.leverage
        
        if required_cash > self.cash:
            available_for_margin = self.cash
            max_total_cost = available_for_margin * self.config.leverage
            max_value = max_total_cost / (1 + self.config.transaction_cost + self.config.slippage)
            size = max_value / price
            size = round(size, 6)
            if size <= 0:
                return  # Still cnat afford any shares
            
            value = price * size
            commission = self._calculate_commission(value)
            slippage_cost = self._calculate_slippage(price, size)
            total_cost = value + commission + slippage_cost
            required_cash = total_cost / self.config.leverage
        
        self.position += size
        self.cash -= required_cash
        self.position_cost_basis += total_cost
        
        self.trades.append(Trade(timestamp=timestamp, type=TradeType.BUY, price=price, size=size, value=value, commission=commission, slippage=slippage_cost, pnl=None))
    
    def _execute_sell_trade(self, timestamp: datetime, price: float, margin_call: bool = False) -> None:
        if self.position <= 0:
            return  # No position to sell
        
        if margin_call: # Liquidate whole position
            size = self.position
        elif self.config.position_sizing == PositionSizing.FIXED_FRACTION:
            size = self.position * self.config.position_size
        else:
            size = min(self.position, self.config.position_size)
        
        size = round(size, 6)
        if size <= 0:
            return  # Nothing to sell
        
        value = price * size
        commission = self._calculate_commission(value)
        slippage_cost = self._calculate_slippage(price, size)
        total_proceeds = value - commission - slippage_cost
        fraction_sold = size / self.position
        cost_of_sold_position = self.position_cost_basis * fraction_sold
        pnl = total_proceeds - cost_of_sold_position
        self.position -= size
        self.cash += total_proceeds
        self.position_cost_basis -= cost_of_sold_position
        self.trades.append(Trade(timestamp=timestamp, type=TradeType.SELL, price=price, size=size, value=value, commission=commission, slippage=slippage_cost, pnl=pnl))
    
    def _calculate_commission(self, value: float) -> float:
        return value * self.config.transaction_cost
    
    def _calculate_slippage(self, price: float, size: float) -> float:
        return price * size * self.config.slippage
    
    def _check_margin_call(self) -> bool:
        if self.position == 0:
            return False

        if self.position_cost_basis == 0:
            return False
            
        avg_cost_per_share = self.position_cost_basis / self.position
        original_position_value = self.position * avg_cost_per_share
        maintenance_margin = original_position_value * 0.3
        return self.equity < maintenance_margin
    
    def _calculate_metrics(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        metrics = {}
        initial_equity = self.config.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        metrics['total_return'] = (final_equity / initial_equity) - 1.0
        metrics['trade_count'] = len(self.trades)
        trades_with_pnl = [t for t in self.trades if t.pnl is not None]
        if trades_with_pnl:
            profitable_trades = sum(1 for t in trades_with_pnl if t.pnl > 0)
            total_trades_with_pnl = len(trades_with_pnl)
            metrics['win_rate'] = profitable_trades / total_trades_with_pnl
            gross_profit = sum(t.pnl for t in trades_with_pnl if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades_with_pnl if t.pnl < 0))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
        
        equity_df = equity_df.copy()  # Avoid modifying original
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['peak'] - equity_df['equity']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
        metrics['max_drawdown'] = equity_df['drawdown'].max()
        metrics['max_drawdown_pct'] = equity_df['drawdown_pct'].max()
        return metrics