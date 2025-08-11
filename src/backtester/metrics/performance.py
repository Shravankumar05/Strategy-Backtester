import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from datetime import datetime

class PerformanceMetrics:
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        if len(equity_curve) < 2:
            return pd.Series(dtype=float)
        returns = equity_curve.pct_change()
        returns = returns.dropna()
        return returns
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        
        period_risk_free_rate = risk_free_rate / periods_per_year
        excess_returns = returns - period_risk_free_rate
        mean_excess_return = excess_returns.mean()
        returns_std = returns.std()
        
        if isinstance(mean_excess_return, pd.Series):
            mean_excess_return = mean_excess_return.iloc[0] if len(mean_excess_return) == 1 else mean_excess_return.values[0]
        if isinstance(returns_std, pd.Series):
            returns_std = returns_std.iloc[0] if len(returns_std) == 1 else returns_std.values[0]
        
        if np.isclose(returns_std, 0.0) or np.isnan(returns_std):
            return 0.0 if np.isclose(mean_excess_return, 0.0) else np.inf
        
        sharpe_ratio = mean_excess_return / returns_std
        annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
        return float(annualized_sharpe)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float=0.0, periods_per_year=252) -> float:
        if len(returns) == 0:
            return 0.0
        
        period_risk_free_rate = risk_free_rate / periods_per_year
        excess_returns = returns - period_risk_free_rate
        mean_excess_return = excess_returns.mean()
        
        if isinstance(mean_excess_return, pd.Series):
            mean_excess_return = mean_excess_return.iloc[0] if len(mean_excess_return) == 1 else mean_excess_return.values[0]
            
        downside_returns = returns[returns < period_risk_free_rate]
        if len(downside_returns) == 0:
            return np.inf if mean_excess_return > 0 else 0.0
        
        downside_deviations = np.sqrt(((downside_returns - period_risk_free_rate) ** 2).mean())
        
        if isinstance(downside_deviations, pd.Series):
            downside_deviations = downside_deviations.iloc[0] if len(downside_deviations) == 1 else downside_deviations.values[0]
        if np.isclose(downside_deviations, 0.0) or np.isnan(downside_deviations):
            return 0.0 if np.isclose(mean_excess_return, 0.0) else np.inf
        
        sortino_ratio = mean_excess_return / downside_deviations
        annualized_sortino = sortino_ratio * np.sqrt(periods_per_year)
        return float(annualized_sortino)
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        
        period_volatility = returns.std()
        
        if isinstance(period_volatility, pd.Series):
            period_volatility = period_volatility.iloc[0] if len(period_volatility) == 1 else period_volatility.values[0]
            
        annualized_volatility = period_volatility * np.sqrt(periods_per_year)
        return float(annualized_volatility)
    
    @staticmethod
    def calculate_downside_volatility(returns: pd.Series, threshold: float = 0.0, periods_per_year: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < threshold]
        if len(downside_returns) == 0:
            return 0.0
        
        period_downside_vol = downside_returns.std()
        if isinstance(period_downside_vol, pd.Series):
            period_downside_vol = period_downside_vol.iloc[0] if len(period_downside_vol) == 1 else period_downside_vol.values[0]
            
        annualized_downside_vol = period_downside_vol * np.sqrt(periods_per_year)
        return float(annualized_downside_vol)
    
    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        if isinstance(initial_value, pd.Series):
            initial_value = initial_value.iloc[0] if len(initial_value) == 1 else initial_value.values[0]
        if isinstance(final_value, pd.Series):
            final_value = final_value.iloc[0] if len(final_value) == 1 else final_value.values[0]
        if initial_value <= 0:
            return 0.0
        
        total_return = (final_value / initial_value) - 1
        return float(total_return)
    
    @staticmethod
    def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        if isinstance(initial_value, pd.Series):
            initial_value = initial_value.iloc[0] if len(initial_value) == 1 else initial_value.values[0]
        if isinstance(final_value, pd.Series):
            final_value = final_value.iloc[0] if len(final_value) == 1 else final_value.values[0]
        if initial_value <= 0:
            return 0.0
        periods_elapsed = len(equity_curve) - 1
        years = periods_elapsed / periods_per_year
        if years <= 0:
            return 0.0
        if final_value <= 0:
            return -1.0
        
        cagr = (final_value / initial_value) ** (1 / years) - 1
        return float(cagr)
    
    @staticmethod
    def calculate_annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        return PerformanceMetrics.calculate_cagr(equity_curve, periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        max_drawdown = drawdown.max()
        if isinstance(max_drawdown, pd.Series):
            max_drawdown = max_drawdown.iloc[0] if len(max_drawdown) == 1 else max_drawdown.values[0]
        return float(max_drawdown)
    
    @staticmethod
    def calculate_max_drawdown_duration(equity_curve: pd.Series) -> int:
        if len(equity_curve) < 2:
            return 0
        
        running_max = equity_curve.cummax()
        at_high = equity_curve >= running_max
        max_duration = 0
        current_duration = 0
        for is_at_high in at_high:
            if is_at_high:
                max_duration = max(max_duration, current_duration)
                current_duration = 0
            else:
                current_duration += 1
        
        max_duration = max(max_duration, current_duration)
        return int(max_duration)
    
    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        if len(equity_curve) == 0:
            return pd.Series(dtype=float)
        if len(equity_curve) == 1:
            return pd.Series([0.0], index=equity_curve.index)
        
        running_max = equity_curve.cummax()
        drawdown_series = (running_max - equity_curve) / running_max
        return drawdown_series
    
    @staticmethod
    def calculate_average_drawdown(equity_curve: pd.Series) -> float:
        drawdown_series = PerformanceMetrics.calculate_drawdown_series(equity_curve)
        
        if len(drawdown_series) == 0:
            return 0.0
        
        drawdown_periods = drawdown_series[drawdown_series > 0]
        if len(drawdown_periods) == 0:
            return 0.0
        
        return float(drawdown_periods.mean())
    
    @staticmethod
    def calculate_all_drawdown_metrics(equity_curve: pd.Series) -> Dict[str, Union[float, int]]:
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration': 0,
                'average_drawdown': 0.0,
                'drawdown_periods': 0,
                'recovery_periods': 0
            }
        
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        max_drawdown_duration = PerformanceMetrics.calculate_max_drawdown_duration(equity_curve)
        average_drawdown = PerformanceMetrics.calculate_average_drawdown(equity_curve)
        drawdown_series = PerformanceMetrics.calculate_drawdown_series(equity_curve)
        drawdown_periods = int((drawdown_series > 0).sum())
        running_max = equity_curve.cummax()
        at_high = equity_curve >= running_max
        recovery_periods = int((~at_high[:-1] & at_high[1:]).sum())
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,  # As percentage
            'max_drawdown_duration': max_drawdown_duration,
            'average_drawdown': average_drawdown,
            'drawdown_periods': drawdown_periods,
            'recovery_periods': recovery_periods
        }
    
    @staticmethod
    def calculate_all_absolute_return_metrics(equity_curve: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
        if len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'cagr': 0.0,
                'annualized_return': 0.0,
                'periods_elapsed': 0,
                'years_elapsed': 0.0
            }
        
        total_return = PerformanceMetrics.calculate_total_return(equity_curve)
        cagr = PerformanceMetrics.calculate_cagr(equity_curve, periods_per_year)
        periods_elapsed = len(equity_curve) - 1
        years_elapsed = periods_elapsed / periods_per_year
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annualized_return': cagr,
            'periods_elapsed': periods_elapsed,
            'years_elapsed': years_elapsed
        }
    
    @staticmethod
    def calculate_win_rate(trades: List) -> float:
        if not trades:
            return 0.0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        if not trades_with_pnl:
            return 0.0
        profitable_trades = sum(1 for trade in trades_with_pnl if trade.pnl > 0)
        win_rate = profitable_trades / len(trades_with_pnl)
        return float(win_rate)
    
    @staticmethod
    def calculate_profit_factor(trades: List) -> float:
        if not trades:
            return 0.0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        if not trades_with_pnl:
            return 0.0
        
        gross_profit = sum(trade.pnl for trade in trades_with_pnl if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades_with_pnl if trade.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        if gross_profit == 0:
            return 0.0
        
        profit_factor = gross_profit / gross_loss
        return float(profit_factor)
    
    @staticmethod
    def calculate_trade_count(trades: List) -> int:
        return len(trades) if trades else 0
    
    @staticmethod
    def calculate_average_trade_pnl(trades: List) -> float:
        if not trades:
            return 0.0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        if not trades_with_pnl:
            return 0.0
        
        total_pnl = sum(trade.pnl for trade in trades_with_pnl)
        return float(total_pnl / len(trades_with_pnl))
    
    @staticmethod
    def calculate_largest_win(trades: List) -> float:
        if not trades:
            return 0.0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        winning_trades = [trade.pnl for trade in trades_with_pnl if trade.pnl > 0]
        return float(max(winning_trades)) if winning_trades else 0.0
    
    @staticmethod
    def calculate_largest_loss(trades: List) -> float:
        if not trades:
            return 0.0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        losing_trades = [trade.pnl for trade in trades_with_pnl if trade.pnl < 0]
        return float(abs(min(losing_trades))) if losing_trades else 0.0
    
    @staticmethod
    def calculate_consecutive_wins(trades: List) -> int:
        if not trades:
            return 0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        if not trades_with_pnl:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        for trade in trades_with_pnl:
            if trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive
    
    @staticmethod
    def calculate_consecutive_losses(trades: List) -> int:
        if not trades:
            return 0
        
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        if not trades_with_pnl:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        for trade in trades_with_pnl:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    @staticmethod
    def calculate_all_trade_statistics(trades: List) -> Dict[str, Union[float, int]]:
        if not trades:
            return {
                'trade_count': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_trade_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'total_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0
            }
        
        trade_count = PerformanceMetrics.calculate_trade_count(trades)
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades)
        average_trade_pnl = PerformanceMetrics.calculate_average_trade_pnl(trades)
        largest_win = PerformanceMetrics.calculate_largest_win(trades)
        largest_loss = PerformanceMetrics.calculate_largest_loss(trades)
        consecutive_wins = PerformanceMetrics.calculate_consecutive_wins(trades)
        consecutive_losses = PerformanceMetrics.calculate_consecutive_losses(trades)
        trades_with_pnl = [trade for trade in trades if hasattr(trade, 'pnl') and trade.pnl is not None]
        
        if trades_with_pnl:
            total_pnl = sum(trade.pnl for trade in trades_with_pnl)
            gross_profit = sum(trade.pnl for trade in trades_with_pnl if trade.pnl > 0)
            gross_loss = abs(sum(trade.pnl for trade in trades_with_pnl if trade.pnl < 0))
        else:
            total_pnl = 0.0
            gross_profit = 0.0
            gross_loss = 0.0
        
        return {
            'trade_count': trade_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_trade_pnl': average_trade_pnl,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    @staticmethod
    def calculate_average_leverage(equity_curve: pd.DataFrame, initial_capital: float) -> float:
        if len(equity_curve) == 0:
            return 0.0
        
        required_columns = ['equity', 'position_value']
        if not all(col in equity_curve.columns for col in required_columns):
            return 0.0
        
        leverage_series = []
        for _, row in equity_curve.iterrows():
            equity = row['equity']
            position_value = abs(row['position_value'])  # Use absolute value
            
            if equity > 0:
                leverage = position_value / equity
                leverage_series.append(leverage)
        
        if not leverage_series:
            return 0.0
        return float(sum(leverage_series) / len(leverage_series))
    
    @staticmethod
    def calculate_maximum_leverage(equity_curve: pd.DataFrame, initial_capital: float) -> float:
        if len(equity_curve) == 0:
            return 0.0
        
        required_columns = ['equity', 'position_value']
        if not all(col in equity_curve.columns for col in required_columns):
            return 0.0
        
        max_leverage = 0.0
        for _, row in equity_curve.iterrows():
            equity = row['equity']
            position_value = abs(row['position_value'])
            
            if equity > 0:
                leverage = position_value / equity
                max_leverage = max(max_leverage, leverage)
        return float(max_leverage)
    
    @staticmethod
    def calculate_leverage_utilization(equity_curve: pd.DataFrame, max_allowed_leverage: float) -> float:
        if max_allowed_leverage <= 0:
            return 0.0
        avg_leverage = PerformanceMetrics.calculate_average_leverage(equity_curve, 0.0)  # initial_capital not used in calculation
        return float(min(avg_leverage / max_allowed_leverage, 1.0))
    
    @staticmethod
    def calculate_leverage_efficiency(equity_curve: pd.DataFrame, returns: pd.Series) -> float:
        if len(equity_curve) == 0 or len(returns) == 0:
            return 0.0
        
        avg_leverage = PerformanceMetrics.calculate_average_leverage(equity_curve, 0.0)
        if avg_leverage <= 0:
            return 0.0
        
        if len(equity_curve) >= 2:
            initial_equity = equity_curve['equity'].iloc[0]
            final_equity = equity_curve['equity'].iloc[-1]
            
            if initial_equity > 0:
                total_return = (final_equity / initial_equity) - 1
                return float(total_return / avg_leverage)
        
        return 0.0
    
    @staticmethod
    def calculate_all_leverage_metrics(equity_curve: pd.DataFrame, initial_capital: float, max_allowed_leverage: float = None) -> Dict[str, float]:
        if len(equity_curve) == 0:
            return {
                'average_leverage': 0.0,
                'maximum_leverage': 0.0,
                'leverage_utilization': 0.0,
                'leverage_efficiency': 0.0,
                'periods_with_leverage': 0,
                'periods_without_position': 0
            }
        
        avg_leverage = PerformanceMetrics.calculate_average_leverage(equity_curve, initial_capital)
        max_leverage = PerformanceMetrics.calculate_maximum_leverage(equity_curve, initial_capital)
        leverage_utilization = 0.0
        if max_allowed_leverage and max_allowed_leverage > 0:
            leverage_utilization = PerformanceMetrics.calculate_leverage_utilization(equity_curve, max_allowed_leverage)
        
        returns = PerformanceMetrics.calculate_returns(equity_curve['equity'])
        leverage_efficiency = PerformanceMetrics.calculate_leverage_efficiency(equity_curve, returns)
        periods_with_leverage = 0
        periods_without_position = 0
        required_columns = ['equity', 'position_value']
        if all(col in equity_curve.columns for col in required_columns):
            for _, row in equity_curve.iterrows():
                equity = row['equity']
                position_value = abs(row['position_value'])
                
                if position_value == 0:
                    periods_without_position += 1
                elif equity > 0:
                    leverage = position_value / equity
                    if leverage > 1.0:
                        periods_with_leverage += 1
        
        return {
            'average_leverage': avg_leverage,
            'maximum_leverage': max_leverage,
            'leverage_utilization': leverage_utilization,
            'leverage_efficiency': leverage_efficiency,
            'periods_with_leverage': periods_with_leverage,
            'periods_without_position': periods_without_position
        }
    
    @staticmethod
    def calculate_all_risk_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'volatility': 0.0,
                'downside_volatility': 0.0,
                'total_return_annualized': 0.0,
                'excess_return_annualized': 0.0
            }
        
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        volatility = PerformanceMetrics.calculate_volatility(returns, periods_per_year)
        downside_volatility = PerformanceMetrics.calculate_downside_volatility(returns, 0.0, periods_per_year)
        if len(equity_curve) >= 2:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            periods_elapsed = len(returns)
            annualization_factor = periods_per_year / periods_elapsed
            total_return_annualized = (1 + total_return) ** annualization_factor - 1
            excess_return_annualized = total_return_annualized - risk_free_rate
        else:
            total_return_annualized = 0.0
            excess_return_annualized = 0.0
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'total_return_annualized': total_return_annualized,
            'excess_return_annualized': excess_return_annualized
        }
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        return float(var)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        var = PerformanceMetrics.calculate_var(returns, confidence_level)
        tail_returns = returns[returns < -var]
        if len(tail_returns) == 0:
            return 0.0
        cvar = -tail_returns.mean()
        return float(cvar)

    @staticmethod
    def calculate_all_metrics(equity_curve: pd.Series, trades: List, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
        metrics = {}
        
        if len(equity_curve) >= 2:
            returns = PerformanceMetrics.calculate_returns(equity_curve)
            
            metrics['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
            metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
            metrics['volatility'] = PerformanceMetrics.calculate_volatility(returns, periods_per_year)
            metrics['total_return'] = PerformanceMetrics.calculate_total_return(equity_curve)
            metrics['cagr'] = PerformanceMetrics.calculate_cagr(equity_curve, periods_per_year)
            metrics['max_drawdown'] = PerformanceMetrics.calculate_max_drawdown(equity_curve)
            metrics['max_drawdown_duration'] = PerformanceMetrics.calculate_max_drawdown_duration(equity_curve)
            metrics['var_95'] = PerformanceMetrics.calculate_var(returns, 0.95) * 100
            metrics['cvar_95'] = PerformanceMetrics.calculate_cvar(returns, 0.95) * 100
            
        else:
            metrics.update({
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'volatility': 0.0,
                'total_return': 0.0,
                'cagr': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'var_95': 0.0,
                'cvar_95': 0.0
            })
        
        if trades and len(trades) > 0:
            metrics['win_rate'] = PerformanceMetrics.calculate_win_rate(trades)
            metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(trades)
            metrics['total_trades'] = PerformanceMetrics.calculate_trade_count(trades)
            metrics['avg_trade_return'] = PerformanceMetrics.calculate_average_trade_pnl(trades)
            metrics['largest_win'] = PerformanceMetrics.calculate_largest_win(trades)
            metrics['largest_loss'] = PerformanceMetrics.calculate_largest_loss(trades)
            metrics['max_consecutive_wins'] = PerformanceMetrics.calculate_consecutive_wins(trades)
            metrics['max_consecutive_losses'] = PerformanceMetrics.calculate_consecutive_losses(trades)
        else:
            # Defaulting values
            metrics.update({
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            })
        
        return metrics