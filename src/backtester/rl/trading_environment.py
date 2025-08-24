import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, date, timedelta

from ..data.duckdb_manager import DuckDBManager
from ..recommendation.synthetic_data_generator import SyntheticDataGenerator
from ..strategy.strategy import StrategyRegistry
from ..simulation.engine import SimulationEngine
from ..simulation.config import SimulationConfig
from ..metrics.performance import PerformanceMetrics

class TradingEnvironment(gym.Env):
    """
    Reinforcement Learning Environment for Strategy Selection
    
    The agent learns to select the optimal trading strategy at each time step
    based on current market conditions and historical performance.
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        lookback_window: int = 20,
        strategy_hold_period: int = 10,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        max_episodes_per_ticker: int = 50
    ):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.db_manager = DuckDBManager()
        self.synthetic_generator = SyntheticDataGenerator()
        
        # Environment parameters
        self.lookback_window = lookback_window
        self.strategy_hold_period = strategy_hold_period
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_episodes_per_ticker = max_episodes_per_ticker
        
        # Available tickers and strategies
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        self.available_strategies = ['BollingerBandsStrategy', 'RSIStrategy', 'MovingAverageCrossoverStrategy', 'StochasticOscillatorStrategy']
        
        # Action space: strategy selection (0-3 for 4 strategies)
        self.action_space = gym.spaces.Discrete(len(self.available_strategies))
        
        # Observation space: market features + portfolio state + strategy performance history
        # Market features: [volatility, mean_return, total_return, skewness, ma_trend, vol_regime] = 6
        # Portfolio state: [current_return, current_drawdown, sharpe_ratio] = 3
        # Strategy performance: [last_4_strategy_returns] = 4
        # Recent performance: [last_5_period_returns] = 5
        # Total: 18 features
        observation_dim = 18
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_ticker = None
        self.current_data = None
        self.current_step = 0
        self.episode_start_date = None
        self.portfolio_value = initial_capital
        self.portfolio_history = []
        self.strategy_performance_history = []
        self.recent_returns = []
        self.current_strategy_hold_count = 0
        self.last_strategy = None
        
        # Performance tracking
        self.episode_rewards = []
        self.total_episodes = 0
        self._episode_rewards = []  # Track rewards for current episode
        
        self.logger.info("TradingEnvironment initialized")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Select random ticker for this episode
        self.current_ticker = np.random.choice(self.tickers)
        
        # Get or generate market data
        try:
            self.current_data = self._get_market_data(self.current_ticker)
            
            if self.current_data is None or len(self.current_data) < self.lookback_window + 50:
                # Generate synthetic data if insufficient real data
                self.current_data = self._generate_synthetic_data(self.current_ticker)
                
            # Final safety check
            if self.current_data is None or len(self.current_data) < self.lookback_window + 20:
                # Generate minimal synthetic data as fallback
                from datetime import date, timedelta
                end_date = date.today()
                start_date = end_date - timedelta(days=365)  # 1 year minimum
                self.current_data = self.synthetic_generator.generate_market_data(self.current_ticker, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Failed to get market data for {self.current_ticker}: {e}")
            # Generate fallback data
            from datetime import date, timedelta
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            self.current_data = self.synthetic_generator.generate_market_data(self.current_ticker, start_date, end_date)
        
        # Reset episode state
        self.current_step = self.lookback_window
        self.episode_start_date = self.current_data.index[self.current_step]
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.strategy_performance_history = [0.0] * len(self.available_strategies)
        self.recent_returns = [0.0] * 5
        self.current_strategy_hold_count = 0
        self.last_strategy = None
        self._episode_rewards = []  # Reset episode rewards
        
        self.total_episodes += 1
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        self.logger.debug(f"Reset episode {self.total_episodes} with ticker {self.current_ticker}")
        
        return observation.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Debug logging
        if hasattr(self, '_debug_step_counter'):
            self._debug_step_counter = getattr(self, '_debug_step_counter', 0) + 1
            if self._debug_step_counter % 50 == 0:
                print(f"DEBUG: step called {self._debug_step_counter} times, action={action}")
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        # Check if we have valid data
        if self.current_data is None or len(self.current_data) == 0:
            return (np.zeros(self.observation_space.shape[0]).astype(np.float32), 
                    0.0, True, False, self._get_info())
        
        # Select strategy based on action
        selected_strategy = self.available_strategies[action]
        
        # Calculate reward for strategy selection
        reward = self._calculate_reward(action, selected_strategy)
        
        # Track episode rewards
        self._episode_rewards.append(reward)
        
        # Update portfolio based on strategy performance
        self._update_portfolio(action, selected_strategy)
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        done = self._is_episode_done()
        truncated = False
        
        # Get next observation
        if not done:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape[0])
            # Calculate final episode reward
            final_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            reward += final_return * 10  # Bonus for overall performance
        
        info = self._get_info()
        
        return observation.astype(np.float32), reward, done, truncated, info
    
    def _get_market_data(self, ticker: str) -> pd.DataFrame:
        """Get market data for ticker from database or generate synthetic"""
        try:
            # Try to get from database first
            data = self.db_manager.get_market_data(ticker)
            
            if len(data) > 100:
                return data
            
            # Generate synthetic data if not available
            return self._generate_synthetic_data(ticker)
            
        except Exception as e:
            self.logger.warning(f"Failed to get market data for {ticker}: {e}")
            return self._generate_synthetic_data(ticker)
    
    def _generate_synthetic_data(self, ticker: str) -> pd.DataFrame:
        """Generate synthetic market data for ticker"""
        end_date = date.today()
        start_date = end_date - timedelta(days=3650)  # 10 years for comprehensive training
        
        data = self.synthetic_generator.generate_market_data(ticker, start_date, end_date)
        
        # Store in database for future use
        if not data.empty:
            self.db_manager.store_market_data(ticker, data)
        
        return data
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state representation)"""
        try:
            # Debug logging
            if hasattr(self, '_debug_obs_counter'):
                self._debug_obs_counter = getattr(self, '_debug_obs_counter', 0) + 1
                if self._debug_obs_counter % 50 == 0:
                    print(f"DEBUG: _get_observation called {self._debug_obs_counter} times")
                    print(f"DEBUG: current_step={self.current_step}, data_len={len(self.current_data) if self.current_data is not None else 'None'}")
            
            # Ensure we don't exceed data bounds
            if self.current_data is None:
                return np.zeros(self.observation_space.shape[0])
                
            if self.current_step >= len(self.current_data):
                self.current_step = len(self.current_data) - 1
            
            # Get market data window with bounds checking
            end_idx = min(self.current_step, len(self.current_data) - 1)
            start_idx = max(0, end_idx - self.lookback_window)
            
            # Additional safety check
            if start_idx >= len(self.current_data) or end_idx >= len(self.current_data):
                return np.zeros(self.observation_space.shape[0])
            
            market_window = self.current_data.iloc[start_idx:end_idx + 1]
            
            if len(market_window) < 10:
                return np.zeros(self.observation_space.shape[0])
            
            # Calculate market features
            market_features = self._calculate_market_features(market_window)
            
            # Portfolio state features
            portfolio_features = self._calculate_portfolio_features()
            
            # Strategy performance features
            strategy_features = np.array(self.strategy_performance_history[-4:] + [0.0] * (4 - len(self.strategy_performance_history[-4:])))
            
            # Recent performance features
            recent_features = np.array(self.recent_returns[-5:] + [0.0] * (5 - len(self.recent_returns[-5:])))
            
            # Combine all features
            observation = np.concatenate([
                market_features,      # 6 features
                portfolio_features,   # 3 features
                strategy_features,    # 4 features
                recent_features       # 5 features
            ])
            
            # Handle any NaN or infinite values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return observation
            
        except Exception as e:
            self.logger.warning(f"Failed to get observation: {e}")
            return np.zeros(self.observation_space.shape[0])
    
    def _calculate_market_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate market condition features"""
        try:
            if len(data) < 5:
                return np.zeros(6)
            
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 3:
                return np.zeros(6)
            
            # Market features
            volatility = returns.std() * np.sqrt(252)
            mean_return = returns.mean() * 252
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
            skewness = returns.skew() if len(returns) > 3 else 0.0
            
            # Technical features
            ma_trend = 0.0
            if len(data) > 10:
                ma_10 = data['Close'].rolling(10).mean()
                if len(ma_10.dropna()) > 1:
                    ma_trend = (ma_10.iloc[-1] / ma_10.iloc[-2]) - 1
            
            # Volatility regime
            vol_regime = 1.0
            if len(returns) > 10:
                rolling_vol = returns.rolling(10).std()
                vol_regime = rolling_vol.iloc[-1] / rolling_vol.mean() if rolling_vol.mean() > 0 else 1.0
            
            features = np.array([
                volatility, mean_return, total_return, skewness, ma_trend, vol_regime
            ])
            
            return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate market features: {e}")
            return np.zeros(6)
    
    def _calculate_portfolio_features(self) -> np.ndarray:
        """Calculate portfolio state features"""
        try:
            if len(self.portfolio_history) < 2:
                return np.zeros(3)
            
            # Current return
            current_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            
            # Current drawdown
            peak_value = max(self.portfolio_history)
            current_drawdown = (peak_value - self.portfolio_value) / peak_value if peak_value > 0 else 0.0
            
            # Sharpe ratio approximation
            if len(self.portfolio_history) > 5:
                portfolio_returns = pd.Series(self.portfolio_history).pct_change().dropna()
                if len(portfolio_returns) > 1 and portfolio_returns.std() > 0:
                    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            features = np.array([current_return, current_drawdown, sharpe_ratio])
            
            return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate portfolio features: {e}")
            return np.zeros(3)
    
    def _calculate_reward(self, action: int, strategy_name: str) -> float:
        """Calculate reward for strategy selection"""
        try:
            # Base reward starts at 0
            reward = 0.0
            
            # Get strategy performance for current market window
            strategy_return = self._simulate_strategy_performance(strategy_name)
            
            # Update strategy performance history
            while len(self.strategy_performance_history) >= len(self.available_strategies):
                self.strategy_performance_history.pop(0)
            self.strategy_performance_history.append(strategy_return)
            
            # Reward based on strategy performance
            reward += strategy_return * 100  # Scale up returns
            
            # Bonus for consistency (avoiding strategy switching penalty)
            if self.last_strategy == strategy_name:
                reward += 0.1  # Small bonus for consistency
                self.current_strategy_hold_count += 1
            else:
                # Small penalty for switching strategies too frequently
                if self.current_strategy_hold_count < 3:
                    reward -= 0.05
                self.current_strategy_hold_count = 1
            
            # Penalty for excessive risk (high drawdown)
            if len(self.portfolio_history) > 1:
                current_drawdown = self._calculate_portfolio_features()[1]
                if current_drawdown > 0.2:  # 20% drawdown penalty
                    reward -= current_drawdown * 5
            
            # Bonus for improving performance
            if len(self.recent_returns) > 0:
                avg_recent_return = np.mean(self.recent_returns[-3:])
                if strategy_return > avg_recent_return:
                    reward += 0.2  # Bonus for improvement
            
            self.last_strategy = strategy_name
            
            # Clip reward to reasonable range
            reward = np.clip(reward, -10.0, 10.0)
            
            return reward
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate reward: {e}")
            return 0.0
    
    def _simulate_strategy_performance(self, strategy_name: str) -> float:
        """Simulate strategy performance for current market window"""
        try:
            # Get recent market data window
            window_size = min(self.strategy_hold_period, 10)
            end_idx = self.current_step
            start_idx = max(0, end_idx - window_size)
            
            if start_idx >= end_idx:
                return 0.0
            
            market_window = self.current_data.iloc[start_idx:end_idx]
            
            if len(market_window) < 3:
                return 0.0
            
            # Create strategy instance
            strategy = StrategyRegistry.create_strategy(strategy_name)
            
            # Generate signals
            signals = strategy.generate_signals(market_window)
            
            # Run quick simulation
            config = SimulationConfig(
                initial_capital=1000.0,
                leverage=1.0,
                transaction_cost=self.transaction_cost,
                position_sizing="fixed_fraction",
                position_size=0.1
            )
            
            engine = SimulationEngine(config)
            result = engine.run_simulation(market_window, signals)
            
            # Calculate return
            equity_curve = result.equity_curve
            
            # Handle different types of equity curve returns with better error checking
            try:
                if isinstance(equity_curve, (int, float)):
                    # Single numeric value
                    final_value = equity_curve
                elif hasattr(equity_curve, 'iloc') and hasattr(equity_curve, '__len__'):
                    # It's a pandas Series/DataFrame
                    if len(equity_curve) > 0:
                        if isinstance(equity_curve, pd.DataFrame):
                            final_value = equity_curve['equity'].iloc[-1] if 'equity' in equity_curve.columns else equity_curve.iloc[-1, 0]
                        else:
                            final_value = equity_curve.iloc[-1]
                    else:
                        final_value = 1000.0
                elif hasattr(equity_curve, '__len__') and hasattr(equity_curve, '__getitem__'):
                    # It's a list or array
                    if len(equity_curve) > 0:
                        final_value = equity_curve[-1]
                    else:
                        final_value = 1000.0
                else:
                    # Fallback for any other type
                    final_value = float(equity_curve) if equity_curve is not None else 1000.0
            except (IndexError, TypeError, AttributeError):
                final_value = 1000.0
            
            strategy_return = (final_value - 1000.0) / 1000.0
            
            return strategy_return
            
        except Exception as e:
            self.logger.debug(f"Strategy simulation failed for {strategy_name}: {e}")
            # Return a small random return to avoid getting stuck
            return np.random.normal(0.0, 0.01)
    
    def _update_portfolio(self, action: int, strategy_name: str):
        """Update portfolio value based on strategy performance"""
        try:
            # Get strategy return
            strategy_return = self.strategy_performance_history[-1] if self.strategy_performance_history else 0.0
            
            # Apply transaction costs for strategy switching
            transaction_cost = 0.0
            if self.last_strategy and self.last_strategy != strategy_name:
                transaction_cost = self.portfolio_value * self.transaction_cost
            
            # Update portfolio value
            portfolio_change = self.portfolio_value * strategy_return - transaction_cost
            self.portfolio_value += portfolio_change
            
            # Ensure portfolio value doesn't go negative
            self.portfolio_value = max(self.portfolio_value, self.initial_capital * 0.1)
            
            # Update portfolio history
            self.portfolio_history.append(self.portfolio_value)
            
            # Update recent returns
            period_return = portfolio_change / self.portfolio_value if self.portfolio_value > 0 else 0.0
            self.recent_returns.append(period_return)
            if len(self.recent_returns) > 10:
                self.recent_returns.pop(0)
            
        except Exception as e:
            self.logger.warning(f"Failed to update portfolio: {e}")
    
    def _is_episode_done(self) -> bool:
        """Check if episode is complete"""
        # Ensure we have valid data
        if not hasattr(self, 'current_data') or self.current_data is None or len(self.current_data) == 0:
            return True
        
        # Episode ends if we've processed most of the data (leave safety margin)
        max_steps = len(self.current_data) - 15  # Increased safety margin
        
        # Additional bounds checking
        if self.current_step >= max_steps or self.current_step >= len(self.current_data) - 5:
            return True
        
        # Or if portfolio loses too much value
        portfolio_loss = (self.initial_capital - self.portfolio_value) / self.initial_capital
        
        # Or if we've run too many steps for this episode
        max_episode_steps = min(200, len(self.current_data) - self.lookback_window - 10)
        steps_taken = self.current_step - self.lookback_window
        
        return (portfolio_loss > 0.8 or  # 80% loss
                steps_taken >= max_episode_steps)
    
    def _get_info(self) -> Dict:
        """Get additional info about current state"""
        info = {
            'ticker': self.current_ticker,
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'portfolio_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'episode_id': self.total_episodes,  # Renamed to avoid conflict
            'current_strategy': self.last_strategy,
            'strategy_hold_count': self.current_strategy_hold_count
        }
        
        # Always ensure info is a proper dictionary
        return info
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode"""
        if len(self.portfolio_history) < 2:
            return {}
        
        # Calculate episode metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        portfolio_returns = pd.Series(self.portfolio_history).pct_change().dropna()
        
        if len(portfolio_returns) > 1:
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0.0
            max_drawdown = (max(self.portfolio_history) - min(self.portfolio_history)) / max(self.portfolio_history)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        summary = {
            'ticker': self.current_ticker,
            'episode': self.total_episodes,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.portfolio_value,
            'steps_taken': self.current_step - self.lookback_window,
            'strategy_changes': len(set([s for s in self.strategy_performance_history if s != 0])),
            'avg_strategy_return': np.mean(self.strategy_performance_history) if self.strategy_performance_history else 0.0
        }
        
        return summary