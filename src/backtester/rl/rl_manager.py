import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging

# Lazy imports for RL components to avoid Streamlit conflicts
def _safe_import_rl_components():
    """Safely import RL components with error handling"""
    try:
        # Set environment variables to prevent torch conflicts
        os.environ['TORCH_DISABLE_AUTOCAST'] = '1'
        
        # Import with minimal torch functionality
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')
        
        from .trading_environment import TradingEnvironment
        from .adaptive_agent import AdaptiveAgent
        return TradingEnvironment, AdaptiveAgent, None
    except Exception as e:
        return None, None, str(e)

from ..data.duckdb_manager import DuckDBManager
from ..recommendation.synthetic_data_generator import SyntheticDataGenerator

class RLManager:
    """
    Reinforcement Learning Manager
    
    Coordinates the RL-based strategy recommendation system, manages training,
    and provides a unified interface for the adaptive agent.
    """
    
    def __init__(self, cache_dir: str = "rl_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.db_manager = DuckDBManager()
        self.synthetic_generator = SyntheticDataGenerator()
        
        # Initialize environment and agent
        self.environment = None
        self.agent = None
        self.is_trained = False
        
        # Training configuration
        self.default_config = {
            'agent_type': 'PPO',
            'learning_rate': 3e-4,
            'total_timesteps': 50000,
            'lookback_window': 20,
            'strategy_hold_period': 10,
            'initial_capital': 10000.0,
            'transaction_cost': 0.001
        }
        
        self.logger.info("RLManager initialized")
    
    def initialize_system(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize the RL system with environment and agent"""
        try:
            # Safely import RL components
            TradingEnvironment, AdaptiveAgent, import_error = _safe_import_rl_components()
            
            if TradingEnvironment is None or AdaptiveAgent is None:
                return {
                    'status': 'failed',
                    'error': f'Failed to import RL components: {import_error}',
                    'solution': 'This may be due to PyTorch/Streamlit conflicts. Try restarting the application or installing torch separately.'
                }
            
            # Merge with default config
            config = {**self.default_config, **(config or {})}
            
            self.logger.info("Initializing RL system...")
            
            # Ensure we have training data
            try:
                self._prepare_training_data()
            except Exception as e:
                self.logger.warning(f"Failed to prepare training data: {e}")
            
            # Create environment
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'KO']
            
            self.environment = TradingEnvironment(
                tickers=tickers,
                lookback_window=config['lookback_window'],
                strategy_hold_period=config['strategy_hold_period'],
                initial_capital=config['initial_capital'],
                transaction_cost=config['transaction_cost']
            )
            
            # Create agent
            self.agent = AdaptiveAgent(
                environment=self.environment,
                agent_type=config['agent_type'],
                model_dir=os.path.join(self.cache_dir, "models"),
                learning_rate=config['learning_rate']
            )
            
            # Try to load existing model
            latest_model = self._find_latest_model(config['agent_type'])
            if latest_model:
                try:
                    self.agent.load_model(latest_model)
                    self.is_trained = True
                    self.logger.info(f"Loaded existing model: {latest_model}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model {latest_model}: {e}")
            
            result = {
                'status': 'initialized',
                'agent_type': config['agent_type'],
                'environment_tickers': len(tickers),
                'model_loaded': self.is_trained,
                'latest_model': latest_model
            }
            
            self.logger.info("RL system initialized successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RL system: {e}")
            return {
                'status': 'failed', 
                'error': str(e),
                'troubleshooting': 'Try restarting the application. This error may be related to PyTorch/Streamlit module conflicts.'
            }
    
    def _prepare_training_data(self):
        """Prepare and store training data in DuckDB"""
        try:
            # Check if we already have data
            summary = self.db_manager.get_data_summary()
            
            if summary.get('market_data_records', 0) > 10000:  # More data needed for 10 years
                self.logger.info(f"Using existing data: {summary.get('market_data_records')} records")
                return
            
            # Generate and store synthetic data for training (10 years)
            self.logger.info("Generating 10 years of training data...")
            
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'KO', 
                      'PG', 'XOM', 'GE', 'CAT', 'SPY', 'QQQ', 'HD', 'MCD', 'UNH', 'JNJ',
                      'WMT', 'V', 'MA', 'DIS', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD']
            
            for ticker in tickers:
                try:
                    # Generate 10 years of synthetic data (3650 days)
                    end_date = date.today()
                    start_date = end_date - timedelta(days=3650)  # 10 years
                    
                    data = self.synthetic_generator.generate_market_data(ticker, start_date, end_date)
                    
                    if not data.empty:
                        self.db_manager.store_market_data(ticker, data)
                        self.logger.debug(f"Stored 10 years of data for {ticker}: {len(data)} records")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate data for {ticker}: {e}")
                    continue
            
            final_summary = self.db_manager.get_data_summary()
            self.logger.info(f"10-year training data prepared: {final_summary.get('market_data_records')} total records")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
    
    def train_agent(
        self, 
        agent_type: str = 'PPO',
        total_episodes: int = 1000,
        learning_rate: float = 3e-4,
        config: Dict[str, Any] = None,
        force_retrain: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train the RL agent"""
        
        # Build config from parameters
        training_config = {
            'agent_type': agent_type,
            'total_timesteps': total_episodes * 100,  # Convert episodes to timesteps
            'learning_rate': learning_rate,
            **(config or {})
        }
        
        if not self.agent:
            init_result = self.initialize_system(training_config)
            if init_result['status'] != 'initialized':
                return init_result
        
        # Check if already trained
        if self.is_trained and not force_retrain:
            return {
                'status': 'already_trained',
                'message': 'Agent is already trained. Use force_retrain=True to retrain.'
            }
        
        try:
            # Verify RL components are available
            if not hasattr(self, 'agent') or self.agent is None:
                return {
                    'status': 'failed',
                    'error': 'RL agent not properly initialized. This may be due to PyTorch import issues.',
                    'solution': 'Try restarting the application or check that PyTorch is properly installed.'
                }
            
            merged_config = {**self.default_config, **training_config}
            
            if verbose:
                self.logger.info(f"Starting RL agent training with {merged_config['total_timesteps']} timesteps")
            
            # Train the agent
            training_results = self.agent.train(
                total_timesteps=merged_config['total_timesteps'],
                session_id=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if training_results['status'] == 'completed':
                self.is_trained = True
                
                # Evaluate the trained agent
                try:
                    eval_results = self.agent.evaluate_performance(num_episodes=10)
                    training_results['evaluation'] = eval_results
                    training_results['final_reward'] = eval_results.get('mean_reward', 0.0)
                except Exception as eval_error:
                    self.logger.warning(f"Evaluation failed: {eval_error}")
                    training_results['evaluation'] = {'mean_reward': 0.0, 'std_reward': 0.0}
                    training_results['final_reward'] = 0.0
                
                training_results['episodes_completed'] = total_episodes
                
                if verbose:
                    self.logger.info("RL agent training completed successfully")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'troubleshooting': 'This error may be related to PyTorch/Streamlit conflicts. Try restarting the application.'
            }
            return {'status': 'failed', 'error': str(e)}
    
    def get_rl_recommendation(self, ticker: str, live_learning: bool = False) -> Dict[str, Any]:
        """Get RL-based strategy recommendation for a ticker"""
        
        if not self.is_trained:
            return {
                'error': 'RL agent not trained. Please train the agent first.',
                'ticker': ticker,
                'recommendation_type': 'reinforcement_learning'
            }
        
        try:
            recommendation = self.agent.recommend_strategy(ticker)
            
            # Add additional context
            recommendation['recommendation_source'] = 'adaptive_rl_agent'
            recommendation['agent_type'] = self.agent.agent_type
            recommendation['is_rl_trained'] = self.is_trained
            recommendation['live_learning'] = live_learning
            recommendation['timestamp'] = datetime.now().isoformat()
            
            # If live learning is enabled, store this as an experience for future training
            if live_learning:
                try:
                    # Store this recommendation as an experience for continuous learning
                    episode = int(datetime.now().timestamp())  # Use timestamp as episode
                    step = 0
                    
                    # Create a simplified state representation
                    state = {
                        'ticker': ticker,
                        'timestamp': recommendation['timestamp'],
                        'market_conditions': 'live_trading'
                    }
                    
                    # Store the experience
                    self.db_manager.store_rl_experience(
                        episode=episode,
                        step=step,
                        ticker=ticker,
                        state=state,
                        action=0,  # Placeholder
                        reward=0.0,  # Will be updated based on actual performance
                        done=True,
                        strategy_used=recommendation.get('recommended_strategy')
                    )
                    
                    recommendation['experience_stored'] = True
                    
                except Exception as e:
                    self.logger.warning(f"Failed to store live learning experience: {e}")
                    recommendation['experience_stored'] = False
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to get RL recommendation for {ticker}: {e}")
            return {
                'error': str(e),
                'ticker': ticker,
                'recommendation_type': 'reinforcement_learning'
            }
    
    def compare_with_traditional_methods(self, ticker: str) -> Dict[str, Any]:
        """Compare RL recommendation with traditional ML methods"""
        try:
            # Get RL recommendation
            rl_recommendation = self.get_rl_recommendation(ticker)
            
            # Simulate traditional ML recommendation (simplified)
            traditional_strategies = ['BollingerBandsStrategy', 'RSIStrategy', 'MovingAverageCrossoverStrategy', 'StochasticOscillatorStrategy']
            traditional_recommendation = {
                'recommended_strategy': np.random.choice(traditional_strategies),
                'confidence': np.random.uniform(0.6, 0.9),
                'method': 'traditional_ml'
            }
            
            comparison = {
                'ticker': ticker,
                'rl_recommendation': rl_recommendation,
                'traditional_recommendation': traditional_recommendation,
                'comparison_timestamp': datetime.now().isoformat(),
                'recommendations_match': (
                    rl_recommendation.get('recommended_strategy') == 
                    traditional_recommendation.get('recommended_strategy')
                )
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare recommendations for {ticker}: {e}")
            return {'error': str(e)}
    
    def get_agent_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about agent performance"""
        try:
            analytics = {
                'system_status': {
                    'is_initialized': self.agent is not None,
                    'is_trained': self.is_trained,
                    'agent_type': self.agent.agent_type if self.agent else None
                },
                'training_history': [],
                'strategy_analysis': {},
                'database_summary': {}
            }
            
            # Get training history
            if self.agent:
                analytics['training_history'] = self.agent.get_training_history()
            
            # Get strategy performance analysis
            if self.is_trained:
                analytics['strategy_analysis'] = self.agent.get_strategy_performance_analysis()
            
            # Get database summary
            analytics['database_summary'] = self.db_manager.get_data_summary()
            
            # Get recent RL experiences summary
            recent_experiences = self.db_manager.get_rl_experiences(limit=100)
            if not recent_experiences.empty:
                analytics['recent_performance'] = {
                    'total_experiences': len(recent_experiences),
                    'unique_episodes': recent_experiences['episode'].nunique(),
                    'avg_reward': recent_experiences['reward'].mean(),
                    'recent_episodes': recent_experiences['episode'].max() if len(recent_experiences) > 0 else 0
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics: {e}")
            return {'error': str(e)}
    
    def _find_latest_model(self, agent_type: str) -> Optional[str]:
        """Find the latest trained model for the agent type"""
        try:
            model_dir = os.path.join(self.cache_dir, "models")
            if not os.path.exists(model_dir):
                return None
            
            model_files = [
                f for f in os.listdir(model_dir) 
                if f.endswith(f"{agent_type}.zip")
            ]
            
            if not model_files:
                return None
            
            # Sort by modification time and return latest
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            latest_model = os.path.join(model_dir, model_files[0])
            
            return latest_model
            
        except Exception as e:
            self.logger.warning(f"Failed to find latest model: {e}")
            return None
    
    def start_continuous_learning(self, update_freq: int = 1000, max_iterations: int = 100):
        """Start continuous learning mode"""
        if not self.is_trained:
            self.logger.error("Agent must be trained before starting continuous learning")
            return {'error': 'Agent not trained'}
        
        try:
            self.logger.info("Starting continuous learning mode")
            self.agent.continuous_learning_mode(update_freq, max_iterations)
            return {'status': 'continuous_learning_completed'}
            
        except Exception as e:
            self.logger.error(f"Continuous learning failed: {e}")
            return {'error': str(e)}
    
    def simulate_live_trading(self, num_days: int = 30) -> Dict[str, Any]:
        """Simulate live trading with the RL agent"""
        if not self.is_trained:
            return {'error': 'Agent not trained'}
        
        try:
            self.logger.info(f"Simulating {num_days} days of live trading")
            
            simulation_results = []
            
            # Select a few tickers for simulation
            test_tickers = ['AAPL', 'MSFT', 'GOOGL']
            
            for ticker in test_tickers:
                # Get recent market data
                end_date = date.today()
                start_date = end_date - timedelta(days=num_days + 50)  # Extra data for context
                
                market_data = self.db_manager.get_market_data(ticker, start_date, end_date)
                
                if len(market_data) < num_days:
                    # Generate synthetic data for simulation
                    market_data = self.synthetic_generator.generate_market_data(ticker, start_date, end_date)
                
                if len(market_data) < num_days:
                    continue
                
                # Simulate trading decisions
                daily_decisions = []
                portfolio_value = 10000.0
                
                for i in range(min(num_days, len(market_data) - 20)):
                    # Prepare observation
                    window_data = market_data.iloc[i:i+20]
                    
                    # This would require more complex state preparation
                    # For now, use a simplified observation
                    obs = np.random.random(18)  # Placeholder
                    
                    # Get agent's recommendation
                    action, prediction_info = self.agent.predict(obs, deterministic=True)
                    
                    # Simulate portfolio update
                    daily_return = np.random.normal(0.001, 0.02)  # Simplified
                    portfolio_value *= (1 + daily_return)
                    
                    daily_decisions.append({
                        'day': i,
                        'ticker': ticker,
                        'recommended_strategy': prediction_info['strategy_name'],
                        'confidence': prediction_info['confidence'],
                        'portfolio_value': portfolio_value,
                        'daily_return': daily_return
                    })
                
                simulation_results.append({
                    'ticker': ticker,
                    'final_portfolio_value': portfolio_value,
                    'total_return': (portfolio_value - 10000.0) / 10000.0,
                    'decisions': daily_decisions
                })
            
            summary = {
                'simulation_days': num_days,
                'tickers_tested': len(simulation_results),
                'results': simulation_results,
                'avg_return': np.mean([r['total_return'] for r in simulation_results]),
                'simulation_timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Live trading simulation failed: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_old: int = 90):
        """Clean up old training data"""
        try:
            self.db_manager.cleanup_old_data(days_old)
            self.logger.info(f"Cleaned up data older than {days_old} days")
            return {'status': 'cleanup_completed'}
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'rl_system_initialized': self.agent is not None,
            'rl_agent_trained': self.is_trained,
            'agent_type': self.agent.agent_type if self.agent else None,
            'environment_ready': self.environment is not None,
            'database_summary': self.db_manager.get_data_summary(),
            'cache_directory': self.cache_dir,
            'available_models': self._list_available_models()
        }
    
    def _list_available_models(self) -> List[str]:
        """List available trained models"""
        try:
            model_dir = os.path.join(self.cache_dir, "models")
            if not os.path.exists(model_dir):
                return []
            
            return [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            
        except Exception:
            return []