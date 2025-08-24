import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
import pickle

from .trading_environment import TradingEnvironment
from ..data.duckdb_manager import DuckDBManager

class EpisodeInfoWrapper(gym.Wrapper):
    """Wrapper to ensure episode info is handled correctly for stable-baselines3"""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0
    
    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_length = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        # If episode is done, clean up episode info to prevent SB3 issues
        if done:
            # Remove any existing 'episode' key that might cause issues
            if 'episode' in info:
                del info['episode']
        
        return obs, reward, done, truncated, info

class PerformanceCallback(BaseCallback):
    """Simplified callback to track training performance"""
    
    def __init__(self, db_manager: DuckDBManager, verbose=0):
        super().__init__(verbose)
        self.db_manager = db_manager
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode = 0
        self.step_count = 0
        
    def _on_step(self) -> bool:
        # Simplified step tracking - just count steps
        self.step_count += 1
        return True
    
    def _on_rollout_end(self) -> None:
        # Simplified rollout end - just track that rollout ended
        if self.verbose > 0:
            print(f"Rollout ended at step {self.step_count}")
        pass

class AdaptiveAgent:
    """
    Adaptive Reinforcement Learning Agent for Strategy Selection
    
    The agent continuously learns from market interactions to improve
    strategy selection decisions over time.
    """
    
    def __init__(
        self,
        environment: TradingEnvironment,
        agent_type: str = "PPO",
        model_dir: str = "rl_models",
        learning_rate: float = 3e-4,
        buffer_size: int = 10000,
        batch_size: int = 64,
        training_freq: int = 100,
        target_update_freq: int = 1000
    ):
        self.env = environment
        self.agent_type = agent_type
        self.model_dir = model_dir
        self.db_manager = DuckDBManager()
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.training_freq = training_freq
        self.target_update_freq = target_update_freq
        
        # Setup vectorized environment with custom wrapper
        wrapped_env = EpisodeInfoWrapper(self.env)
        self.vec_env = DummyVecEnv([lambda: wrapped_env])
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Performance tracking
        self.training_history = []
        self.episode_rewards = []
        self.strategy_performance = {}
        
        # Callback for tracking
        self.callback = PerformanceCallback(self.db_manager, verbose=1)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AdaptiveAgent initialized with {agent_type}")
    
    def _create_agent(self):
        """Create the RL agent based on specified type"""
        
        if self.agent_type == "PPO":
            return PPO(
                "MlpPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=256,  # Further reduced to prevent buffer overflow
                batch_size=min(self.batch_size, 16),  # Smaller batch size for stability
                n_epochs=5,  # Reduced epochs for faster training
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device='cpu'  # Use CPU for compatibility
            )
        
        elif self.agent_type == "A2C":
            return A2C(
                "MlpPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                verbose=1,
                device='cpu'
            )
        
        elif self.agent_type == "DQN":
            return DQN(
                "MlpPolicy",
                self.vec_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=1000,
                batch_size=self.batch_size,
                tau=1.0,
                gamma=0.99,
                train_freq=self.training_freq,
                target_update_interval=self.target_update_freq,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=1,
                device='cpu'
            )
        
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def train(
        self, 
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        save_freq: int = 10000,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Train the agent"""
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting training session {session_id} for {total_timesteps} timesteps")
        
        training_start_time = datetime.now()
        
        try:
            # Train the agent
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=self.callback,
                progress_bar=True
            )
            
            training_end_time = datetime.now()
            
            # Save the model
            model_path = os.path.join(self.model_dir, f"{session_id}_{self.agent_type}.zip")
            self.agent.save(model_path)
            
            # Calculate final performance
            final_reward = np.mean(self.callback.episode_rewards[-10:]) if self.callback.episode_rewards else 0.0
            
            # Store training session info
            self.db_manager.store_rl_training_session(
                session_id=session_id,
                agent_type=self.agent_type,
                hyperparameters={
                    'learning_rate': self.learning_rate,
                    'buffer_size': self.buffer_size,
                    'batch_size': self.batch_size,
                    'total_timesteps': total_timesteps
                },
                total_episodes=len(self.callback.episode_rewards),
                final_reward=final_reward,
                model_path=model_path
            )
            
            training_results = {
                'session_id': session_id,
                'agent_type': self.agent_type,
                'total_timesteps': total_timesteps,
                'total_episodes': len(self.callback.episode_rewards),
                'final_reward': final_reward,
                'avg_episode_length': np.mean(self.callback.episode_lengths) if self.callback.episode_lengths else 0,
                'training_time_minutes': (training_end_time - training_start_time).total_seconds() / 60,
                'model_path': model_path,
                'episode_rewards': self.callback.episode_rewards[-50:],  # Last 50 episodes
                'status': 'completed'
            }
            
            self.training_history.append(training_results)
            
            self.logger.info(f"Training completed successfully. Final reward: {final_reward:.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'session_id': session_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """Predict the best action (strategy) for given observation"""
        try:
            action, _states = self.agent.predict(observation, deterministic=deterministic)
            
            # Get action probabilities if available
            action_probs = None
            if hasattr(self.agent, 'policy') and hasattr(self.agent.policy, 'get_distribution'):
                try:
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    with torch.no_grad():
                        distribution = self.agent.policy.get_distribution(obs_tensor)
                        if hasattr(distribution, 'probs'):
                            action_probs = distribution.probs.numpy()[0]
                        elif hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'probs'):
                            action_probs = distribution.distribution.probs.numpy()[0]
                except:
                    action_probs = None
            
            # Map action to strategy name
            strategy_name = self.env.available_strategies[action]
            
            prediction_info = {
                'action': int(action),
                'strategy_name': strategy_name,
                'confidence': float(np.max(action_probs)) if action_probs is not None else 0.0,
                'action_probabilities': action_probs.tolist() if action_probs is not None else None,
                'strategy_probabilities': {
                    strategy: float(prob) for strategy, prob in 
                    zip(self.env.available_strategies, action_probs)
                } if action_probs is not None else None
            }
            
            return action, prediction_info
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return random action as fallback
            action = np.random.choice(len(self.env.available_strategies))
            return action, {
                'action': action,
                'strategy_name': self.env.available_strategies[action],
                'confidence': 0.25,  # Random confidence
                'error': str(e)
            }
    
    def recommend_strategy(self, ticker: str) -> Dict[str, Any]:
        """Get strategy recommendation for a specific ticker"""
        try:
            # Reset environment with specific ticker
            original_tickers = self.env.tickers
            self.env.tickers = [ticker]
            
            obs, _ = self.env.reset()
            action, prediction_info = self.predict(obs, deterministic=True)
            
            # Restore original tickers
            self.env.tickers = original_tickers
            
            recommendation = {
                'ticker': ticker,
                'recommended_strategy': prediction_info['strategy_name'],
                'confidence': prediction_info['confidence'],
                'action_probabilities': prediction_info.get('action_probabilities'),
                'strategy_probabilities': prediction_info.get('strategy_probabilities'),
                'agent_type': self.agent_type,
                'recommendation_type': 'reinforcement_learning',
                'timestamp': datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'recommended_strategy': 'MovingAverageCrossoverStrategy',  # Fallback
                'confidence': 0.0
            }
    
    def evaluate_performance(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate agent performance over multiple episodes"""
        try:
            self.logger.info(f"Evaluating agent performance over {num_episodes} episodes")
            
            episode_results = []
            total_rewards = []
            
            for episode in range(num_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                done = False
                
                while not done and episode_steps < 100:  # Limit steps per episode
                    action, _ = self.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    episode_reward += reward
                    episode_steps += 1
                    
                    if truncated:
                        done = True
                
                # Get episode summary
                episode_summary = self.env.get_episode_summary()
                episode_summary['episode_reward'] = episode_reward
                episode_summary['episode_steps'] = episode_steps
                
                episode_results.append(episode_summary)
                total_rewards.append(episode_reward)
            
            # Calculate evaluation metrics
            evaluation_metrics = {
                'num_episodes': num_episodes,
                'mean_episode_reward': np.mean(total_rewards),
                'std_episode_reward': np.std(total_rewards),
                'mean_total_return': np.mean([ep.get('total_return', 0) for ep in episode_results]),
                'mean_sharpe_ratio': np.mean([ep.get('sharpe_ratio', 0) for ep in episode_results]),
                'mean_max_drawdown': np.mean([ep.get('max_drawdown', 0) for ep in episode_results]),
                'success_rate': len([ep for ep in episode_results if ep.get('total_return', 0) > 0]) / num_episodes,
                'evaluation_timestamp': datetime.now().isoformat(),
                'episode_details': episode_results
            }
            
            self.logger.info(f"Evaluation completed. Mean reward: {evaluation_metrics['mean_episode_reward']:.4f}")
            
            return evaluation_metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            if self.agent_type == "PPO":
                self.agent = PPO.load(model_path, env=self.vec_env)
            elif self.agent_type == "A2C":
                self.agent = A2C.load(model_path, env=self.vec_env)
            elif self.agent_type == "DQN":
                self.agent = DQN.load(model_path, env=self.vec_env)
            else:
                raise ValueError(f"Unsupported agent type: {self.agent_type}")
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def get_strategy_performance_analysis(self) -> Dict[str, Any]:
        """Analyze which strategies the agent prefers in different conditions"""
        try:
            # Get recent RL experiences
            experiences_df = self.db_manager.get_rl_experiences(limit=1000)
            
            if experiences_df.empty:
                return {'error': 'No experience data available'}
            
            # Parse the JSON state and performance data
            analysis = {
                'total_experiences': len(experiences_df),
                'strategy_usage': {},
                'strategy_performance': {},
                'market_condition_preferences': {}
            }
            
            # Analyze strategy usage frequency
            strategy_counts = experiences_df['strategy_used'].value_counts()
            analysis['strategy_usage'] = strategy_counts.to_dict()
            
            # Analyze average rewards per strategy
            strategy_rewards = experiences_df.groupby('strategy_used')['reward'].agg(['mean', 'std', 'count'])
            analysis['strategy_performance'] = strategy_rewards.to_dict('index')
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze strategy performance: {e}")
            return {'error': str(e)}
    
    def continuous_learning_mode(self, update_freq: int = 1000, max_iterations: int = 100):
        """Enable continuous learning mode where agent keeps improving"""
        self.logger.info("Starting continuous learning mode")
        
        iteration = 0
        while iteration < max_iterations:
            try:
                # Train for a short period
                self.agent.learn(total_timesteps=update_freq, reset_num_timesteps=False)
                
                # Evaluate performance
                if iteration % 10 == 0:
                    eval_results = self.evaluate_performance(num_episodes=5)
                    self.logger.info(f"Iteration {iteration}: Mean reward = {eval_results.get('mean_episode_reward', 0):.4f}")
                
                # Save model periodically
                if iteration % 50 == 0:
                    model_path = os.path.join(self.model_dir, f"continuous_{iteration}_{self.agent_type}.zip")
                    self.agent.save(model_path)
                
                iteration += 1
                
            except KeyboardInterrupt:
                self.logger.info("Continuous learning interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous learning: {e}")
                break
        
        self.logger.info(f"Continuous learning completed after {iteration} iterations")