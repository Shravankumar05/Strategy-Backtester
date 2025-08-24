import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

from .trading_environment import TradingEnvironment
from .adaptive_agent import AdaptiveAgent
from ..data.duckdb_manager import DuckDBManager
from ..recommendation.synthetic_data_generator import SyntheticDataGenerator

class ImprovedRLManager:
    """
    Improved Reinforcement Learning Manager with enhanced logic and performance
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
        
        # Enhanced training configuration
        self.default_config = {
            'agent_type': 'PPO',
            'learning_rate': 3e-4,
            'total_timesteps': 100000,  # Increased for better learning
            'lookback_window': 30,      # Increased for more context
            'strategy_hold_period': 15, # Increased for more stable decisions
            'initial_capital': 10000.0,
            'transaction_cost': 0.001,
            'reward_scaling': 100.0,    # Scale rewards for better learning
            'exploration_bonus': 0.1    # Encourage exploration
        }
        
        # Performance tracking
        self.training_sessions = []
        self.recommendation_history = []
        
        self.logger.info("ImprovedRLManager initialized")
    
    def initialize_system(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize the improved RL system"""
        try:
            config = {**self.default_config, **(config or {})}
            
            self.logger.info("Initializing improved RL system...")
            
            # Prepare comprehensive training data
            self._prepare_comprehensive_training_data()
            
            # Create enhanced environment
            tickers = self._get_diverse_ticker_set()
            
            self.environment = TradingEnvironment(
                tickers=tickers,
                lookback_window=config['lookback_window'],
                strategy_hold_period=config['strategy_hold_period'],
                initial_capital=config['initial_capital'],
                transaction_cost=config['transaction_cost']
            )
            
            # Create enhanced agent
            self.agent = AdaptiveAgent(
                environment=self.environment,
                agent_type=config['agent_type'],
                model_dir=os.path.join(self.cache_dir, "models"),
                learning_rate=config['learning_rate']
            )
            
            # Try to load existing model
            latest_model = self._find_latest_model(config['agent_type'])
            if latest_model:
                if self.agent.load_model(latest_model):
                    self.is_trained = True
                    self.logger.info(f"Loaded existing model: {latest_model}")
            
            result = {
                'status': 'initialized',
                'agent_type': config['agent_type'],
                'environment_tickers': len(tickers),
                'model_loaded': self.is_trained,
                'latest_model': latest_model,
                'training_data_records': self._get_training_data_count()
            }
            
            self.logger.info("Improved RL system initialized successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to initialize improved RL system: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _get_diverse_ticker_set(self) -> List[str]:
        """Get a diverse set of tickers for training"""
        # Diversified ticker set across sectors and market caps
        tickers = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',
            
            # Consumer
            'KO', 'PG', 'WMT', 'HD', 'MCD',
            
            # Industrial
            'CAT', 'GE', 'BA', 'MMM',
            
            # Energy
            'XOM', 'CVX', 'COP',
            
            # ETFs for market exposure
            'SPY', 'QQQ', 'IWM', 'VTI'
        ]
        
        return tickers
    
    def _prepare_comprehensive_training_data(self):
        """Prepare comprehensive training data with market regime awareness"""
        try:
            summary = self.db_manager.get_data_summary()
            
            # Check if we have sufficient data (aim for 5+ years across all tickers)
            required_records = 50000  # Approximately 5 years * 252 days * 40 tickers
            
            if summary.get('market_data_records', 0) >= required_records:
                self.logger.info(f"Using existing comprehensive data: {summary.get('market_data_records')} records")
                return
            
            self.logger.info("Generating comprehensive training data with market regime awareness...")
            
            tickers = self._get_diverse_ticker_set()
            
            # Generate data for different market regimes
            regimes = [
                {'name': 'bull_market', 'trend': 0.08, 'volatility': 0.15, 'years': 2},
                {'name': 'bear_market', 'trend': -0.12, 'volatility': 0.25, 'years': 1},
                {'name': 'sideways_market', 'trend': 0.02, 'volatility': 0.12, 'years': 1.5},
                {'name': 'high_volatility', 'trend': 0.05, 'volatility': 0.35, 'years': 0.5}
            ]
            
            for ticker in tickers:
                try:
                    all_data = pd.DataFrame()
                    current_date = date.today() - timedelta(days=365 * 5)  # Start 5 years ago
                    
                    for regime in regimes:
                        days = int(regime['years'] * 365)
                        end_date = current_date + timedelta(days=days)
                        
                        # Generate regime-specific data
                        regime_data = self.synthetic_generator.generate_market_data(
                            ticker=ticker,
                            start_date=current_date,
                            end_date=end_date,
                            trend=regime['trend'],
                            volatility=regime['volatility']
                        )
                        
                        if not regime_data.empty:
                            # Add regime metadata
                            regime_data['market_regime'] = regime['name']
                            all_data = pd.concat([all_data, regime_data], ignore_index=True)
                        
                        current_date = end_date
                    
                    if not all_data.empty:
                        # Remove regime column before storing (not needed in OHLCV data)
                        ohlcv_data = all_data.drop('market_regime', axis=1, errors='ignore')
                        self.db_manager.store_market_data(ticker, ohlcv_data)
                        self.logger.debug(f"Stored comprehensive data for {ticker}: {len(ohlcv_data)} records")
                
                except Exception as e:
                    self.logger.warning(f"Failed to generate comprehensive data for {ticker}: {e}")
                    continue
            
            final_summary = self.db_manager.get_data_summary()
            self.logger.info(f"Comprehensive training data prepared: {final_summary.get('market_data_records')} total records")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare comprehensive training data: {e}")
    
    def _get_training_data_count(self) -> int:
        """Get the count of training data records"""
        try:
            summary = self.db_manager.get_data_summary()
            return summary.get('market_data_records', 0)
        except:
            return 0
    
    def train_agent_with_curriculum(
        self,
        agent_type: str = 'PPO',
        curriculum_stages: List[Dict] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train agent using curriculum learning for better performance"""
        
        if not curriculum_stages:
            # Default curriculum: start simple, increase complexity
            curriculum_stages = [
                {
                    'name': 'basic_patterns',
                    'timesteps': 25000,
                    'tickers': ['SPY', 'QQQ'],  # Start with ETFs (smoother)
                    'lookback_window': 20,
                    'description': 'Learn basic market patterns'
                },
                {
                    'name': 'sector_diversity',
                    'timesteps': 50000,
                    'tickers': ['AAPL', 'JPM', 'JNJ', 'XOM'],  # Add sector diversity
                    'lookback_window': 25,
                    'description': 'Learn sector-specific patterns'
                },
                {
                    'name': 'full_complexity',
                    'timesteps': 75000,
                    'tickers': None,  # Use all tickers
                    'lookback_window': 30,
                    'description': 'Master complex market dynamics'
                }
            ]
        
        if not self.agent:
            init_result = self.initialize_system({'agent_type': agent_type})
            if init_result['status'] != 'initialized':
                return init_result
        
        try:
            session_id = f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if verbose:
                self.logger.info(f"Starting curriculum learning with {len(curriculum_stages)} stages")
            
            total_training_time = 0
            stage_results = []
            
            for i, stage in enumerate(curriculum_stages, 1):
                stage_start_time = datetime.now()
                
                if verbose:
                    self.logger.info(f"Stage {i}/{len(curriculum_stages)}: {stage['name']} - {stage['description']}")
                
                # Update environment for this stage
                if stage['tickers']:
                    self.environment.tickers = stage['tickers']
                else:
                    self.environment.tickers = self._get_diverse_ticker_set()
                
                self.environment.lookback_window = stage['lookback_window']
                
                # Train for this stage
                stage_result = self.agent.train(
                    total_timesteps=stage['timesteps'],
                    session_id=f"{session_id}_stage_{i}"
                )
                
                stage_end_time = datetime.now()
                stage_duration = (stage_end_time - stage_start_time).total_seconds() / 60
                total_training_time += stage_duration
                
                stage_result.update({
                    'stage_name': stage['name'],
                    'stage_number': i,
                    'stage_duration_minutes': stage_duration
                })
                
                stage_results.append(stage_result)
                
                if verbose:
                    final_reward = stage_result.get('final_reward', 0)
                    self.logger.info(f"Stage {i} completed. Final reward: {final_reward:.4f}, Duration: {stage_duration:.1f} min")
            
            # Final evaluation on full dataset
            if verbose:
                self.logger.info("Conducting final evaluation...")
            
            self.environment.tickers = self._get_diverse_ticker_set()
            evaluation_results = self.agent.evaluate_performance(num_episodes=20)
            
            # Compile final results
            final_results = {
                'session_id': session_id,
                'status': 'completed',
                'training_method': 'curriculum_learning',
                'agent_type': agent_type,
                'total_stages': len(curriculum_stages),
                'total_training_time_minutes': total_training_time,
                'stage_results': stage_results,
                'final_evaluation': evaluation_results,
                'curriculum_stages': curriculum_stages
            }
            
            # Mark as trained
            self.is_trained = True
            
            # Store training session
            self.training_sessions.append(final_results)
            
            if verbose:
                mean_reward = evaluation_results.get('mean_episode_reward', 0)
                self.logger.info(f"Curriculum training completed! Final evaluation reward: {mean_reward:.4f}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Curriculum training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_enhanced_rl_recommendation(
        self,
        ticker: str,
        market_context: Dict[str, Any] = None,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Get enhanced RL recommendation with market context awareness"""
        
        if not self.is_trained:
            return {
                'error': 'RL agent not trained. Please train the agent first.',
                'ticker': ticker,
                'recommendation_type': 'reinforcement_learning'
            }
        
        try:
            # Get basic recommendation
            recommendation = self.agent.recommend_strategy(ticker)
            
            # Enhance with market context
            if market_context:
                recommendation = self._enhance_with_market_context(recommendation, market_context)
            
            # Add confidence analysis
            confidence = recommendation.get('confidence', 0.0)
            
            if confidence < confidence_threshold:
                recommendation['warning'] = f"Low confidence ({confidence:.1%}). Consider additional analysis."
                recommendation['reliability'] = 'low'
            elif confidence < 0.8:
                recommendation['reliability'] = 'medium'
            else:
                recommendation['reliability'] = 'high'
            
            # Add strategy reasoning
            recommendation['reasoning'] = self._generate_strategy_reasoning(
                recommendation['recommended_strategy'],
                confidence,
                market_context
            )
            
            # Store recommendation for learning
            self.recommendation_history.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'recommendation': recommendation,
                'market_context': market_context
            })
            
            # Enhanced metadata
            recommendation.update({
                'recommendation_source': 'enhanced_rl_agent',
                'agent_type': self.agent.agent_type,
                'training_sessions': len(self.training_sessions),
                'recommendation_id': len(self.recommendation_history)
            })
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced RL recommendation for {ticker}: {e}")
            return {
                'error': str(e),
                'ticker': ticker,
                'recommendation_type': 'reinforcement_learning'
            }
    
    def _enhance_with_market_context(
        self,
        recommendation: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance recommendation with market context"""
        
        # Adjust confidence based on market conditions
        base_confidence = recommendation.get('confidence', 0.0)
        
        # Market volatility adjustment
        volatility = market_context.get('volatility', 'medium')
        if volatility == 'high':
            # Lower confidence in high volatility
            recommendation['confidence'] = base_confidence * 0.9
            recommendation['volatility_adjustment'] = -0.1
        elif volatility == 'low':
            # Higher confidence in low volatility
            recommendation['confidence'] = min(base_confidence * 1.1, 1.0)
            recommendation['volatility_adjustment'] = 0.1
        
        # Trend strength adjustment
        trend_strength = market_context.get('trend_strength', 'medium')
        if trend_strength == 'strong':
            # Prefer trend-following strategies
            if 'MovingAverage' in recommendation['recommended_strategy']:
                recommendation['confidence'] = min(recommendation['confidence'] * 1.15, 1.0)
                recommendation['trend_bonus'] = 0.15
        
        # Market regime adjustment
        market_regime = market_context.get('market_regime', 'normal')
        if market_regime == 'bear':
            # Prefer defensive strategies
            if 'RSI' in recommendation['recommended_strategy']:
                recommendation['confidence'] = min(recommendation['confidence'] * 1.1, 1.0)
                recommendation['regime_adjustment'] = 0.1
        
        recommendation['market_context_applied'] = True
        
        return recommendation
    
    def _generate_strategy_reasoning(
        self,
        strategy: str,
        confidence: float,
        market_context: Dict[str, Any] = None
    ) -> str:
        """Generate human-readable reasoning for strategy selection"""
        
        strategy_explanations = {
            'MovingAverageCrossoverStrategy': 'Moving Average Crossover is effective in trending markets with clear directional moves.',
            'RSIStrategy': 'RSI Strategy works well in ranging markets and helps identify overbought/oversold conditions.',
            'BollingerBandsStrategy': 'Bollinger Bands Strategy is suitable for volatile markets and mean reversion scenarios.',
            'StochasticOscillatorStrategy': 'Stochastic Oscillator Strategy excels in sideways markets and momentum identification.'
        }
        
        base_reasoning = strategy_explanations.get(strategy, f'{strategy} was selected based on current market analysis.')
        
        # Add confidence reasoning
        if confidence >= 0.8:
            confidence_text = "The agent has high confidence in this recommendation based on extensive training data."
        elif confidence >= 0.6:
            confidence_text = "The agent has moderate confidence, suggesting this strategy aligns well with current conditions."
        else:
            confidence_text = "The agent has lower confidence, indicating mixed signals or unusual market conditions."
        
        # Add market context reasoning
        context_text = ""
        if market_context:
            volatility = market_context.get('volatility', 'medium')
            trend = market_context.get('trend_strength', 'medium')
            
            if volatility == 'high':
                context_text += " High market volatility supports strategies that can adapt quickly to price changes."
            elif volatility == 'low':
                context_text += " Low market volatility favors strategies that can capture small, consistent moves."
            
            if trend == 'strong':
                context_text += " Strong market trends favor momentum-based approaches."
            elif trend == 'weak':
                context_text += " Weak trends suggest mean-reversion strategies may be more effective."
        
        return f"{base_reasoning} {confidence_text}{context_text}"
    
    def analyze_strategy_performance_by_regime(self) -> Dict[str, Any]:
        """Analyze how different strategies perform in different market regimes"""
        try:
            # Get RL experiences
            experiences_df = self.db_manager.get_rl_experiences(limit=5000)
            
            if experiences_df.empty:
                return {'error': 'No experience data available for analysis'}
            
            analysis = {
                'total_experiences': len(experiences_df),
                'regime_analysis': {},
                'strategy_effectiveness': {},
                'recommendations': []
            }
            
            # Analyze by strategy
            strategy_groups = experiences_df.groupby('strategy_used')
            
            for strategy, group in strategy_groups:
                if strategy and len(group) > 10:  # Minimum sample size
                    avg_reward = group['reward'].mean()
                    reward_std = group['reward'].std()
                    success_rate = (group['reward'] > 0).mean()
                    
                    analysis['strategy_effectiveness'][strategy] = {
                        'average_reward': float(avg_reward),
                        'reward_volatility': float(reward_std),
                        'success_rate': float(success_rate),
                        'sample_size': len(group),
                        'confidence_score': min(success_rate * (1 - reward_std), 1.0)
                    }
            
            # Generate recommendations based on analysis
            if analysis['strategy_effectiveness']:
                best_strategy = max(
                    analysis['strategy_effectiveness'].items(),
                    key=lambda x: x[1]['confidence_score']
                )
                
                analysis['recommendations'].append({
                    'type': 'best_overall_strategy',
                    'strategy': best_strategy[0],
                    'confidence_score': best_strategy[1]['confidence_score'],
                    'reasoning': f"Highest confidence score based on reward consistency and success rate"
                })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze strategy performance by regime: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the RL system"""
        try:
            analytics = {
                'system_overview': {
                    'is_initialized': self.agent is not None,
                    'is_trained': self.is_trained,
                    'agent_type': self.agent.agent_type if self.agent else None,
                    'training_sessions_completed': len(self.training_sessions),
                    'recommendations_generated': len(self.recommendation_history)
                },
                'training_analytics': {},
                'performance_analytics': {},
                'data_analytics': {},
                'recommendation_analytics': {}
            }
            
            # Training analytics
            if self.training_sessions:
                latest_session = self.training_sessions[-1]
                
                analytics['training_analytics'] = {
                    'latest_session_id': latest_session.get('session_id'),
                    'training_method': latest_session.get('training_method', 'standard'),
                    'total_training_time': sum(
                        session.get('total_training_time_minutes', 0) 
                        for session in self.training_sessions
                    ),
                    'average_final_reward': np.mean([
                        session.get('final_evaluation', {}).get('mean_episode_reward', 0)
                        for session in self.training_sessions
                    ]),
                    'training_progression': [
                        {
                            'session': session.get('session_id'),
                            'reward': session.get('final_evaluation', {}).get('mean_episode_reward', 0),
                            'timestamp': session.get('session_id', '').split('_')[-2:] if '_' in session.get('session_id', '') else []
                        }
                        for session in self.training_sessions[-5:]  # Last 5 sessions
                    ]
                }
            
            # Performance analytics
            if self.is_trained:
                strategy_analysis = self.analyze_strategy_performance_by_regime()
                analytics['performance_analytics'] = strategy_analysis
            
            # Data analytics
            analytics['data_analytics'] = {
                'training_data_records': self._get_training_data_count(),
                'tickers_available': len(self._get_diverse_ticker_set()),
                'database_summary': self.db_manager.get_data_summary()
            }
            
            # Recommendation analytics
            if self.recommendation_history:
                recent_recommendations = self.recommendation_history[-10:]
                
                analytics['recommendation_analytics'] = {
                    'total_recommendations': len(self.recommendation_history),
                    'recent_recommendations': len(recent_recommendations),
                    'average_confidence': np.mean([
                        rec['recommendation'].get('confidence', 0)
                        for rec in recent_recommendations
                    ]),
                    'strategy_distribution': self._analyze_recommendation_distribution(),
                    'reliability_distribution': self._analyze_reliability_distribution()
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive analytics: {e}")
            return {'error': str(e)}
    
    def _analyze_recommendation_distribution(self) -> Dict[str, int]:
        """Analyze distribution of recommended strategies"""
        strategy_counts = {}
        
        for rec_data in self.recommendation_history:
            strategy = rec_data['recommendation'].get('recommended_strategy', 'Unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return strategy_counts
    
    def _analyze_reliability_distribution(self) -> Dict[str, int]:
        """Analyze distribution of recommendation reliability levels"""
        reliability_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for rec_data in self.recommendation_history:
            reliability = rec_data['recommendation'].get('reliability', 'medium')
            reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1
        
        return reliability_counts
    
    def continuous_improvement_cycle(
        self,
        improvement_cycles: int = 5,
        timesteps_per_cycle: int = 10000
    ) -> Dict[str, Any]:
        """Run continuous improvement cycles for ongoing learning"""
        
        if not self.is_trained:
            return {'error': 'Agent must be trained before continuous improvement'}
        
        try:
            self.logger.info(f"Starting {improvement_cycles} continuous improvement cycles")
            
            improvement_results = []
            
            for cycle in range(1, improvement_cycles + 1):
                cycle_start_time = datetime.now()
                
                # Evaluate current performance
                pre_improvement = self.agent.evaluate_performance(num_episodes=5)
                
                # Run improvement training
                self.agent.continuous_learning_mode(
                    update_freq=timesteps_per_cycle,
                    max_iterations=1
                )
                
                # Evaluate post-improvement performance
                post_improvement = self.agent.evaluate_performance(num_episodes=5)
                
                cycle_end_time = datetime.now()
                cycle_duration = (cycle_end_time - cycle_start_time).total_seconds() / 60
                
                # Calculate improvement
                pre_reward = pre_improvement.get('mean_episode_reward', 0)
                post_reward = post_improvement.get('mean_episode_reward', 0)
                improvement = post_reward - pre_reward
                
                cycle_result = {
                    'cycle': cycle,
                    'pre_improvement_reward': pre_reward,
                    'post_improvement_reward': post_reward,
                    'improvement': improvement,
                    'improvement_percentage': (improvement / abs(pre_reward)) * 100 if pre_reward != 0 else 0,
                    'cycle_duration_minutes': cycle_duration
                }
                
                improvement_results.append(cycle_result)
                
                self.logger.info(f"Cycle {cycle} completed. Improvement: {improvement:.4f} ({cycle_result['improvement_percentage']:.1f}%)")
            
            # Summary
            total_improvement = sum(result['improvement'] for result in improvement_results)
            avg_improvement = total_improvement / improvement_cycles
            
            summary = {
                'status': 'completed',
                'total_cycles': improvement_cycles,
                'total_improvement': total_improvement,
                'average_improvement_per_cycle': avg_improvement,
                'cycle_results': improvement_results,
                'final_performance': improvement_results[-1]['post_improvement_reward'] if improvement_results else 0
            }
            
            self.logger.info(f"Continuous improvement completed. Total improvement: {total_improvement:.4f}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Continuous improvement failed: {e}")
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
    
    def export_training_report(self, output_path: str = None) -> Dict[str, Any]:
        """Export comprehensive training report"""
        try:
            if not output_path:
                output_path = os.path.join(self.cache_dir, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'rl_manager_version': 'improved_v1.0',
                    'system_status': 'trained' if self.is_trained else 'not_trained'
                },
                'system_analytics': self.get_comprehensive_system_analytics(),
                'training_sessions': self.training_sessions,
                'recommendation_history': self.recommendation_history[-50:],  # Last 50 recommendations
                'performance_analysis': self.analyze_strategy_performance_by_regime()
            }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Training report exported to: {output_path}")
            
            return {
                'status': 'success',
                'report_path': output_path,
                'report_size_mb': os.path.getsize(output_path) / (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export training report: {e}")
            return {'error': str(e)}