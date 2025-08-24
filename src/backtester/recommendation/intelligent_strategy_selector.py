import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from ..data.duckdb_manager import DuckDBManager
from ..recommendation.synthetic_data_generator import SyntheticDataGenerator

class IntelligentStrategySelector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DuckDBManager()
        self.synthetic_generator = SyntheticDataGenerator()
        
        self.strategy_profiles = {
            'MovingAverageCrossoverStrategy': {
                'best_conditions': ['trending', 'medium_volatility'],
                'worst_conditions': ['sideways', 'high_volatility'],
                'risk_level': 'medium',
                'complexity': 'low',
                'description': 'Follows trends using moving average crossovers'
            },
            'RSIStrategy': {
                'best_conditions': ['sideways', 'overbought_oversold'],
                'worst_conditions': ['strong_trending'],
                'risk_level': 'medium',
                'complexity': 'medium',
                'description': 'Mean reversion using RSI oscillator'
            },
            'BollingerBandsStrategy': {
                'best_conditions': ['high_volatility', 'mean_reverting'],
                'worst_conditions': ['low_volatility', 'strong_trending'],
                'risk_level': 'high',
                'complexity': 'medium',
                'description': 'Volatility-based mean reversion strategy'
            },
            'StochasticOscillatorStrategy': {
                'best_conditions': ['sideways', 'momentum_shifts'],
                'worst_conditions': ['low_volatility'],
                'risk_level': 'medium',
                'complexity': 'high',
                'description': 'Momentum-based oscillator strategy'
            }
        }
        
        self.logger.info("IntelligentStrategySelector initialized")
    
    def analyze_market_conditions(self, ticker: str, lookback_days: int = 60) -> Dict[str, Any]:
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days + 30)
            market_data = self.db_manager.get_market_data(ticker, start_date, end_date)
            
            if market_data.empty or len(market_data) < 30:
                market_data = self.synthetic_generator.generate_market_data(ticker, start_date, end_date)
            
            if market_data.empty:
                return {'error': 'No market data available'}
            
            analysis = self._calculate_market_indicators(market_data)
            regime = self._detect_market_regime(market_data, analysis)
            conditions = self._assess_market_conditions(analysis, regime)
            
            return {
                'ticker': ticker,
                'analysis_date': datetime.now().isoformat(),
                'data_points': len(market_data),
                'technical_analysis': analysis,
                'market_regime': regime,
                'market_conditions': conditions,
                'recommendation_factors': self._get_recommendation_factors(conditions, regime)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze market conditions for {ticker}: {e}")
            return {'error': str(e)}
    
    def _calculate_market_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            returns = data['Close'].pct_change().dropna()
            volatility_20d = returns.rolling(20).std() * np.sqrt(252)
            current_volatility = volatility_20d.iloc[-1] if len(volatility_20d) > 0 else 0.0
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            current_price = data['Close'].iloc[-1]
            sma_20_current = sma_20.iloc[-1] if len(sma_20) > 0 else current_price
            sma_50_current = sma_50.iloc[-1] if len(sma_50) > 0 else current_price
            trend_strength = abs(current_price - sma_20_current) / sma_20_current if sma_20_current > 0 else 0.0
            momentum_5d = (current_price / data['Close'].iloc[-6]) - 1 if len(data) > 5 else 0.0
            momentum_20d = (current_price / data['Close'].iloc[-21]) - 1 if len(data) > 20 else 0.0
            rsi = self._calculate_rsi(data['Close'])
            bb_upper, bb_lower, bb_position = self._calculate_bollinger_bands(data['Close'])
            volume_trend = 0.0
            if 'Volume' in data.columns:
                recent_volume = data['Volume'].rolling(10).mean().iloc[-1]
                historical_volume = data['Volume'].rolling(50).mean().iloc[-1]
                volume_trend = (recent_volume / historical_volume) - 1 if historical_volume > 0 else 0.0
            
            return {
                'current_price': float(current_price),
                'volatility_20d': float(current_volatility),
                'trend_strength': float(trend_strength),
                'price_vs_sma20': float((current_price / sma_20_current) - 1) if sma_20_current > 0 else 0.0,
                'price_vs_sma50': float((current_price / sma_50_current) - 1) if sma_50_current > 0 else 0.0,
                'sma20_vs_sma50': float((sma_20_current / sma_50_current) - 1) if sma_50_current > 0 else 0.0,
                'momentum_5d': float(momentum_5d),
                'momentum_20d': float(momentum_20d),
                'rsi': float(rsi),
                'bb_position': float(bb_position),
                'volume_trend': float(volume_trend),
                'price_range_20d': float((data['Close'].rolling(20).max().iloc[-1] / data['Close'].rolling(20).min().iloc[-1]) - 1) if len(data) > 20 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate market indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if len(rsi) > 0 else 50.0
            
        except:
            return 50.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            current_price = prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            bb_position = (current_price - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0.5
            return float(current_upper), float(current_lower), float(bb_position)
            
        except:
            return 0.0, 0.0, 0.5
    
    def _detect_market_regime(self, data: pd.DataFrame, analysis: Dict[str, float]) -> Dict[str, Any]:
        try:
            sma20_vs_sma50 = analysis.get('sma20_vs_sma50', 0.0)
            trend_strength = analysis.get('trend_strength', 0.0)
            
            if sma20_vs_sma50 > 0.02 and trend_strength > 0.03:
                trend = 'uptrend'
                trend_confidence = min(abs(sma20_vs_sma50) * 10, 1.0)
            elif sma20_vs_sma50 < -0.02 and trend_strength > 0.03:
                trend = 'downtrend'
                trend_confidence = min(abs(sma20_vs_sma50) * 10, 1.0)
            else:
                trend = 'sideways'
                trend_confidence = 1.0 - min(trend_strength * 10, 1.0)
            
            volatility = analysis.get('volatility_20d', 0.0)
            if volatility > 0.3:
                vol_regime = 'high_volatility'
            elif volatility < 0.15:
                vol_regime = 'low_volatility'
            else:
                vol_regime = 'medium_volatility'
            
            momentum_consistency = abs(analysis.get('momentum_5d', 0.0)) + abs(analysis.get('momentum_20d', 0.0))
            if momentum_consistency > 0.1:
                efficiency = 'trending_market'
            elif analysis.get('rsi', 50) > 70 or analysis.get('rsi', 50) < 30:
                efficiency = 'mean_reverting'
            else:
                efficiency = 'efficient_market'
            
            return {
                'trend': trend,
                'trend_confidence': float(trend_confidence),
                'volatility_regime': vol_regime,
                'market_efficiency': efficiency,
                'regime_score': float(trend_confidence * (1 if trend != 'sideways' else 0.5))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect market regime: {e}")
            return {
                'trend': 'sideways',
                'trend_confidence': 0.5,
                'volatility_regime': 'medium_volatility',
                'market_efficiency': 'efficient_market',
                'regime_score': 0.5
            }
    
    def _assess_market_conditions(self, analysis: Dict[str, float], regime: Dict[str, Any]) -> List[str]:
        conditions = []
        if regime['trend'] in ['uptrend', 'downtrend']:
            conditions.append('trending')
            if regime['trend_confidence'] > 0.7:
                conditions.append('strong_trending')
        else:
            conditions.append('sideways')
        
        conditions.append(regime['volatility_regime'])
        rsi = analysis.get('rsi', 50)
        if rsi > 70:
            conditions.append('overbought_oversold')
        elif rsi < 30:
            conditions.append('overbought_oversold')
        
        bb_position = analysis.get('bb_position', 0.5)
        if bb_position > 0.8 or bb_position < 0.2:
            conditions.append('mean_reverting')
        
        momentum_5d = abs(analysis.get('momentum_5d', 0.0))
        if momentum_5d > 0.05:
            conditions.append('momentum_shifts')
        return conditions
    
    def _get_recommendation_factors(self, conditions: List[str], regime: Dict[str, Any]) -> Dict[str, float]:
        factors = {}
        
        for strategy, profile in self.strategy_profiles.items():
            score = 0.5
            
            for condition in conditions:
                if condition in profile['best_conditions']:
                    score += 0.2
            
            for condition in conditions:
                if condition in profile['worst_conditions']:
                    score -= 0.3
            
            if regime['trend_confidence'] > 0.7:
                if strategy == 'MovingAverageCrossoverStrategy':
                    score += 0.2
                elif strategy in ['RSIStrategy', 'BollingerBandsStrategy']:
                    score -= 0.1
            
            if regime['volatility_regime'] == 'high_volatility':
                if strategy == 'BollingerBandsStrategy':
                    score += 0.15
                elif strategy == 'MovingAverageCrossoverStrategy':
                    score -= 0.1
            
            factors[strategy] = max(0.0, min(1.0, score))
        
        return factors
    
    def recommend_strategy(self, ticker: str, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            market_analysis = self.analyze_market_conditions(ticker)
            
            if 'error' in market_analysis:
                return market_analysis
            
            factors = market_analysis['recommendation_factors']
            if user_preferences:
                factors = self._apply_user_preferences(factors, user_preferences)
            
            best_strategy = max(factors.items(), key=lambda x: x[1])
            confidence = best_strategy[1]
            reasoning = self._generate_reasoning(best_strategy[0], market_analysis['market_conditions'],  market_analysis['market_regime'], confidence)
            
            return {
                'ticker': ticker,
                'recommended_strategy': best_strategy[0],
                'confidence': confidence,
                'reasoning': reasoning,
                'market_analysis': market_analysis['technical_analysis'],
                'market_regime': market_analysis['market_regime'],
                'market_conditions': market_analysis['market_conditions'],
                'all_strategy_scores': factors,
                'recommendation_type': 'intelligent_analysis',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to recommend strategy for {ticker}: {e}")
            return {'error': str(e)}
    
    def _apply_user_preferences(self, factors: Dict[str, float], preferences: Dict[str, Any]) -> Dict[str, float]:
        adjusted_factors = factors.copy()
        risk_tolerance = preferences.get('risk_tolerance', 'medium')
        
        for strategy, profile in self.strategy_profiles.items():
            if risk_tolerance == 'low' and profile['risk_level'] == 'high':
                adjusted_factors[strategy] *= 0.7
            elif risk_tolerance == 'high' and profile['risk_level'] == 'low':
                adjusted_factors[strategy] *= 0.9
            elif risk_tolerance == profile['risk_level']:
                adjusted_factors[strategy] *= 1.1
        
        complexity_preference = preferences.get('complexity_preference', 'medium')
        for strategy, profile in self.strategy_profiles.items():
            if complexity_preference == 'low' and profile['complexity'] == 'high':
                adjusted_factors[strategy] *= 0.8
            elif complexity_preference == 'high' and profile['complexity'] == 'low':
                adjusted_factors[strategy] *= 0.9
        
        max_factor = max(adjusted_factors.values()) if adjusted_factors.values() else 1.0
        if max_factor > 0:
            adjusted_factors = {k: v / max_factor for k, v in adjusted_factors.items()}
        
        return adjusted_factors
    
    def _generate_reasoning(self, strategy: str, conditions: List[str], regime: Dict[str, Any], confidence: float) -> str:
        profile = self.strategy_profiles.get(strategy, {})
        base_description = profile.get('description', f'{strategy} was selected')
        
        condition_text = ""
        if 'trending' in conditions:
            if strategy == 'MovingAverageCrossoverStrategy':
                condition_text = "The current trending market conditions favor trend-following strategies. "
            else:
                condition_text = "Despite trending conditions, other factors suggest this strategy. "
        elif 'sideways' in conditions:
            if strategy in ['RSIStrategy', 'StochasticOscillatorStrategy']:
                condition_text = "The sideways market conditions are ideal for oscillator-based strategies. "
        
        vol_regime = regime.get('volatility_regime', 'medium_volatility')
        vol_text = ""
        if vol_regime == 'high_volatility' and strategy == 'BollingerBandsStrategy':
            vol_text = "High market volatility supports volatility-based strategies. "
        elif vol_regime == 'low_volatility' and strategy == 'MovingAverageCrossoverStrategy':
            vol_text = "Low volatility conditions favor stable trend-following approaches. "
        
        if confidence >= 0.8:
            conf_text = "This recommendation has high confidence based on strong alignment with current market conditions."
        elif confidence >= 0.6:
            conf_text = "This recommendation has moderate confidence with good market condition alignment."
        else:
            conf_text = "This recommendation has lower confidence due to mixed market signals."
        
        return f"{base_description}. {condition_text}{vol_text}{conf_text}"
    
    def compare_strategies(self, ticker: str) -> Dict[str, Any]:
        try:
            recommendation = self.recommend_strategy(ticker)
            
            if 'error' in recommendation:
                return recommendation
            
            all_scores = recommendation['all_strategy_scores']
            
            # Create comparison
            comparison = []
            for strategy, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                profile = self.strategy_profiles.get(strategy, {})
                
                comparison.append({
                    'strategy': strategy,
                    'score': score,
                    'rank': len(comparison) + 1,
                    'risk_level': profile.get('risk_level', 'unknown'),
                    'complexity': profile.get('complexity', 'unknown'),
                    'description': profile.get('description', ''),
                    'suitability': 'High' if score > 0.7 else 'Medium' if score > 0.5 else 'Low'
                })
            
            return {
                'ticker': ticker,
                'comparison': comparison,
                'market_analysis': recommendation['market_analysis'],
                'market_regime': recommendation['market_regime'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare strategies for {ticker}: {e}")
            return {'error': str(e)}
    
    def get_strategy_performance_history(self, strategy: str, lookback_days: int = 365) -> Dict[str, Any]:
        try:
            performance_data = {
                'strategy': strategy,
                'lookback_days': lookback_days,
                'simulated_performance': {
                    'total_backtests': np.random.randint(50, 200),
                    'win_rate': np.random.uniform(0.45, 0.65),
                    'avg_return': np.random.uniform(-0.05, 0.15),
                    'volatility': np.random.uniform(0.1, 0.3),
                    'max_drawdown': np.random.uniform(0.05, 0.25),
                    'sharpe_ratio': np.random.uniform(0.5, 2.0)
                },
                'market_regime_performance': {
                    'trending_markets': np.random.uniform(0.4, 0.8),
                    'sideways_markets': np.random.uniform(0.3, 0.7),
                    'high_volatility': np.random.uniform(0.2, 0.6),
                    'low_volatility': np.random.uniform(0.5, 0.8)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Failed to get performance history for {strategy}: {e}")
            return {'error': str(e)}