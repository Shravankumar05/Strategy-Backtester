import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import os
import pickle
from .synthetic_data_generator import SyntheticDataGenerator
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from ..data.yfinance_fetcher import YFinanceDataFetcher
from ..data.cache_manager import CacheManager

class EnhancedRecommendationEngine:

    def __init__(self, cache_dir: str = "recommendation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.synthetic_generator = SyntheticDataGenerator()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(cache_dir)
        self.data_fetcher = YFinanceDataFetcher(CacheManager())

        self._is_trained = False
        self._training_info = {}

        self.ticker_categories = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'financial': ['JPM', 'BAC', 'GS', 'WFC', 'C', 'AXP', 'BLK'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'DHR'],
            'consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'industrial': ['BA', 'CAT', 'GE', 'LMT', 'HON'],
            'etf': ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'XLF', 'XLK']
        }

        self.all_tickers = []
        for tickers in self.ticker_categories.values():
            self.all_tickers.extend(tickers)

    def build_recommendation_system(
        self,
        retrain: bool = False,
        use_synthetic_data: bool = True
    ) -> Dict[str, Any]:

        self.logger.info("Starting enhanced recommendation system build...")

        if not retrain and self.model_trainer.get_latest_models():
            self.logger.info("Loaded existing models successfully")
            self._is_trained = True
            return {"status": "loaded_existing", "message": "Using existing trained models"}

        build_results = {
            "status": "training",
            "steps": [],
            "performance": {},
            "errors": []
        }

        try:

            self.logger.info("Step 1: Generating training data...")
            build_results["steps"].append("data_generation_started")

            if use_synthetic_data:

                strategies = ['BollingerBandsStrategy', 'RSIStrategy', 'MovingAverageCrossoverStrategy', 'StochasticOscillatorStrategy']
                training_records = self.synthetic_generator.create_training_dataset(
                    self.all_tickers[:20],
                    strategies
                )
            else:

                training_records = self._collect_real_data_with_fallback()

            if not training_records:
                raise ValueError("No training data could be generated")

            build_results["steps"].append("data_generation_completed")
            build_results["data_stats"] = {
                "total_records": len(training_records),
                "unique_tickers": len(set(r['ticker'] for r in training_records)),
                "unique_strategies": len(set(r['strategy_name'] for r in training_records))
            }

            self.logger.info("Step 2: Preparing training data...")
            build_results["steps"].append("data_preparation_started")

            X, y = self._prepare_training_data(training_records)

            build_results["steps"].append("data_preparation_completed")
            build_results["feature_stats"] = {
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_names": X.columns.tolist()
            }

            self.logger.info("Step 3: Training models...")
            build_results["steps"].append("model_training_started")

            training_results = self.model_trainer.train_models(X, y)

            build_results["steps"].append("model_training_completed")
            build_results["performance"] = training_results

            model_filepath = self.model_trainer.save_models()
            build_results["model_filepath"] = model_filepath

            self._is_trained = True
            self._training_info = build_results

            build_results["status"] = "completed"
            build_results["message"] = "Enhanced recommendation system built successfully"

            self.logger.info("Enhanced recommendation system build completed successfully")
            return build_results

        except Exception as e:
            self.logger.error(f"Failed to build recommendation system: {e}")
            build_results["status"] = "failed"
            build_results["errors"].append(str(e))
            return build_results

    def _collect_real_data_with_fallback(self) -> List[Dict[str, Any]]:

        real_records = []
        failed_tickers = []

        for ticker in self.all_tickers[:10]:
            try:
                end_date = date.today()
                start_date = end_date - timedelta(days=252)

                data = self.data_fetcher.fetch_ohlcv(ticker, start_date, end_date)

                if len(data) > 50:

                    pass
                else:
                    failed_tickers.append(ticker)

            except Exception as e:
                self.logger.warning(f"Failed to fetch real data for {ticker}: {e}")
                failed_tickers.append(ticker)

        self.logger.info("Using synthetic data for training due to API limitations")
        strategies = ['BollingerBandsStrategy', 'RSIStrategy', 'MovingAverageCrossoverStrategy', 'StochasticOscillatorStrategy']
        synthetic_records = self.synthetic_generator.create_training_dataset(
            self.all_tickers[:20], strategies
        )

        return synthetic_records

    def _prepare_training_data(self, records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series]:

        data = []

        for record in records:
            row = record['market_features'].copy()
            row['ticker'] = record['ticker']
            row['strategy_name'] = record['strategy_name']

            for key, value in record['performance_metrics'].items():
                row[f'perf_{key}'] = value

            data.append(row)

        df = pd.DataFrame(data)

        target_data = []
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            best_strategy = ticker_data.loc[ticker_data['perf_sharpe_ratio'].idxmax(), 'strategy_name']

            for _, row in ticker_data.iterrows():
                target_data.append(best_strategy)

        feature_cols = [col for col in df.columns if col not in ['ticker', 'strategy_name'] and not col.startswith('perf_')]
        X = df[feature_cols].fillna(0)

        y = pd.Series(target_data)

        return X, y

    def recommend_strategy_for_ticker(
        self,
        ticker: str,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:

        if not self._is_trained:
            if not self.model_trainer.get_latest_models():
                return {
                    "error": "No trained models available. Please build the recommendation system first.",
                    "ticker": ticker
                }
            self._is_trained = True

        try:
            self.logger.info(f"Generating strategy recommendation for {ticker}")

            market_features = self._get_market_features_for_ticker(ticker)

            if not market_features:
                return {
                    "error": f"Could not analyze market conditions for {ticker}",
                    "ticker": ticker
                }

            feature_data = pd.DataFrame([market_features])

            prediction_results = self.model_trainer.predict(feature_data, use_ensemble=True)

            recommendation = {
                "ticker": ticker,
                "recommended_strategy": prediction_results.get("ensemble_prediction"),
                "confidence": prediction_results.get("ensemble_confidence", 0.0),
                "individual_predictions": prediction_results.get("individual_predictions", {}),
                "timestamp": datetime.now().isoformat()
            }

            if include_reasoning:
                recommendation["market_analysis"] = self._analyze_market_conditions(market_features)
                recommendation["reasoning"] = self._generate_strategy_reasoning(
                    recommendation["recommended_strategy"],
                    market_features
                )

            return recommendation

        except Exception as e:
            self.logger.error(f"Failed to generate recommendation for {ticker}: {e}")
            return {
                "error": f"Failed to generate recommendation: {str(e)}",
                "ticker": ticker
            }

    def recommend_strategy_and_ticker(
        self,
        criteria: Dict[str, Any],
        max_recommendations: int = 5
    ) -> Dict[str, Any]:

        if not self._is_trained:
            if not self.model_trainer.get_latest_models():
                return {
                    "error": "No trained models available. Please build the recommendation system first."
                }
            self._is_trained = True

        try:
            self.logger.info("Generating strategy and ticker recommendations based on criteria")

            risk_tolerance = criteria.get('risk_tolerance', 'medium')
            investment_horizon = criteria.get('investment_horizon', 'medium')
            market_preference = criteria.get('market_preference', 'any')
            sector_preference = criteria.get('sector_preference', 'any')

            candidate_tickers = self._filter_tickers_by_sector(sector_preference)

            recommendations = []

            for ticker in candidate_tickers[:20]:
                try:

                    market_features = self._get_market_features_for_ticker(ticker)

                    if not market_features:
                        continue

                    strategy_rec = self.recommend_strategy_for_ticker(ticker, include_reasoning=False)

                    if 'error' in strategy_rec:
                        continue

                    compatibility_score = self._calculate_criteria_compatibility(
                        market_features,
                        strategy_rec['recommended_strategy'],
                        criteria
                    )

                    risk_score = self._calculate_risk_score(market_features)

                    if self._matches_risk_tolerance(risk_score, risk_tolerance):
                        recommendations.append({
                            'ticker': ticker,
                            'recommended_strategy': strategy_rec['recommended_strategy'],
                            'compatibility_score': compatibility_score,
                            'risk_score': risk_score,
                            'confidence': strategy_rec.get('confidence', 0.0),
                            'market_features': market_features
                        })

                except Exception as e:
                    self.logger.warning(f"Failed to analyze {ticker}: {e}")
                    continue

            recommendations.sort(
                key=lambda x: (x['compatibility_score'] * x['confidence']),
                reverse=True
            )

            top_recommendations = recommendations[:max_recommendations]

            result = {
                "criteria": criteria,
                "recommendations": top_recommendations,
                "total_analyzed": len(candidate_tickers),
                "total_matches": len(recommendations),
                "timestamp": datetime.now().isoformat()
            }

            if top_recommendations:
                result["summary"] = self._generate_recommendation_summary(top_recommendations, criteria)

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate criteria-based recommendations: {e}")
            return {
                "error": f"Failed to generate recommendations: {str(e)}"
            }

    def _get_market_features_for_ticker(self, ticker: str) -> Dict[str, float]:

        try:

            end_date = date.today()
            start_date = end_date - timedelta(days=252)

            try:
                data = self.data_fetcher.fetch_ohlcv(ticker, start_date, end_date)
                if len(data) > 50:
                    return self._calculate_market_features_from_data(data)
            except:
                pass

            synthetic_data = self.synthetic_generator.generate_market_data(ticker, start_date, end_date)
            if not synthetic_data.empty:
                return self.synthetic_generator._calculate_market_features(synthetic_data)

            return {}

        except Exception as e:
            self.logger.warning(f"Failed to get market features for {ticker}: {e}")
            return {}

    def _calculate_market_features_from_data(self, data: pd.DataFrame) -> Dict[str, float]:

        try:
            features = {}

            returns = data['Close'].pct_change().dropna()
            features['volatility'] = returns.std() * np.sqrt(252)
            features['mean_return'] = returns.mean() * 252
            features['total_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
            features['skewness'] = returns.skew() if len(returns) > 3 else 0
            features['kurtosis'] = returns.kurtosis() if len(returns) > 3 else 0

            ma_20 = data['Close'].rolling(20).mean()
            features['ma_trend'] = (ma_20.iloc[-1] / ma_20.iloc[-21] - 1) if len(ma_20) > 21 else 0

            rolling_vol = returns.rolling(20).std()
            features['vol_regime'] = rolling_vol.iloc[-1] / rolling_vol.mean() if len(rolling_vol) > 20 else 1

            for key, value in features.items():
                if pd.isna(value):
                    features[key] = 0.0

            return features

        except Exception as e:
            self.logger.warning(f"Failed to calculate market features: {e}")
            return {}

    def _filter_tickers_by_sector(self, sector_preference: str) -> List[str]:

        if sector_preference == 'any' or sector_preference not in self.ticker_categories:
            return self.all_tickers

        return self.ticker_categories[sector_preference]

    def _calculate_criteria_compatibility(
        self,
        market_features: Dict[str, float],
        strategy: str,
        criteria: Dict[str, Any]
    ) -> float:

        score = 0.0

        market_pref = criteria.get('market_preference', 'any')
        volatility = market_features.get('volatility', 0.2)
        total_return = market_features.get('total_return', 0)

        if market_pref == 'trending' and abs(total_return) > 0.15:
            score += 0.3
        elif market_pref == 'volatile' and volatility > 0.25:
            score += 0.3
        elif market_pref == 'stable' and volatility < 0.2:
            score += 0.3
        elif market_pref == 'any':
            score += 0.2

        strategy_fits = {
            'BollingerBandsStrategy': volatility > 0.2,
            'MovingAverageCrossoverStrategy': abs(total_return) > 0.1,
            'RSIStrategy': volatility > 0.15,
            'StochasticOscillatorStrategy': True
        }

        if strategy_fits.get(strategy, False):
            score += 0.4
        else:
            score += 0.1

        horizon = criteria.get('investment_horizon', 'medium')
        if horizon == 'long' and strategy == 'MovingAverageCrossoverStrategy':
            score += 0.3
        elif horizon == 'short' and strategy in ['RSIStrategy', 'StochasticOscillatorStrategy']:
            score += 0.3
        else:
            score += 0.2

        return min(score, 1.0)

    def _calculate_risk_score(self, market_features: Dict[str, float]) -> float:

        volatility = market_features.get('volatility', 0.2)
        vol_regime = market_features.get('vol_regime', 1.0)

        risk_score = min(volatility / 0.5 + (vol_regime - 1) / 2, 1.0)
        return max(risk_score, 0.0)

    def _matches_risk_tolerance(self, risk_score: float, risk_tolerance: str) -> bool:

        if risk_tolerance == 'low':
            return risk_score < 0.4
        elif risk_tolerance == 'medium':
            return 0.3 <= risk_score <= 0.7
        elif risk_tolerance == 'high':
            return risk_score > 0.5

        return True

    def _analyze_market_conditions(self, market_features: Dict[str, float]) -> Dict[str, Any]:

        analysis = {}

        volatility = market_features.get('volatility', 0.2)
        total_return = market_features.get('total_return', 0)
        vol_regime = market_features.get('vol_regime', 1.0)

        if volatility > 0.3:
            analysis['volatility_regime'] = 'High'
        elif volatility < 0.15:
            analysis['volatility_regime'] = 'Low'
        else:
            analysis['volatility_regime'] = 'Normal'

        if total_return > 0.15:
            analysis['trend'] = 'Strong Uptrend'
        elif total_return < -0.15:
            analysis['trend'] = 'Strong Downtrend'
        elif abs(total_return) > 0.05:
            analysis['trend'] = 'Weak Trend'
        else:
            analysis['trend'] = 'Sideways'

        if vol_regime > 1.3:
            analysis['market_efficiency'] = 'Low (High opportunity for active strategies)'
        else:
            analysis['market_efficiency'] = 'Normal'

        return analysis

    def _generate_strategy_reasoning(self, strategy: str, market_features: Dict[str, float]) -> str:

        volatility = market_features.get('volatility', 0.2)
        total_return = market_features.get('total_return', 0)

        reasoning_map = {
            'BollingerBandsStrategy': f"Market volatility ({volatility:.1%}) creates opportunities for mean-reversion trading around Bollinger Bands.",
            'MovingAverageCrossoverStrategy': f"Strong directional movement ({total_return:+.1%}) favors trend-following with moving average signals.",
            'RSIStrategy': f"Current market conditions support momentum-based RSI signals for entry/exit timing.",
            'StochasticOscillatorStrategy': f"Market oscillations provide good opportunities for stochastic-based trading signals."
        }

        return reasoning_map.get(strategy, f"Machine learning analysis indicates {strategy} is optimal for current market conditions.")

    def _generate_recommendation_summary(self, recommendations: List[Dict[str, Any]], criteria: Dict[str, Any]) -> Dict[str, Any]:

        if not recommendations:
            return {}

        strategies = [rec['recommended_strategy'] for rec in recommendations]
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        avg_risk = np.mean([rec['risk_score'] for rec in recommendations])

        top_rec = recommendations[0]

        summary = {
            'total_recommendations': len(recommendations),
            'top_recommendation': {
                'ticker': top_rec['ticker'],
                'strategy': top_rec['recommended_strategy'],
                'score': top_rec['compatibility_score']
            },
            'strategy_distribution': strategy_counts,
            'average_risk_score': avg_risk,
            'risk_level': 'Low' if avg_risk < 0.4 else 'Medium' if avg_risk < 0.7 else 'High',
            'criteria_match': 'Good' if top_rec['compatibility_score'] > 0.7 else 'Moderate'
        }

        return summary

    def get_system_status(self) -> Dict[str, Any]:

        status = {
            "is_trained": self._is_trained,
            "available_features": [
                "Feature 1: Ticker → Strategy Recommendation",
                "Feature 2: Criteria → Strategy + Ticker Recommendation"
            ],
            "cache_directory": self.cache_dir,
            "training_info": self._training_info,
            "supported_sectors": list(self.ticker_categories.keys()),
            "total_tickers": len(self.all_tickers)
        }

        if self._is_trained:
            status["model_summary"] = self.model_trainer.get_model_summary()

        return status
