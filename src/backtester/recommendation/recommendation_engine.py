import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import os
from .data_collector import DataCollector
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from ..data.yfinance_fetcher import YFinanceDataFetcher
from ..data.cache_manager import CacheManager

class RecommendationEngine:
    def __init__(self, cache_dir: str = "recommendation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.data_collector = DataCollector(cache_dir)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(cache_dir)
        self.data_fetcher = YFinanceDataFetcher(CacheManager())
        self._is_trained = False
        self._training_info = {}

    def build_recommendation_system(self, retrain: bool = False, tickers: Optional[List[str]] = None, lookback_days: int = 252, target_metric: str = "sharpe_ratio", use_grid_search: bool = False) -> Dict[str, Any]:
        self.logger.info("Starting recommendation system build...")
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
            self.logger.info("Step 1: Collecting training data...")
            build_results["steps"].append("data_collection_started")
            training_records = self.data_collector.collect_training_data(tickers=tickers, lookback_days=lookback_days)

            if not training_records:
                raise ValueError("No training data collected")

            build_results["steps"].append("data_collection_completed")
            build_results["data_stats"] = {
                "total_records": len(training_records),
                "unique_tickers": len(set(r.ticker for r in training_records)),
                "unique_strategies": len(set(r.strategy_name for r in training_records))
            }

            data_filepath = self.data_collector.save_training_data(training_records)
            build_results["data_filepath"] = data_filepath
            self.logger.info("Step 2: Engineering features...")
            build_results["steps"].append("feature_engineering_started")
            X, y, feature_names = self.feature_engineer.prepare_training_data(training_records, target_metric=target_metric)
            build_results["steps"].append("feature_engineering_completed")
            build_results["feature_stats"] = {
                "n_samples": len(X),
                "n_features": len(feature_names),
                "feature_names": feature_names
            }

            self.logger.info("Step 3: Training models...")
            build_results["steps"].append("model_training_started")
            training_results = self.model_trainer.train_models(X, y, use_grid_search=use_grid_search)
            build_results["steps"].append("model_training_completed")
            build_results["performance"] = training_results
            model_filepath = self.model_trainer.save_models()
            build_results["model_filepath"] = model_filepath
            self._is_trained = True
            self._training_info = build_results
            build_results["status"] = "completed"
            build_results["message"] = "Recommendation system built successfully"
            self.logger.info("Recommendation system build completed successfully")
            return build_results

        except Exception as e:
            self.logger.error(f"Failed to build recommendation system: {e}")
            build_results["status"] = "failed"
            build_results["errors"].append(str(e))
            return build_results

    def recommend_strategy(self, ticker: str, lookback_days: int = 252, include_confidence: bool = True, include_explanations: bool = True) -> Dict[str, Any]:
        if not self._is_trained:
            if not self.model_trainer.get_latest_models():
                return {
                    "error": "No trained models available. Please build the recommendation system first.",
                    "ticker": ticker
                }
            self._is_trained = True
        try:
            self.logger.info(f"Generating recommendation for {ticker}")
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days + 50)
            data = self.data_fetcher.fetch_ohlcv(ticker, start_date, end_date)
            if len(data) < 50:
                return {
                    "error": f"Insufficient data for {ticker}. Need at least 50 data points, got {len(data)}",
                    "ticker": ticker
                }

            if len(data) > lookback_days:
                data = data.tail(lookback_days)
            feature_data = self.feature_engineer.prepare_prediction_data(data)
            prediction_results = self.model_trainer.predict(feature_data, use_ensemble=True)
            recommendation = {
                "ticker": ticker,
                "recommended_strategy": prediction_results.get("ensemble_prediction"),
                "data_period": {
                    "start_date": data.index[0].strftime('%Y-%m-%d'),
                    "end_date": data.index[-1].strftime('%Y-%m-%d'),
                    "data_points": len(data)
                },
                "prediction_timestamp": datetime.now().isoformat()
            }

            if include_confidence:
                recommendation["individual_predictions"] = prediction_results["individual_predictions"]
                recommendation["ensemble_confidence"] = prediction_results.get("ensemble_confidence", 0.0)
                if "prediction_probabilities" in prediction_results:
                    recommendation["strategy_probabilities"] = self._aggregate_probabilities(
                        prediction_results["prediction_probabilities"]
                    )

            if include_explanations:
                recommendation["market_analysis"] = self._generate_market_analysis(data, feature_data)
                recommendation["recommendation_reasoning"] = self._generate_reasoning(recommendation["recommended_strategy"], feature_data)
            return recommendation

        except Exception as e:
            self.logger.error(f"Failed to generate recommendation for {ticker}: {e}")
            return {
                "error": f"Failed to generate recommendation: {str(e)}",
                "ticker": ticker
            }

    def _aggregate_probabilities(self, prob_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        strategy_probs = {}
        all_strategies = set()
        for model_probs in prob_dict.values():
            all_strategies.update(model_probs.keys())

        for strategy in all_strategies:
            probs = [model_probs.get(strategy, 0.0) for model_probs in prob_dict.values()]
            strategy_probs[strategy] = np.mean(probs)
        return strategy_probs

    def _generate_market_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        try:
            analysis = {}
            returns = data['Close'].pct_change().dropna()
            analysis["price_trend"] = {
                "total_return": ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100,
                "direction": "upward" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "downward"
            }
            analysis["volatility"] = {
                "level": returns.std() * np.sqrt(252) * 100,
                "regime": "high" if returns.std() > returns.rolling(50).std().mean() else "normal"
            }
            analysis["volume_pattern"] = {
                "average_volume": data['Volume'].mean(),
                "recent_vs_average": data['Volume'].tail(10).mean() / data['Volume'].mean()
            }
            ma_20 = data['Close'].rolling(20).mean()
            analysis["technical_signals"] = {
                "above_ma20": data['Close'].iloc[-1] > ma_20.iloc[-1],
                "ma20_trend": "upward" if ma_20.iloc[-1] > ma_20.iloc[-10] else "downward"
            }
            return analysis

        except Exception as e:
            self.logger.warning(f"Failed to generate market analysis: {e}")
            return {}

    def _generate_reasoning(self, recommended_strategy: str, features: pd.DataFrame) -> Dict[str, str]:
        try:
            reasoning = {
                "primary_factors": [],
                "market_conditions": "",
                "strategy_rationale": ""
            }
            strategy_rationales = {
                "BollingerBandsStrategy": "Market conditions suggest price movements that would benefit from a mean reversion signals when price touches bands.",
                "RSIStrategy": "Current market volatility and momentum patterns align well with RSI-based overbought/oversold signals.",
                "MovingAverageCrossoverStrategy": "Market shows trending behavior that would benefit from moving average crossover signals.",
                "StochasticOscillatorStrategy": "Market momentum characteristics suggest stochastic oscillator signals would be effective for timing entries and exits."
            }
            reasoning["strategy_rationale"] = strategy_rationales.get(
                recommended_strategy,
                f"Machine learning models indicate {recommended_strategy} is optimal for current market conditions."
            )
            if not features.empty:
                feature_values = features.iloc[0]
                if 'market_volatility' in feature_values:
                    vol = feature_values['market_volatility']
                    if vol > 0.3:
                        reasoning["primary_factors"].append("High volatility environment")
                    elif vol < 0.15:
                        reasoning["primary_factors"].append("Low volatility environment")

                if 'market_mean_return' in feature_values:
                    ret = feature_values['market_mean_return']
                    if ret > 0.1:
                        reasoning["primary_factors"].append("Strong positive momentum")
                    elif ret < -0.1:
                        reasoning["primary_factors"].append("Negative market trend")

                if 'market_vol_regime' in feature_values:
                    vol_regime = feature_values['market_vol_regime']
                    if vol_regime > 1.2:
                        reasoning["primary_factors"].append("Elevated volatility regime")

            return reasoning
        except Exception as e:
            self.logger.warning(f"Failed to generate reasoning: {e}")
            return {"error": "Could not generate reasoning"}

    def get_system_status(self) -> Dict[str, Any]:
        status = {
            "is_trained": self._is_trained,
            "available_models": self.model_trainer.get_model_summary(),
            "cache_directory": self.cache_dir,
            "training_info": self._training_info
        }

        try:
            latest_data = self.data_collector.get_latest_training_data()
            if latest_data:
                status["training_data"] = {
                    "available": True,
                    "records_count": len(latest_data),
                    "last_updated": max(r.backtest_timestamp for r in latest_data).isoformat()
                }
            else:
                status["training_data"] = {"available": False}
        except Exception:
            status["training_data"] = {"available": False}

        return status

    def validate_ticker(self, ticker: str) -> Dict[str, Any]:
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=300)
            data = self.data_fetcher.fetch_ohlcv(ticker, start_date, end_date)

            validation = {
                "ticker": ticker,
                "valid": len(data) >= 50,
                "data_points": len(data),
                "date_range": {
                    "start": data.index[0].strftime('%Y-%m-%d') if not data.empty else None,
                    "end": data.index[-1].strftime('%Y-%m-%d') if not data.empty else None
                }
            }

            if not validation["valid"]:
                validation["error"] = f"Insufficient data: {len(data)} points (minimum 50 required)"

            return validation
        except Exception as e:
            return {
                "ticker": ticker,
                "valid": False,
                "error": f"Failed to validate ticker: {str(e)}"
            }