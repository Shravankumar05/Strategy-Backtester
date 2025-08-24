import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from .data_collector import BacktestRecord

class FeatureEngineer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_scaler = StandardScaler()
        self.strategy_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False

    def prepare_training_data(
        self,
        records: List[BacktestRecord],
        target_metric: str = "sharpe_ratio",
        min_records_per_ticker: int = 3
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:

        self.logger.info(f"Preparing training data from {len(records)} records")

        df = self._records_to_dataframe(records)

        ticker_counts = df['ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= min_records_per_ticker].index
        df = df[df['ticker'].isin(valid_tickers)]

        self.logger.info(f"After filtering: {len(df)} records from {len(valid_tickers)} tickers")

        target_series = self._create_target_labels(df, target_metric)

        features_df = self._create_features(df)

        common_index = features_df.index.intersection(target_series.index)
        features_df = features_df.loc[common_index]
        target_series = target_series.loc[common_index]

        self.feature_columns = features_df.columns.tolist()
        self.is_fitted = True

        self.logger.info(f"Created {len(features_df)} training samples with {len(features_df.columns)} features")
        return features_df, target_series, self.feature_columns

    def _records_to_dataframe(self, records: List[BacktestRecord]) -> pd.DataFrame:

        data = []

        for record in records:
            row = {
                'ticker': record.ticker,
                'strategy_name': record.strategy_name,
                'start_date': record.start_date,
                'end_date': record.end_date,
                'backtest_timestamp': record.backtest_timestamp
            }

            for key, value in record.metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
                    row[f'metric_{key}'] = value

            for key, value in record.market_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
                    row[f'market_{key}'] = value

            for key, value in record.data_quality.items():
                if isinstance(value, (int, float)):
                    row[f'quality_{key}'] = value

            for key, value in record.strategy_params.items():
                if isinstance(value, (int, float)):
                    row[f'param_{key}'] = value

            data.append(row)

        df = pd.DataFrame(data)

        df['record_id'] = df['ticker'] + '_' + df['strategy_name']
        df.set_index('record_id', inplace=True)

        return df

    def _create_target_labels(self, df: pd.DataFrame, target_metric: str) -> pd.Series:

        target_col = f'metric_{target_metric}'

        if target_col not in df.columns:
            raise ValueError(f"Target metric '{target_metric}' not found in data")

        best_strategies = df.groupby('ticker').apply(
            lambda group: group.loc[group[target_col].idxmax(), 'strategy_name']
        )

        target_series = pd.Series(index=df.index, dtype=str)

        for ticker, best_strategy in best_strategies.items():
            ticker_mask = df['ticker'] == ticker
            target_series.loc[ticker_mask] = best_strategy

        return target_series.dropna()

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:

        feature_cols = []

        market_cols = [col for col in df.columns if col.startswith('market_')]
        feature_cols.extend(market_cols)

        quality_cols = [col for col in df.columns if col.startswith('quality_')]
        feature_cols.extend(quality_cols)

        core_features = [
            'market_volatility', 'market_mean_return', 'market_total_return',
            'market_vol_regime', 'market_ma_trend', 'market_skewness',
            'quality_data_points', 'quality_extreme_moves'
        ]

        available_features = [col for col in core_features if col in df.columns]

        if not available_features:
            self.logger.warning("No core features available, using all market features")
            available_features = market_cols

        features_df = df[available_features].copy()

        features_df = features_df.fillna(features_df.median())

        features_df = self._add_engineered_features(features_df)

        return features_df

    def _add_engineered_features(self, features_df: pd.DataFrame) -> pd.DataFrame:

        try:

            if 'market_volatility' in features_df.columns:
                features_df['high_vol_regime'] = (features_df['market_volatility'] > features_df['market_volatility'].quantile(0.75)).astype(int)
                features_df['low_vol_regime'] = (features_df['market_volatility'] < features_df['market_volatility'].quantile(0.25)).astype(int)

            if 'market_mean_return' in features_df.columns:
                features_df['bull_market'] = (features_df['market_mean_return'] > 0.1).astype(int)
                features_df['bear_market'] = (features_df['market_mean_return'] < -0.1).astype(int)

            if 'market_total_return' in features_df.columns and 'market_volatility' in features_df.columns:
                features_df['trend_strength'] = np.abs(features_df['market_total_return']) / (features_df['market_volatility'] + 1e-6)

            if 'market_skewness' in features_df.columns and 'market_kurtosis' in features_df.columns:
                features_df['market_efficiency'] = np.abs(features_df['market_skewness']) + np.abs(features_df['market_kurtosis'] - 3)

        except Exception as e:
            self.logger.warning(f"Failed to add some engineered features: {e}")

        return features_df

    def prepare_prediction_data(self, ticker_data: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before making predictions")

        market_features = self._calculate_market_features_for_ticker(ticker_data)

        feature_vector = pd.DataFrame([market_features])

        for col in self.feature_columns:
            if col not in feature_vector.columns:
                if col.startswith('quality_'):

                    feature_vector[col] = len(ticker_data) if 'data_points' in col else 0
                else:
                    feature_vector[col] = 0

        feature_vector = feature_vector[self.feature_columns]

        feature_vector = feature_vector.fillna(feature_vector.median())

        return feature_vector

    def _calculate_market_features_for_ticker(self, data: pd.DataFrame) -> Dict[str, float]:

        try:
            features = {}

            returns = data['Close'].pct_change().dropna()
            features['market_volatility'] = returns.std() * np.sqrt(252)
            features['market_mean_return'] = returns.mean() * 252
            features['market_skewness'] = returns.skew()
            features['market_kurtosis'] = returns.kurtosis()

            features['market_total_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

            features['market_avg_volume'] = data['Volume'].mean()
            features['market_volume_volatility'] = data['Volume'].std() / data['Volume'].mean()

            features['market_price_level'] = data['Close'].iloc[-1]
            features['market_price_range'] = (data['High'].max() - data['Low'].min()) / data['Close'].mean()

            ma_20 = data['Close'].rolling(20).mean()
            features['market_ma_trend'] = (ma_20.iloc[-1] / ma_20.iloc[-21] - 1) if len(ma_20) > 21 else 0

            rolling_vol = returns.rolling(20).std()
            features['market_vol_regime'] = rolling_vol.iloc[-1] / rolling_vol.mean()

            features['quality_data_points'] = len(data)
            features['quality_missing_values'] = data.isnull().sum().sum()
            features['quality_zero_volume_days'] = (data['Volume'] == 0).sum()
            features['quality_extreme_moves'] = (returns.abs() > 0.2).sum()

            return features

        except Exception as e:
            self.logger.warning(f"Failed to calculate market features: {e}")
            return {}

    def get_feature_importance_interpretation(self, feature_names: List[str], importances: np.ndarray) -> Dict[str, str]:

        interpretations = {
            'market_volatility': 'Market volatility level - higher values may favor trend-following strategies',
            'market_mean_return': 'Average market return - positive values may favor buy-and-hold strategies',
            'market_total_return': 'Total return over period - indicates overall market direction',
            'market_vol_regime': 'Volatility regime - values >1 indicate high volatility periods',
            'market_ma_trend': 'Moving average trend strength - indicates trend persistence',
            'market_skewness': 'Return distribution skewness - indicates asymmetric price movements',
            'high_vol_regime': 'High volatility indicator - may favor mean-reversion strategies',
            'low_vol_regime': 'Low volatility indicator - may favor momentum strategies',
            'bull_market': 'Bull market indicator - favors long-biased strategies',
            'bear_market': 'Bear market indicator - may favor defensive strategies',
            'trend_strength': 'Trend strength measure - higher values favor trend-following',
            'market_efficiency': 'Market efficiency proxy - higher values indicate less efficient markets'
        }

        feature_importance = {}
        for feature, importance in zip(feature_names, importances):
            interpretation = interpretations.get(feature, f"Feature: {feature}")
            feature_importance[feature] = f"{interpretation} (Importance: {importance:.3f})"

        return feature_importance