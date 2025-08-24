import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple
import logging

class SyntheticDataGenerator:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_market_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:

        try:

            total_days = (end_date - start_date).days
            trading_days = int(total_days * 0.7)

            if trading_days < 50:
                trading_days = 50

            dates = pd.date_range(start=start_date, periods=trading_days, freq='B')

            market_params = self._get_ticker_params(ticker)

            initial_price = market_params['initial_price']
            drift = market_params['drift']
            volatility = market_params['volatility']

            dt = 1/252
            returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), trading_days)

            price_series = initial_price * np.exp(np.cumsum(returns))

            data = []
            for i, (date, close) in enumerate(zip(dates, price_series)):

                noise = np.random.normal(0, volatility * 0.1)

                high = close * (1 + abs(noise) + np.random.uniform(0, 0.02))
                low = close * (1 - abs(noise) - np.random.uniform(0, 0.02))
                open_price = close * (1 + np.random.normal(0, volatility * 0.05))

                high = max(high, open_price, close)
                low = min(low, open_price, close)

                base_volume = market_params['base_volume']
                volume_noise = np.random.uniform(0.5, 2.0)
                volume = int(base_volume * volume_noise)

                data.append({
                    'Date': date,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(close, 2),
                    'Volume': volume
                })

            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)

            self.logger.info(f"Generated synthetic data for {ticker}: {len(df)} data points")
            return df

        except Exception as e:
            self.logger.error(f"Failed to generate synthetic data for {ticker}: {e}")
            return pd.DataFrame()

    def _get_ticker_params(self, ticker: str) -> Dict[str, Any]:

        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        financial_stocks = ['JPM', 'BAC', 'GS', 'WFC', 'C']
        consumer_stocks = ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD']
        energy_stocks = ['XOM', 'CVX', 'COP', 'SLB']
        etf_stocks = ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA']

        if ticker in tech_stocks:
            return {
                'initial_price': np.random.uniform(100, 300),
                'drift': np.random.uniform(0.08, 0.15),
                'volatility': np.random.uniform(0.25, 0.35),
                'base_volume': np.random.randint(20000000, 50000000)
            }
        elif ticker in financial_stocks:
            return {
                'initial_price': np.random.uniform(30, 150),
                'drift': np.random.uniform(0.05, 0.10),
                'volatility': np.random.uniform(0.20, 0.30),
                'base_volume': np.random.randint(15000000, 30000000)
            }
        elif ticker in consumer_stocks:
            return {
                'initial_price': np.random.uniform(50, 200),
                'drift': np.random.uniform(0.03, 0.08),
                'volatility': np.random.uniform(0.15, 0.25),
                'base_volume': np.random.randint(5000000, 20000000)
            }
        elif ticker in energy_stocks:
            return {
                'initial_price': np.random.uniform(40, 120),
                'drift': np.random.uniform(-0.02, 0.08),
                'volatility': np.random.uniform(0.30, 0.45),
                'base_volume': np.random.randint(10000000, 25000000)
            }
        elif ticker in etf_stocks:
            return {
                'initial_price': np.random.uniform(100, 400),
                'drift': np.random.uniform(0.06, 0.10),
                'volatility': np.random.uniform(0.18, 0.22),
                'base_volume': np.random.randint(50000000, 100000000)
            }
        else:

            return {
                'initial_price': np.random.uniform(50, 200),
                'drift': np.random.uniform(0.02, 0.12),
                'volatility': np.random.uniform(0.20, 0.35),
                'base_volume': np.random.randint(1000000, 20000000)
            }

    def create_training_dataset(self, tickers: List[str], strategies: List[str]) -> List[Dict[str, Any]]:

        training_records = []

        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        for ticker in tickers:
            try:

                data = self.generate_market_data(ticker, start_date, end_date)

                if data.empty:
                    continue

                market_features = self._calculate_market_features(data)

                for strategy_name in strategies:
                    performance_metrics = self._simulate_strategy_performance(
                        strategy_name, market_features
                    )

                    record = {
                        'ticker': ticker,
                        'strategy_name': strategy_name,
                        'market_features': market_features,
                        'performance_metrics': performance_metrics,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                    training_records.append(record)

            except Exception as e:
                self.logger.warning(f"Failed to create training data for {ticker}: {e}")
                continue

        self.logger.info(f"Created {len(training_records)} synthetic training records")
        return training_records

    def _calculate_market_features(self, data: pd.DataFrame) -> Dict[str, float]:

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

    def _simulate_strategy_performance(self, strategy_name: str, market_features: Dict[str, float]) -> Dict[str, float]:

        try:

            strategy_preferences = {
                'BollingerBandsStrategy': {
                    'high_volatility': 0.8,
                    'trending': 0.3,
                    'mean_reverting': 0.9
                },
                'RSIStrategy': {
                    'high_volatility': 0.7,
                    'trending': 0.4,
                    'mean_reverting': 0.8
                },
                'MovingAverageCrossoverStrategy': {
                    'high_volatility': 0.4,
                    'trending': 0.9,
                    'mean_reverting': 0.3
                },
                'StochasticOscillatorStrategy': {
                    'high_volatility': 0.6,
                    'trending': 0.5,
                    'mean_reverting': 0.7
                }
            }

            if strategy_name not in strategy_preferences:

                return {
                    'sharpe_ratio': np.random.normal(0.5, 0.3),
                    'total_return': np.random.normal(0.08, 0.15),
                    'max_drawdown': np.random.uniform(0.05, 0.25),
                    'win_rate': np.random.uniform(0.4, 0.7)
                }

            prefs = strategy_preferences[strategy_name]

            volatility = market_features.get('volatility', 0.2)
            total_return = market_features.get('total_return', 0)
            vol_regime = market_features.get('vol_regime', 1)

            high_vol_mult = prefs['high_volatility'] if volatility > 0.25 else 1.0
            trend_mult = prefs['trending'] if abs(total_return) > 0.15 else 1.0
            mean_rev_mult = prefs['mean_reverting'] if vol_regime > 1.2 else 1.0

            base_sharpe = 0.8
            base_return = 0.10
            base_drawdown = 0.15
            base_win_rate = 0.55

            performance_mult = (high_vol_mult + trend_mult + mean_rev_mult) / 3
            noise = np.random.normal(1.0, 0.2)

            metrics = {
                'sharpe_ratio': max(0, base_sharpe * performance_mult * noise),
                'total_return': base_return * performance_mult * noise,
                'max_drawdown': base_drawdown / (performance_mult * noise),
                'win_rate': np.clip(base_win_rate * performance_mult * noise, 0.2, 0.9)
            }

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to simulate performance for {strategy_name}: {e}")
            return {
                'sharpe_ratio': 0.5,
                'total_return': 0.05,
                'max_drawdown': 0.15,
                'win_rate': 0.5
            }
