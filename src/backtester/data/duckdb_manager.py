import duckdb
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import json

class DuckDBManager:
    def __init__(self, db_path: str = "backtester_data.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        try:
            with duckdb.connect(self.db_path) as conn:
                # Market data table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        ticker VARCHAR NOT NULL,
                        date DATE NOT NULL,
                        open DOUBLE NOT NULL,
                        high DOUBLE NOT NULL,
                        low DOUBLE NOT NULL,
                        close DOUBLE NOT NULL,
                        volume BIGINT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(ticker, date)
                    )
                """)
                
                # Market features table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_features (
                        ticker VARCHAR NOT NULL,
                        date DATE NOT NULL,
                        volatility DOUBLE,
                        mean_return DOUBLE,
                        total_return DOUBLE,
                        skewness DOUBLE,
                        kurtosis DOUBLE,
                        ma_trend DOUBLE,
                        vol_regime DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(ticker, date)
                    )
                """)
                
                # Strategy performance table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        ticker VARCHAR NOT NULL,
                        strategy_name VARCHAR NOT NULL,
                        start_date DATE NOT NULL,
                        end_date DATE NOT NULL,
                        sharpe_ratio DOUBLE,
                        total_return DOUBLE,
                        max_drawdown DOUBLE,
                        win_rate DOUBLE,
                        sortino_ratio DOUBLE,
                        calmar_ratio DOUBLE,
                        parameters JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(ticker, strategy_name, start_date, end_date)
                    )
                """)
                
                # RL agent experiences table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rl_experiences (
                        id INTEGER PRIMARY KEY,
                        episode INTEGER NOT NULL,
                        step INTEGER NOT NULL,
                        ticker VARCHAR NOT NULL,
                        state_json TEXT NOT NULL,
                        action INTEGER NOT NULL,
                        reward DOUBLE NOT NULL,
                        next_state_json TEXT,
                        done BOOLEAN NOT NULL,
                        strategy_used VARCHAR,
                        performance_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # RL training sessions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rl_training_sessions (
                        session_id VARCHAR NOT NULL,
                        agent_type VARCHAR NOT NULL,
                        hyperparameters JSON,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        total_episodes INTEGER,
                        final_reward DOUBLE,
                        model_path VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(session_id)
                    )
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON market_features(ticker, date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_ticker ON strategy_performance(ticker)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_rl_episode ON rl_experiences(episode)")
                
                self.logger.info(f"DuckDB database initialized at {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_market_data(self, ticker: str, data: pd.DataFrame) -> bool:
        try:
            with duckdb.connect(self.db_path) as conn:
                data_to_store = data.copy()
                data_to_store['ticker'] = ticker
                data_to_store['date'] = data_to_store.index
                data_to_store = data_to_store.reset_index(drop=True)
                column_mapping = {
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                data_to_store = data_to_store.rename(columns=column_mapping)
                
                required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
                data_to_store = data_to_store[required_cols]
                conn.execute("DELETE FROM market_data WHERE ticker = ?", [ticker])
                conn.register('temp_data', data_to_store)
                conn.execute("""
                    INSERT INTO market_data (ticker, date, open, high, low, close, volume)
                    SELECT ticker, date, open, high, low, close, volume FROM temp_data
                """)
                
                self.logger.info(f"Stored {len(data_to_store)} market data records for {ticker}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store market data for {ticker}: {e}")
            return False
    
    def get_market_data(self, ticker: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
        try:
            with duckdb.connect(self.db_path) as conn:
                query = "SELECT date, open, high, low, close, volume FROM market_data WHERE ticker = ?"
                params = [ticker]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date"
                result = conn.execute(query, params).fetchdf()
                
                if not result.empty:
                    result.set_index('date', inplace=True)
                    result.columns = [col.capitalize() for col in result.columns]
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve market data for {ticker}: {e}")
            return pd.DataFrame()
    
    def store_market_features(self, ticker: str, features: Dict[str, float], feature_date: date) -> bool:
        try:
            with duckdb.connect(self.db_path) as conn:
                features_data = {
                    'ticker': ticker,
                    'date': feature_date,
                    'volatility': features.get('volatility'),
                    'mean_return': features.get('mean_return'),
                    'total_return': features.get('total_return'),
                    'skewness': features.get('skewness'),
                    'kurtosis': features.get('kurtosis'),
                    'ma_trend': features.get('ma_trend'),
                    'vol_regime': features.get('vol_regime')
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO market_features 
                    (ticker, date, volatility, mean_return, total_return, skewness, kurtosis, ma_trend, vol_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    features_data['ticker'], features_data['date'], features_data['volatility'],
                    features_data['mean_return'], features_data['total_return'], features_data['skewness'],
                    features_data['kurtosis'], features_data['ma_trend'], features_data['vol_regime']
                ])
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store market features for {ticker}: {e}")
            return False
    
    def get_market_features(self, ticker: str, feature_date: date) -> Dict[str, float]:
        try:
            with duckdb.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT volatility, mean_return, total_return, skewness, kurtosis, ma_trend, vol_regime
                    FROM market_features 
                    WHERE ticker = ? AND date = ?
                """, [ticker, feature_date]).fetchone()
                
                if result:
                    return {
                        'volatility': result[0] or 0.0,
                        'mean_return': result[1] or 0.0,
                        'total_return': result[2] or 0.0,
                        'skewness': result[3] or 0.0,
                        'kurtosis': result[4] or 0.0,
                        'ma_trend': result[5] or 0.0,
                        'vol_regime': result[6] or 1.0
                    }
                
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve market features for {ticker}: {e}")
            return {}
    
    def store_strategy_performance(self, ticker: str, strategy_name: str, start_date: date, end_date: date, performance_metrics: Dict[str, float], parameters: Dict[str, Any] = None) -> bool:
        try:
            with duckdb.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO strategy_performance 
                    (ticker, strategy_name, start_date, end_date, sharpe_ratio, total_return, 
                     max_drawdown, win_rate, sortino_ratio, calmar_ratio, parameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    ticker, strategy_name, start_date, end_date,
                    performance_metrics.get('sharpe_ratio'),
                    performance_metrics.get('total_return'),
                    performance_metrics.get('max_drawdown'),
                    performance_metrics.get('win_rate'),
                    performance_metrics.get('sortino_ratio'),
                    performance_metrics.get('calmar_ratio'),
                    json.dumps(parameters) if parameters else None
                ])
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store strategy performance: {e}")
            return False
    
    def get_strategy_performance_history(self, ticker: str = None, strategy_name: str = None) -> pd.DataFrame:
        try:
            with duckdb.connect(self.db_path) as conn:
                query = """
                    SELECT ticker, strategy_name, start_date, end_date, sharpe_ratio, 
                           total_return, max_drawdown, win_rate, sortino_ratio, calmar_ratio,
                           parameters, created_at
                    FROM strategy_performance
                    WHERE 1=1
                """
                params = []
                
                if ticker:
                    query += " AND ticker = ?"
                    params.append(ticker)
                
                if strategy_name:
                    query += " AND strategy_name = ?"
                    params.append(strategy_name)
                
                query += " ORDER BY created_at DESC"
                return conn.execute(query, params).fetchdf()
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve strategy performance history: {e}")
            return pd.DataFrame()
    
    def store_rl_experience(self, episode: int, step: int, ticker: str, state: Dict[str, Any], action: int, reward: float, next_state: Dict[str, Any] = None, done: bool = False, strategy_used: str = None, performance_metrics: Dict[str, float] = None) -> bool:
        try:
            with duckdb.connect(self.db_path) as conn:
                unique_id = abs(hash(f"{episode}_{step}_{ticker}")) % (10**9)
                state_json = '{}'
                if state:
                    try:
                        state_json = json.dumps(state, default=str)
                    except (TypeError, ValueError):
                        state_json = json.dumps({'error': 'failed_to_serialize_state'})
                
                next_state_json = None
                if next_state:
                    try:
                        next_state_json = json.dumps(next_state, default=str)
                    except (TypeError, ValueError):
                        next_state_json = json.dumps({'error': 'failed_to_serialize_next_state'})
                
                performance_json = None
                if performance_metrics:
                    try:
                        performance_json = json.dumps(performance_metrics, default=str)
                    except (TypeError, ValueError):
                        performance_json = json.dumps({'error': 'failed_to_serialize_performance'})
                
                conn.execute("""
                    INSERT OR IGNORE INTO rl_experiences 
                    (id, episode, step, ticker, state_json, action, reward, next_state_json, done, 
                     strategy_used, performance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    unique_id, episode, step, ticker, 
                    state_json, 
                    action, reward,
                    next_state_json, 
                    done,
                    strategy_used, 
                    performance_json
                ])
                
                return True
                
        except Exception as e:
            if "Constraint Error" not in str(e) and "Conversion Error" not in str(e):
                self.logger.error(f"Failed to store RL experience: {e}")
            return False
    
    def get_rl_experiences(self, episode: int = None, limit: int = 1000) -> pd.DataFrame:
        try:
            with duckdb.connect(self.db_path) as conn:
                query = """
                    SELECT episode, step, ticker, state_json, action, reward, next_state_json, done,
                           strategy_used, performance_json, created_at
                    FROM rl_experiences
                    WHERE 1=1
                """
                params = []
                
                if episode is not None:
                    query += " AND episode = ?"
                    params.append(episode)
                
                query += " ORDER BY episode DESC, step ASC"
                if limit:
                    query += f" LIMIT {limit}"
                
                return conn.execute(query, params).fetchdf()
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve RL experiences: {e}")
            return pd.DataFrame()
    
    def store_rl_training_session(self, session_id: str, agent_type: str, hyperparameters: Dict[str, Any], total_episodes: int, final_reward: float, model_path: str = None) -> bool:
        try:
            with duckdb.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO rl_training_sessions 
                    (session_id, agent_type, hyperparameters, start_time, end_time, 
                     total_episodes, final_reward, model_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    session_id, agent_type, json.dumps(hyperparameters),
                    datetime.now(), datetime.now(), total_episodes, final_reward, model_path
                ])
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store RL training session: {e}")
            return False
    
    def get_available_tickers(self) -> List[str]:
        try:
            with duckdb.connect(self.db_path) as conn:
                result = conn.execute("SELECT DISTINCT ticker FROM market_data ORDER BY ticker").fetchall()
                return [row[0] for row in result]
                
        except Exception as e:
            self.logger.error(f"Failed to get available tickers: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        try:
            with duckdb.connect(self.db_path) as conn:
                summary = {}
                market_data_count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
                tickers_count = conn.execute("SELECT COUNT(DISTINCT ticker) FROM market_data").fetchone()[0]
                date_range = conn.execute("""
                    SELECT MIN(date) as min_date, MAX(date) as max_date 
                    FROM market_data
                """).fetchone()
                
                performance_count = conn.execute("SELECT COUNT(*) FROM strategy_performance").fetchone()[0]
                rl_count = conn.execute("SELECT COUNT(*) FROM rl_experiences").fetchone()[0]
                episodes_count = conn.execute("SELECT COUNT(DISTINCT episode) FROM rl_experiences").fetchone()[0]
                
                summary = {
                    'market_data_records': market_data_count,
                    'unique_tickers': tickers_count,
                    'date_range': {
                        'start': date_range[0] if date_range[0] else None,
                        'end': date_range[1] if date_range[1] else None
                    },
                    'strategy_performance_records': performance_count,
                    'rl_experiences': rl_count,
                    'rl_episodes': episodes_count,
                    'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Failed to get data summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_old: int = 90) -> bool:
        try:
            with duckdb.connect(self.db_path) as conn:
                cutoff_date = datetime.now().date() - pd.Timedelta(days=days_old)
                
                conn.execute("""
                    DELETE FROM rl_experiences 
                    WHERE created_at < ?
                """, [cutoff_date])
                
                conn.execute("""
                    DELETE FROM strategy_performance 
                    WHERE created_at < ?
                """, [cutoff_date])
                
                conn.execute("VACUUM")
                self.logger.info(f"Cleaned up data older than {days_old} days")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return False