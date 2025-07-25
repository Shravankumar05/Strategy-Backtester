import yfinance as yf
import pandas as pd
from datetime import date
from typing import Dict
from .fetcher import DataFetcher, DataError
from .cache_manager import CacheManager

class YFinanceDataFetcher(DataFetcher):
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self._valid_symbols_cache: Dict[str, bool] = {}
    
    def fetch_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        if not self.validate_symbol(symbol):
            raise DataError(f"Invalid symbol: {symbol}")
        
        self._validate_date_range(start_date, end_date)
        
        if self.cache_manager:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            cached_data = self.cache_manager.get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [str(col).capitalize() for col in df.columns]
            df = self._validate_dataframe(df, symbol)
            
            if self.cache_manager:
                self.cache_manager.cache_data(cache_key, df)
            
            return df
            
        except Exception as e:
            raise DataError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def validate_symbol(self, symbol: str) -> bool:
        if not symbol or not isinstance(symbol, str):
            return False
        
        if symbol in self._valid_symbols_cache:
            return self._valid_symbols_cache[symbol]
        
        if not symbol.strip():
            return False
        
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^")
        if not all(c in valid_chars for c in symbol.upper()):
            return False
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            is_valid = 'regularMarketPrice' in info or 'shortName' in info
            self._valid_symbols_cache[symbol] = is_valid
            return is_valid
            
        except Exception:
            self._valid_symbols_cache[symbol] = False
            return False