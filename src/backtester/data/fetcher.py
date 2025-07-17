from abc import ABC, abstractmethod
from datetime import date
from typing import Optional
import pandas as pd


class DataError(Exception):
    pass

class DataFetcher(ABC): 
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        pass
    
    def _validate_date_range(self, start_date: date, end_date: date) -> None:
        if start_date > end_date:
            raise ValueError(f"Start date ({start_date}) cannot be after end date ({end_date})")
        
        if start_date.year != 2024 or end_date.year != 2024: # limiting the dates to 2024 for now but this can be changed easily in the future
            raise ValueError("Date range must be within 2024")
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty:
            raise DataError(f"No data available for symbol {symbol}")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataError(f"Missing required columns for {symbol}: {missing_columns}")
        
        df = df.fillna(method='ffill')
        df = df.dropna()
        
        if df.empty:
            raise DataError(f"No valid data remaining after cleaning for symbol {symbol}")
        
        return df[required_columns]