import pandas as pd
from typing import Optional
from datetime import date
from abc import abstractmethod, ABC

class DataFetcher(ABC):
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        pass

class DataErorr(Exception):
    pass