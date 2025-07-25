import pandas as pd
import os, sys, pytest
from datetime import date
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.backtester.data.fetcher import DataFetcher, DataError

class MockDataFetcher(DataFetcher):
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
    
    async def fetch_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        self._validate_date_range(start_date, end_date)
        
        if self.should_fail or symbol == "INVALID":
            raise DataError(f"Failed to fetch data for symbol: {symbol}")
        
        if symbol == "EMPTY":
            return pd.DataFrame()  # Test empty data handling
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = date_range[date_range.weekday < 5]  # Setting monday as 0 and friday as 4
        
        if len(date_range) == 0:
            return pd.DataFrame()
        
        base_price = 100.0
        data = []
        
        for i, dt in enumerate(date_range):
            open_price = base_price + i * 0.5
            high_price = open_price + 2.0
            low_price = open_price - 1.5
            close_price = open_price + 1.0
            volume = 1000000 + i * 10000
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        return self._validate_dataframe(df, symbol)
    
    def validate_symbol(self, symbol: str) -> bool:
        if not symbol or not isinstance(symbol, str):
            return False
        
        if len(symbol) > 5 or len(symbol) < 1:
            return False
        
        if not symbol.isalpha():
            return False
        
        invalid_symbols = ["INVALID", ""]
        return symbol.upper() not in invalid_symbols

class TestDataFetcher:
    def test_data_fetcher_is_abstract(self):
        with pytest.raises(TypeError):
            DataFetcher()
    
    def test_data_fetcher_requires_implementation(self):
        pass
        
        class IncompleteDataFetcher(DataFetcher):
            pass
        
        with pytest.raises(TypeError):
            IncompleteDataFetcher()
    
    @pytest.mark.asyncio
    async def test_mock_fetcher_success(self):
        fetcher = MockDataFetcher()
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)
        data = await fetcher.fetch_ohlcv("AAPL", start_date, end_date)
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert list(data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert data['Open'].dtype == 'float64'
        assert data['High'].dtype == 'float64'
        assert data['Low'].dtype == 'float64'
        assert data['Close'].dtype == 'float64'
        assert data['Volume'].dtype == 'int64'
        assert (data['High'] >= data['Open']).all()
        assert (data['High'] >= data['Close']).all()
        assert (data['Low'] <= data['Open']).all()
        assert (data['Low'] <= data['Close']).all()
        assert not data.isnull().any().any()
    
    @pytest.mark.asyncio
    async def test_mock_fetcher_invalid_symbol(self):
        fetcher = MockDataFetcher()
        with pytest.raises(DataError):
            await fetcher.fetch_ohlcv("INVALID", date(2024, 1, 1), date(2024, 1, 5))
    
    @pytest.mark.asyncio
    async def test_mock_fetcher_empty_data(self):
        fetcher = MockDataFetcher()
        with pytest.raises(DataError):
            await fetcher.fetch_ohlcv("EMPTY", date(2024, 1, 1), date(2024, 1, 5))
    
    @pytest.mark.asyncio
    async def test_mock_fetcher_invalid_date_range(self):
        fetcher = MockDataFetcher()
        with pytest.raises(ValueError):
            await fetcher.fetch_ohlcv("AAPL", date(2024, 1, 10), date(2024, 1, 5))
        
        with pytest.raises(ValueError): # dates outside 2024 should fail for now
            await fetcher.fetch_ohlcv("AAPL", date(2023, 1, 1), date(2023, 12, 31))
    
    def test_symbol_validation(self):
        fetcher = MockDataFetcher()
        assert fetcher.validate_symbol("AAPL") == True
        assert fetcher.validate_symbol("GOOGL") == True
        assert fetcher.validate_symbol("MSFT") == True
        assert fetcher.validate_symbol("") == False
        assert fetcher.validate_symbol("INVALID") == False
        assert fetcher.validate_symbol("123") == False
        assert fetcher.validate_symbol("TOOLONG") == False