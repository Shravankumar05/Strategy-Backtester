import pytest
from datetime import date
from src.backtester.data.fetcher import DataFetcher, DataError


class MockDataFetcher(DataFetcher):
    """Mock implementation for testing."""
    
    async def fetch_ohlcv(self, symbol, start_date, end_date):
        if symbol == "INVALID":
            raise DataError("Invalid symbol")
        import pandas as pd
        return pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [95.0],
            'Close': [102.0],
            'Volume': [1000000]
        })
    
    def validate_symbol(self, symbol):
        return symbol != "INVALID"


def test_data_fetcher_interface():
    """Test that DataFetcher is properly defined as abstract."""
    with pytest.raises(TypeError):
        DataFetcher()


@pytest.mark.asyncio
async def test_mock_data_fetcher():
    """Test the mock implementation."""
    fetcher = MockDataFetcher()
    
    # Test valid symbol
    data = await fetcher.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert not data.empty
    assert list(data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Test invalid symbol
    with pytest.raises(DataError):
        await fetcher.fetch_ohlcv("INVALID", date(2024, 1, 1), date(2024, 1, 31))
    
    # Test symbol validation
    assert fetcher.validate_symbol("AAPL") == True
    assert fetcher.validate_symbol("INVALID") == False
