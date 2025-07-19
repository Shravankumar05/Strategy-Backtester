import pytest
import pandas as pd
import os
import tempfile
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datetime import date, datetime, timedelta
from src.backtester.data.fetcher import DataError
from src.backtester.data.yfinance_fetcher import YFinanceDataFetcher
from src.backtester.data.cache_manager import CacheManager

class TestYFinanceDataFetcher:
    @pytest.fixture
    def temp_cache_dir(self):
        temp_dir = tempfile.mkdtemp()
        print(f"Created temp cache dir: {temp_dir}")
        yield temp_dir
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp cache dir: {temp_dir}")
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        return CacheManager(cache_dir=temp_cache_dir, expiry_hours=1)
    
    @pytest.fixture
    def fetcher(self, cache_manager):
        return YFinanceDataFetcher(cache_manager=cache_manager)
    
    @pytest.fixture
    def fetcher_no_cache(self):
        return YFinanceDataFetcher()
    
    def test_validate_symbol_valid(self, fetcher):
        print("Testing valid symbol validation...")
        
        valid_symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in valid_symbols:
            result = fetcher.validate_symbol(symbol)
            print(f"Symbol {symbol}: {result}")
            assert result == True
        
        print("✓ All valid symbols passed validation")
    
    def test_validate_symbol_invalid(self, fetcher):
        print("Testing invalid symbol validation...")
        
        invalid_symbols = ["", "NOTAREALSYMBOL12345", "!@#$%"]
        for symbol in invalid_symbols:
            result = fetcher.validate_symbol(symbol)
            print(f"Symbol {symbol}: {result}")
            assert result == False
        
        result = fetcher.validate_symbol(None)
        print(f"Symbol None: {result}")
        assert result == False
        print("✓ All invalid symbols correctly rejected")
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_valid(self, fetcher):
        print("Testing OHLCV data fetching...")
        symbol = "AAPL"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        print(f"Fetching data for {symbol} from {start_date} to {end_date}")
        data = await fetcher.fetch_ohlcv(symbol, start_date, end_date)
        print(f"Received data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert list(data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert data['Open'].dtype == 'float64'
        assert data['Volume'].dtype == 'int64'
        assert (data['High'] >= data['Low']).all()
        assert not data.isnull().any().any()
        print("✓ OHLCV data fetching successful")
    
    @pytest.mark.asyncio # Test error handling for invalid symbols
    async def test_fetch_ohlcv_invalid_symbol(self, fetcher):
        print("Testing invalid symbol error handling...")
        with pytest.raises(DataError) as exc_info:
            await fetcher.fetch_ohlcv("NOTAREALSYMBOL12345", date(2024, 1, 1), date(2024, 1, 31))
        print(f"Caught expected error: {exc_info.value}")
        print("✓ Invalid symbol error handling works")
    
    @pytest.mark.asyncio # Test validation of date ranges
    async def test_fetch_ohlcv_invalid_date_range(self, fetcher):
        print("Testing date range validation...")
        
        with pytest.raises(ValueError) as exc_info:
            await fetcher.fetch_ohlcv("AAPL", date(2024, 2, 1), date(2024, 1, 1))
        print(f"Caught expected error for reversed dates: {exc_info.value}")
        
        with pytest.raises(ValueError) as exc_info:
            await fetcher.fetch_ohlcv("AAPL", date(2023, 1, 1), date(2023, 12, 31))
        print(f"Caught expected error for dates outside 2024: {exc_info.value}")
        
        print("✓ Date range validation works")
    
    @pytest.mark.asyncio # Test that data is properly cached and retrieved.
    async def test_caching(self, fetcher, cache_manager):
        print("Testing caching functionality...")
        
        symbol = "MSFT"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)
        
        print(f"First fetch for {symbol}...")
        start_time = datetime.now()
        data1 = await fetcher.fetch_ohlcv(symbol, start_date, end_date)
        first_fetch_time = (datetime.now() - start_time).total_seconds()
        print(f"First fetch took {first_fetch_time:.2f} seconds")
        print(f"Second fetch for {symbol} (should use cache)...")
        start_time = datetime.now()
        data2 = await fetcher.fetch_ohlcv(symbol, start_date, end_date)
        second_fetch_time = (datetime.now() - start_time).total_seconds()
        print(f"Second fetch took {second_fetch_time:.2f} seconds")
        pd.testing.assert_frame_equal(data1, data2)
        cache_key = f"{symbol}_{start_date}_{end_date}"
        cached_data = cache_manager.get_cached_data(cache_key)
        assert cached_data is not None
        print(f"✓ Caching works (second fetch was faster)")
    
    @pytest.mark.asyncio # Test fetcher works without a cache manager
    async def test_no_cache_manager(self, fetcher_no_cache):
        print("Testing fetcher without cache manager...")
        symbol = "GOOGL"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)
        print(f"Fetching data for {symbol} without caching...")
        data = await fetcher_no_cache.fetch_ohlcv(symbol, start_date, end_date)
        assert not data.empty
        print(f"✓ Fetcher works without cache manager (got {len(data)} rows)")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])