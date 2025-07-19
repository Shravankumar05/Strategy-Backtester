import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.backtester.data.preprocessor import DataPreprocessor, DataQualityReport
from src.backtester.data.fetcher import DataError


class TestDataPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor(enable_logging=False)
    
    @pytest.fixture
    def sample_good_data(self):
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = { # mock data
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_bad_data(self):
        # Problematic OHLCV data.
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = {
            'Open': [100.0, 101.0, np.nan, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [95.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0],  # High < Low on first day
            'Low': [105.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],  # Low > High on first day
            'Close': [102.0, 103.0, 104.0, -105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],  # Negative close
            'Volume': [1000, -1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]  # Negative volume
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_outlier_data(self):
        # sample with outliers
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 1090.0],  # Outlier
            'High': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 1140.0],  # Outlier
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 1110.0],  # Outlier
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 19000000]  # Outlier
        }
        return pd.DataFrame(data, index=dates)
    
    def test_preprocess_good_data(self, preprocessor, sample_good_data):
        # preprocessing with some nice data
        print("Testing preprocessing with good data...")
        df, report = preprocessor.preprocess_ohlcv_data(sample_good_data, "TEST", strict_mode=True)
        pd.testing.assert_frame_equal(df, sample_good_data)
        
        assert report.total_rows == 10
        assert not report.has_errors()
        assert all(count == 0 for count in report.missing_values.values())
        assert not report.outliers_detected
        assert not report.data_type_issues
        assert report.ohlcv_violations == 0
        print("✓ Good data passed preprocessing without modifications")
    
    def test_preprocess_bad_data_strict(self, preprocessor, sample_bad_data):
        # Bad data preprocessing in strict mode
        print("Testing preprocessing with bad data in strict mode...")
        with pytest.raises(DataError) as exc_info: # should throw an error
            preprocessor.preprocess_ohlcv_data(sample_bad_data, "TEST", strict_mode=True)
        print(f"✓ Strict mode correctly raised error: {exc_info.value}")
    
    def test_preprocess_bad_data_lenient(self, preprocessor, sample_bad_data):
        # Test bad data in not strict mode
        print("Testing preprocessing with bad data in lenient mode...")
        df, report = preprocessor.preprocess_ohlcv_data(sample_bad_data, "TEST", strict_mode=False)
        assert not df.isnull().any().any()
        assert (df['High'] >= df['Low']).all()
        assert (df['Close'] > 0).all()
        assert (df['Volume'] >= 0).all()
        assert report.has_errors() or report.has_warnings()
        assert 'Open' in report.missing_values
        assert report.ohlcv_violations > 0
        print("✓ Lenient mode fixed data issues without raising errors")
        print(f"  - Fixed issues: {report.warnings}")
    
    def test_handle_missing_values(self, preprocessor, sample_bad_data):
        # Handling of any missing values
        print("Testing missing value handling...")
        
        report = DataQualityReport(
            total_rows=len(sample_bad_data),
            missing_values={},
            outliers_detected={},
            data_type_issues=[],
            ohlcv_violations=0,
            date_range_issues=[],
            warnings=[],
            errors=[]
        )
        
        df = preprocessor._handle_missing_values(sample_bad_data, "TEST", report)
        assert not df['Open'].isnull().any()
        assert 'Open' in report.missing_values
        assert report.missing_values['Open'] > 0
        print("✓ Missing values were properly handled")
        print(f"  - Missing value report: {report.missing_values}")
    
    def test_validate_ohlcv_relationships(self, preprocessor, sample_bad_data):
        # Validation of OHLCV relationships
        print("Testing OHLCV relationship validation...")
        report = DataQualityReport(
            total_rows=len(sample_bad_data),
            missing_values={},
            outliers_detected={},
            data_type_issues=[],
            ohlcv_violations=0,
            date_range_issues=[],
            warnings=[],
            errors=[]
        )
        df = preprocessor._validate_ohlcv_relationships(sample_bad_data, "TEST", report)
        assert (df['High'] >= df['Low']).all()
        assert report.ohlcv_violations > 0
        assert (df['Close'] > 0).all()
        assert (df['Volume'] >= 0).all()
        print("✓ OHLCV relationship violations were fixed")
        print(f"  - Fixed violations: {report.ohlcv_violations}")
    
    def test_handle_outliers(self, preprocessor):
        # Outlier detection and handling
        print("Testing outlier detection...")
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 1000.0],  # Last value is an outlier
            'High': [105.0, 106.0, 107.0, 108.0, 1100.0],  # Same here
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 1050.0],  # Again
            'Volume': [1000, 1100, 1200, 1300, 100000]  # And again
        }
        df = pd.DataFrame(data, index=dates)
        
        report = DataQualityReport(
            total_rows=len(df),
            missing_values={},
            outliers_detected={},
            data_type_issues=[],
            ohlcv_violations=0,
            date_range_issues=[],
            warnings=[],
            errors=[]
        )
        
        preprocessor.outlier_threshold = 1.5
        result_df = preprocessor._handle_outliers(df, "TEST", report)
        assert len(report.outliers_detected) > 0
        assert any("outlier" in warning.lower() for warning in report.warnings)
        print("✓ Outliers were properly detected and handled")
        print(f"  - Detected outliers: {report.outliers_detected}")
        print(f"  - Warnings: {report.warnings}")
    
    def test_validate_data_types(self, preprocessor):
        #Data type checks
        print("Testing data type validation...")
        
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = {
            'Open': ['100.0', '101.0', '102.0', '103.0', '104.0'],  # Strings instead of floats
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000.5, 1100.5, 1200.5, 1300.5, 1400.5]  # Floats instead of ints
        }
        df = pd.DataFrame(data, index=dates)
        
        report = DataQualityReport(
            total_rows=len(df),
            missing_values={},
            outliers_detected={},
            data_type_issues=[],
            ohlcv_violations=0,
            date_range_issues=[],
            warnings=[],
            errors=[]
        )
        df = preprocessor._validate_data_types(df, "TEST", report)
        assert df['Open'].dtype == 'float64'
        assert df['Volume'].dtype == 'int64'
        print("✓ Data types were properly validated and converted")
        print(f"  - Type conversion warnings: {report.warnings}")
    
    def test_validate_basic_structure(self, preprocessor):
        # Basic DataFrame structure
        print("Testing basic structure validation...")
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        df = pd.DataFrame(data, index=['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        
        report = DataQualityReport(
            total_rows=len(df),
            missing_values={},
            outliers_detected={},
            data_type_issues=[],
            ohlcv_violations=0,
            date_range_issues=[],
            warnings=[],
            errors=[]
        )

        df = preprocessor._validate_basic_structure(df, "TEST", report)
        assert isinstance(df.index, pd.DatetimeIndex)
        print("✓ Basic structure was properly validated")
        print(f"  - Structure validation warnings: {report.warnings}")
    
    def test_empty_dataframe(self, preprocessor):
        # Empty DataFrame
        print("Testing empty DataFrame handling...")
        df = pd.DataFrame()
        df_result, report = preprocessor.preprocess_ohlcv_data(df, "TEST", strict_mode=False)
        assert report.has_errors()
        assert "empty" in report.errors[0].lower()
        print("✓ Empty DataFrame was properly detected")
        print(f"  - Error message: {report.errors}")
    
    def test_missing_columns(self, preprocessor):
        # Handling of DataFrames with columns that are missing
        print("Testing missing columns handling...")
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            # 'Low' : [blah, blah] - low disappeared
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        df = pd.DataFrame(data, index=dates)
        df_result, report = preprocessor.preprocess_ohlcv_data(df, "TEST", strict_mode=False)
        assert report.has_errors()
        assert "missing" in report.errors[0].lower()
        assert "low" in report.errors[0].lower()
        print("✓ Missing columns were properly detected")
        print(f"  - Error message: {report.errors}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])