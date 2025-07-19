import pandas as pd
import numpy as np
import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from .fetcher import DataError

@dataclass
class DataQualityReport:
    total_rows: int
    missing_values: Dict[str, int]
    outliers_detected: Dict[str, int]
    data_type_issues: List[str]
    ohlcv_violations: int
    date_range_issues: List[str]
    warnings: List[str]
    errors: List[str]

    def has_error(self) -> bool:
        x = len(self.errors) > 0
        return x
    
    def has_warnings(Self) -> bool:
        y = len(self.warnings) > 0
        return y
    

class DataPreprocessor:
    def __init__(self, outlier_threshold: float = 3.0, max_missing_ratio: float=0.1, enable_logging: bool=True):
        self.outlier_threshold = outlier_threshold
        self.max_missing_ratio = max_missing_ratio
        self.enable_logging = enable_logging

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
        
    
    def preprocess_ohlcv_data(self, df: pd.DataFrame, symbol: str, strict_mode: bool=True) -> Tuple[pd.DataFrame, DataQualityReport]:
        self.logger.info(f"started preprocessing the {symbol}")
        report = DataQualityReport(total_rows=len(df), missing_values={}, outliers_detected={}, data_type_issues=[], ohlcv_violations=0, date_range_issues=[], warnings=[], errors=[])

        try:
            df_cleaned = self._validate_basic_structure(df, symbol, report) # validate structure
            df_cleaned = self._handle_missing_values(df_cleaned, symbol, report) # handle missing vals
            df_cleaned = self._validate_data_types(df_cleaned, symbol, report) # check/fix data types
            df_cleaned = self._handle_outliers(df_cleaned, symbol, report) # handle outliers
            df_cleaned = self._validate_ohlcv_relationships(df_cleaned, symbol, report) # check ohclv relations
            self._final_validation(df_cleaned, symbol, report)

            if strict_mode and report.has_errors():
                error_msg = f"data quality issue - {symbol}: {'; '.join(report.errors)}"
                raise DataError(error_msg)
            
            self.logger.info(f"Preprocessing done: {symbol}"
                        f"warnings: {len(report.warnings)}, Errors: {len(report.errors)}")
            
            return df_cleaned, report

        except Exception as e:
            report.errors.append(f"Failed preprocessing:{str(e)}")
            if strict_mode:
                raise DataError(f"Failed preprocess for {symbol}")
            return df, report
        
    def _validate_basic_structure(self, df: pd.DataFrame, symbol: str, report: DataQualityReport) -> pd.DataFrame:
        if df.empty:
            report.errors.append("Empty data frame")
            return df
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            report.errors.append(f"Missing required columns of: {missing_columns}")
            return df
        
        if not isinstance(df.index, pd.DataFrame):
            try:
                df.index = pd.to_datetime(df.index)
                report.warnings.append(f"Converted index to a DataTimeIndex")
                
            except Exception as e:
                report.errors.append(f"Could not conver index to a DateTimeIndex")
        
        df = df.sort_index()
        return df
    
    def _handle_missing_values(self, df:pd.DataFrame, symbol: str, report: DataQualityReport) -> pd.DataFrame:
        missing_counts = df.isnull().sum()
        report.missing_values = missing_counts.to_dict()
        total_rows = len(df)

        for column, missing_count in missing_counts.items():
            if missing_count > 0:
                missing_ratio = missing_count / total_rows
                if missing_ratio > self.max_missing_ratio:
                    report.errors.append(f"Column {column} has the {missing_ratio:.2%} missing values"
                                        f"This is exceeds threshold of {self.max_missing_ratio:.2%}")
                else:
                    report.warnings.append(f"Column {column} has {missing_count} missing values")
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        df[price_columns] = df[price_columns].ffill()

        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)

        if rows_after != rows_before:
            report.warnings.append(f"Dropped {rows_before - rows_after} number of rows which had missing vlaues")
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame, symbol: str, report: DataQualityReport) -> pd.DataFrame:
        expected_types = {
            'Open': 'float64',
            'High': 'float64',
            'Low' : 'float64',
            'Close' : 'float64',
            'Volume' : 'int64'
        }

        for column, expected_type in expected_types.items():
            if column in df.columns:
                current_type = str(df[column].dtype)
                if current_type != expected_type:
                    try:
                        if expected_type == 'int64':
                            df[column] = df[column].astype('float64').astype('int64')
                        else:
                            df[column] = df[column].astype(expected_type)
                        report.warnings.append(f"Converted {column} to data type {expected_type}")
                    
                    except Exception as e:
                        report.data_type_issues(f"Could not convert {column} to {expected_type} kept as {current_type}")
        
        return df
    
    def _handle_outliers(self, df: pf.DataFrame, symbol: str, report: DataQualityReport) -> pd.DataFrame:
        price_columns = ['Open', 'High', 'Low', 'Close']
        for column in price_columns + ['Volume']:
            if column not in df.columns:
                continue
            
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers_mask = z_scores > self.outlier_threshold
            outlier_count = outliers_mask.sum()

            if outlier_count > 0:
                report.outliers_detected[column] = outlier_count
                report.warnings.append(f"Detected {outlier_count} outliers in {column}"
                            f"(Z-score > {self.outlier_threshold})")
                
                if column in price_columns:
                    mean_val = df[column].mean()
                    std_val = df[column].std()
                    upper_bound = mean_val + (std_val * 3)
                    lower_bound = mean_val - (std_val * 3)
                    df.loc[df[column] > upper_bound, column] = upper_bound
                    df.loc[df[column] < lower_bound, column] = lower_bound
                
                elif column == 'Volume':
                    report.warnings.append(f"Volume outliers present but not changed - this could cause issuses")

        return df
    
    def _validate_ohlcv_relationships(self, df: pd.DataFrame, symbol: str, report: DataQualityReport) -> pd.DataFrame:
        violations = 0
        high_violations = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
        low_violations = (df['Low'] > df['Open', 'Close'].min(axis=1)).sum()

        if high_violations > 0:
            violations += high_violations
            mask = df['High'] < df[['Open', 'Close']].max(axis=1)
            df.loc[mask, 'High'] = df.loc[mas, ['Open', 'Close']].max(axis=1)
            report.warnings.append(f"Fixed {high_violations} violations where high wasn't the actual high")
        
        if low_violations > 0:
            violations += low_violations
            mask = df['Low'] > df[['Open', 'Close']].min(axis=1)
            df.loc[mask, 'Low'] = df.loc[mas, ['Open', 'Close']].min(axis=1)
            report.warnings.append(f"Fixed {low_violations} violations where low wasn't the actual low")
        
        negative_volume = (df['Volume' < 0]).sum()
        if negative_volume > 0:
            violations += negative_volume
            df.loc[df['Volume'] < 0, 'Volume'] = 0
            report.warnings.append(f"There were negative volumes that have been sorted, but this is an issue")
        
        for col in ['Open', 'High', 'Low', 'Close']:
            negative_prices = (df[col] <= 0).sum()
            if negative_prices > 0:
                violations += negative_prices
                df = df[df[col] > 0]
                report.warnings.append(f"Removed {negative_prices} rows which had negative prices in it")
        
        report.ohlcv_violations = violations
        return df
    
    def _final_validation(self, df:pd.DataFrame, symbol: str, report: DataQualityReport) -> None:
        if df.empty:
            report.errors.append("DataFrame is empty after preprocessing")
            return
        
        nan_counts = df.isnull().sum()
        
        if nan_counts.sum()>0:
            report.errors.append(f"NaN values are still present affter cleaning: {nan_counts.to_dict()}")
        
        if not df.index.is_monotonic_increasing:
            report.warnings.append("Date index is not sorted, fixing that.")
            df.sort_index(inplace=True)
        
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            report.warnings.append(f"There are {duplicate_dates} duplicate dates presents")
            df = df[~df.index.duplicated(keep='first')]
        
        report.warnings.append(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")


def create_great_expectations_suite(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    suite = {
        "expectation_suite_name": f"ohlcv_suite_{symbol}",
        "expectations": [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 10000}
            },
            {
                "expectation_type": "expect_table_columns_to_match_ordered_list",
                "kwargs": {"column_list": ["Open", "High", "Low", "Close", "Volume"]}
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "Open", "type_": "float64"}
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "High", "type_": "float64"}
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "Low", "type_": "float64"}
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "Close", "type_": "float64"}
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "Volume", "type_": "int64"}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "Open", "min_value": 0.01, "max_value": 10000}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "High", "min_value": 0.01, "max_value": 10000}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "Low", "min_value": 0.01, "max_value": 10000}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "Close", "min_value": 0.01, "max_value": 10000}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "Volume", "min_value": 0, "max_value": 1000000000}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "Open"}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "High"}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "Low"}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "Close"}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "Volume"}
            }
        ]
    }
    return suite