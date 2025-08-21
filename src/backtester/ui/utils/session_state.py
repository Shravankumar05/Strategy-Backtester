"""
Session state management utilities for Streamlit.

This module provides functions to manage Streamlit session state
in a consistent and type-safe manner.
"""

import streamlit as st
from typing import Any, Optional, Dict
from datetime import date


def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        # Data settings
        "ticker": "AAPL",
        "start_date": date(2020, 1, 1),
        "end_date": date(2023, 12, 31),
        
        # Capital settings
        "initial_capital": 10000.0,
        "leverage": 1.0,
        
        # Strategy settings
        "strategy_type": "Moving Average Crossover",
        "strategy_index": 0,
        "short_window": 20,
        "long_window": 50,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "bb_period": 20,
        "bb_std_multiplier": 2.0,
        "bb_buy_threshold": 0.0,
        "bb_sell_threshold": 0.0,
        "stoch_k_period": 14,
        "stoch_d_period": 3,
        "stoch_oversold": 20.0,
        "stoch_overbought": 80.0,
        "custom_strategy_name": "My Custom Strategy",
        "custom_strategy_description": "Custom trading strategy",
        "custom_strategy_json": '{"name": "My Custom Strategy", "description": "Custom trading strategy", "indicators": {"rsi_14": {"type": "rsi", "window": 14, "source": "Close"}}, "rules": [{"conditions": [{"indicator": "rsi_14", "operator": "<", "value": 30}], "action": "buy"}, {"conditions": [{"indicator": "rsi_14", "operator": ">", "value": 70}], "action": "sell"}]}',
        
        # Execution settings
        "transaction_cost": 0.001,  # 0.1%
        "slippage": 0.0005,  # 0.05%
        "position_sizing": "Fixed Fraction",
        "position_sizing_index": 0,
        "position_size": 0.1,  # 10%
        "position_size_pct": 10,
        "position_size_fixed": 1000.0,
        
        # Results
        "backtest_results": None,
        "backtest_running": False,
        
        # UI state
        "current_tab": 0,
        "show_advanced": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_state(key: str, default: Any = None) -> Any:
    """
    Get a value from session state with optional default.
    
    Args:
        key: The session state key
        default: Default value if key doesn't exist
        
    Returns:
        The session state value or default
    """
    return st.session_state.get(key, default)


def set_session_state(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: The session state key
        value: The value to set
    """
    st.session_state[key] = value


def clear_session_state(keys: Optional[list] = None) -> None:
    """
    Clear session state keys.
    
    Args:
        keys: List of keys to clear. If None, clears all.
    """
    if keys is None:
        st.session_state.clear()
        initialize_session_state()
    else:
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]


def get_backtest_config() -> Dict[str, Any]:
    """
    Get current backtest configuration from session state.
    
    Returns:
        Dictionary containing all backtest configuration parameters
    """
    config = {
        # Data configuration
        "ticker": get_session_state("ticker"),
        "start_date": get_session_state("start_date"),
        "end_date": get_session_state("end_date"),
        
        # Capital configuration
        "initial_capital": get_session_state("initial_capital"),
        "leverage": get_session_state("leverage"),
        
        # Strategy configuration
        "strategy_type": get_session_state("strategy_type"),
        "strategy_params": _get_strategy_params(),
        
        # Execution configuration
        "transaction_cost": get_session_state("transaction_cost"),
        "slippage": get_session_state("slippage"),
        "position_sizing": get_session_state("position_sizing"),
        "position_size": get_session_state("position_size"),
    }
    
    return config


def _get_strategy_params() -> Dict[str, Any]:
    """Get strategy-specific parameters based on selected strategy."""
    strategy_type = get_session_state("strategy_type")
    
    if strategy_type == "Moving Average Crossover":
        return {
            "short_window": get_session_state("short_window"),
            "long_window": get_session_state("long_window"),
        }
    elif strategy_type == "RSI Strategy":
        return {
            "period": get_session_state("rsi_period"),
            "overbought": get_session_state("rsi_overbought"),
            "oversold": get_session_state("rsi_oversold"),
        }
    elif strategy_type == "Bollinger Bands":
        return {
            "period": get_session_state("bb_period"),
            "std_multiplier": get_session_state("bb_std_multiplier"),
            "buy_threshold": get_session_state("bb_buy_threshold"),
            "sell_threshold": get_session_state("bb_sell_threshold"),
        }
    elif strategy_type == "Stochastic Oscillator":
        return {
            "k_period": get_session_state("stoch_k_period"),
            "d_period": get_session_state("stoch_d_period"),
            "oversold_level": get_session_state("stoch_oversold"),
            "overbought_level": get_session_state("stoch_overbought"),
        }
    elif strategy_type == "Custom Strategy":
        try:
            import json
            strategy_json = get_session_state("custom_strategy_json", "")
            
            if not strategy_json.strip():
                # Return default parameters if no JSON is configured
                return {
                    "name": get_session_state("custom_strategy_name", "My Custom Strategy"),
                    "description": get_session_state("custom_strategy_description", "Custom strategy"),
                    "rules": [],
                    "indicators": {}
                }
            
            strategy_config = json.loads(strategy_json)
            return {
                "name": strategy_config.get("name", get_session_state("custom_strategy_name", "My Custom Strategy")),
                "description": strategy_config.get("description", get_session_state("custom_strategy_description", "Custom strategy")),
                "rules": strategy_config.get("rules", []),
                "indicators": strategy_config.get("indicators", {})
            }
        except (json.JSONDecodeError, Exception):
            return {
                "name": get_session_state("custom_strategy_name", "My Custom Strategy"),
                "description": get_session_state("custom_strategy_description", "Custom strategy"),
                "rules": [],
                "indicators": {}
            }
    else:  # Buy and Hold
        return {}


def is_config_valid() -> tuple[bool, str]:
    """
    Validate current configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_backtest_config()
    
    # Check capital first (most fundamental)
    if config["initial_capital"] <= 0:
        return False, "Initial capital must be positive"
    
    # Check date range
    if config["start_date"] >= config["end_date"]:
        return False, "Start date must be before end date"
    
    # Check strategy parameters
    if config["strategy_type"] == "Moving Average Crossover":
        if config["strategy_params"]["short_window"] >= config["strategy_params"]["long_window"]:
            return False, "Short MA period must be less than Long MA period"
    
    # Check position size
    if config["position_sizing"] == "Fixed Fraction":
        if config["position_size"] <= 0 or config["position_size"] > 1:
            return False, "Position size fraction must be between 0 and 1"
    else:  # Fixed Size
        if config["position_size"] <= 0 or config["position_size"] > config["initial_capital"]:
            return False, "Position size must be positive and less than initial capital"
    
    return True, ""


def update_backtest_status(status: str, progress: Optional[float] = None, message: Optional[str] = None):
    """
    Update backtest execution status.
    
    Args:
        status: Status string ('running', 'completed', 'failed')
        progress: Progress percentage (0-100)
        message: Status message
    """
    set_session_state("backtest_status", status)
    if progress is not None:
        set_session_state("backtest_progress", progress)
    if message is not None:
        set_session_state("backtest_message", message)