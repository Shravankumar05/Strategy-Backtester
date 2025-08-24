import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from typing import Dict, Any, Optional
import traceback

try:
    from backtester.metrics.performance import PerformanceMetrics
    from backtester.visualization.visualizer import Visualizer
    from backtester.data.yfinance_fetcher import YFinanceDataFetcher
    from backtester.data.cache_manager import CacheManager
    from backtester.strategy.ma_crossover import MovingAverageCrossoverStrategy
    from backtester.strategy.rsi_strategy import RSIStrategy
    from backtester.strategy.bollinger_bands import BollingerBandsStrategy
    from backtester.strategy.stochastic_oscillator import StochasticOscillatorStrategy
    from backtester.strategy.custom_strategy import CustomStrategy
    from backtester.simulation.engine import SimulationEngine
    from backtester.simulation.config import SimulationConfig
    from backtester.ui.components.progress_indicators import ProgressManager
    from backtester.ui.utils.session_state import initialize_session_state, get_session_state, set_session_state
    from backtester.ui.components.alpha_news_ui import render_alpha_news_tab
    from backtester.ui.components.enhanced_recommendation_ui import render_enhanced_recommendation_tab
    from backtester.recommendation.enhanced_recommendation_engine import EnhancedRecommendationEngine
    from backtester.ui.components.ui_components import section_header, card, apply_global_styles
    
    # Apply global styles
    apply_global_styles()
    
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all required modules are properly installed and the application is run from the correct directory.")
    st.stop()

def configure_page():
    """Initialize page configuration (moved to main.py)"""
    # Page config is now handled in main.py to ensure it's the first Streamlit command
    pass

# Global styles are already applied in the try block above

# Responsive layout styles have been integrated into main ui_components

# Apply custom styles
# Note: All styles are now in apply_global_styles() to avoid duplication

def render_header():
    """Render the main application header with clean styling"""
    st.markdown("""
    <div style="
        background-color: #f8f9fa;
        color: #1e3c72;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        text-align: left;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    ">
        <h1 style="
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            line-height: 1.2;
            color: #1e3c72;
        ">Trading Strategy Backtester</h1>
        <p style="
            margin: 0.75rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.8;
            font-weight: 400;
            max-width: 800px;
            line-height: 1.5;
            color: #2a5298;
        ">Advanced AI-powered trading strategy analysis and optimization platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display status message if any
    status_message = get_session_state("status_message", "")
    status_type = get_session_state("status_type", "info")
    
    if status_message:
        if status_type == "error":
            st.error(status_message, icon=None)
        elif status_type == "warning":
            st.warning(status_message, icon=None)
        elif status_type == "success":
            st.success(status_message, icon=None)
        else:
            st.info(status_message, icon=None)

def render_navigation():
    """Render the main navigation tabs with consistent styling"""
    # Define tab labels without emojis
    tab_labels = [
        "Strategy Builder",
        "Backtest Results", 
        "Performance Charts",
        "Trade Analysis",
        "Market News",
        "ML Strategy Advisor",
        "About"
    ]
    
    # Create tabs with full-width clean styling
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        justify-content: space-between;
        gap: 0;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        text-align: center;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        background: transparent;
        color: #1e3c72;
        font-weight: 600;
        transition: background-color 0.2s ease;
        border: none;
        margin: 0 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #1e3c72;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72 !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(30, 60, 114, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(tab_labels)
    return tabs

def render_configuration_tab():
    section_header("Backtest Configuration", level=1, description="Configure your backtest parameters and strategy settings")
    
    with st.container():
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            # Data Settings Card
            with st.container():
                section_header("Data Settings", level=2, divider=False)
                render_data_settings()
            
            # Capital Settings Card
            with st.container():
                section_header("Capital Settings", level=2, divider=False)
                render_capital_settings()
        
        with col2:
            # Strategy Settings Card
            with st.container():
                section_header("Strategy Settings", level=2, divider=False)
                render_strategy_settings()
            
            # Execution Settings Card
            with st.container():
                render_execution_settings()
        
        # Run Backtest Button
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Run Backtest", type="primary", use_container_width=True):
                run_backtest()

def render_data_settings():
    with st.container():
        # Create a card container for the data settings
        with st.container():
            ticker = st.text_input(
                "Stock Ticker",
                value=get_session_state("ticker", "AAPL"),
                help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
                key="ticker_input"
            )
            set_session_state("ticker", ticker)
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=get_session_state("start_date", date(2020, 1, 1)),
                    min_value=date(2000, 1, 1),
                    max_value=date.today(),
                    help="Select the start date for backtesting",
                    key="start_date_input"
                )
                set_session_state("start_date", start_date)
                
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=get_session_state("end_date", date(2023, 12, 31)),
                    min_value=date(2000, 1, 1),
                    max_value=date.today(),
                    help="Select the end date for backtesting",
                    key="end_date_input"
                )
                set_session_state("end_date", end_date)
                
            if start_date >= end_date:
                st.error("Start date must be before end date")

def render_capital_settings():
    with st.container():
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=get_session_state("initial_capital", 10000.0),
            step=1000.0,
            help="Starting capital for the backtest",
            key="initial_capital_input"
        )
        set_session_state("initial_capital", initial_capital)
        
        leverage = st.slider(
            "Leverage",
            min_value=1.0,
            max_value=5.0,
            value=get_session_state("leverage", 1.0),
            step=0.1,
            help="Maximum leverage to use (1.0 = no leverage)",
            key="leverage_slider"
        )
        set_session_state("leverage", leverage)

def render_strategy_settings():
    with st.container():
        # Strategy Type Selection
        strategy_type = st.selectbox(
            "Strategy Type",
            options=["Moving Average Crossover", "RSI Strategy", "Bollinger Bands", "Stochastic Oscillator", "Custom Strategy", "Buy and Hold"],
            index=get_session_state("strategy_index", 0),
            help="Select the trading strategy to backtest",
            key="strategy_type_select"
        )
        set_session_state("strategy_type", strategy_type)
        set_session_state("strategy_index", ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands", "Stochastic Oscillator", "Custom Strategy", "Buy and Hold"].index(strategy_type))
        
        # Strategy Parameters
        if strategy_type == "Moving Average Crossover":
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    short_window = st.number_input(
                        "Short MA Period",
                        min_value=5,
                        max_value=50,
                        value=get_session_state("short_window", 20),
                        help="Short moving average period",
                        key="short_ma_input"
                    )
                    set_session_state("short_window", short_window)
                
                with col2:
                    long_window = st.number_input(
                        "Long MA Period",
                        min_value=20,
                        max_value=200,
                        value=get_session_state("long_window", 50),
                        help="Long moving average period",
                        key="long_ma_input"
                    )
                    set_session_state("long_window", long_window)
                
                if short_window >= long_window:
                    st.error("Short MA period must be less than Long MA period")
        
        elif strategy_type == "RSI Strategy":
            with st.container():
                section_header("RSI Parameters", level=3, divider=False)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rsi_period = st.number_input(
                        "RSI Period",
                        min_value=5,
                        max_value=30,
                        value=get_session_state("rsi_period", 14),
                        help="RSI calculation period",
                        key="rsi_period_input"
                    )
                    set_session_state("rsi_period", rsi_period)
                
                with col2:
                    rsi_overbought = st.number_input(
                        "Overbought Level",
                        min_value=60,
                        max_value=90,
                        value=get_session_state("rsi_overbought", 70),
                        help="RSI overbought threshold",
                        key="rsi_overbought_input"
                    )
                    set_session_state("rsi_overbought", rsi_overbought)
                
                with col3:
                    rsi_oversold = st.number_input(
                        "Oversold Level",
                        min_value=10,
                        max_value=40,
                        value=get_session_state("rsi_oversold", 30),
                        help="RSI oversold threshold",
                        key="rsi_oversold_input"
                    )
                    set_session_state("rsi_oversold", rsi_oversold)
                
                if rsi_oversold >= rsi_overbought:
                    st.error("Oversold level must be less than Overbought level")
        
        elif strategy_type == "Bollinger Bands":
            with st.container():
                section_header("Bollinger Bands Parameters", level=3, divider=False)
                col1, col2 = st.columns(2)
                
                with col1:
                    bb_period = st.number_input(
                        "Period",
                        min_value=5,
                        max_value=100,
                        value=get_session_state("bb_period", 20),
                        help="Period for moving average calculation",
                        key="bb_period_input"
                    )
                    set_session_state("bb_period", bb_period)
                    
                    bb_buy_threshold = st.number_input(
                        "Buy Threshold",
                        min_value=0.0,
                        max_value=0.3,
                        value=get_session_state("bb_buy_threshold", 0.0),
                        step=0.01,
                        format="%.2f",
                        help="Distance from lower band to trigger buy (0 = touch band)",
                        key="bb_buy_threshold_input"
                    )
                    set_session_state("bb_buy_threshold", bb_buy_threshold)
                
                with col2:
                    bb_std_multiplier = st.number_input(
                        "Standard Deviation Multiplier",
                        min_value=0.5,
                        max_value=4.0,
                        value=get_session_state("bb_std_multiplier", 2.0),
                        step=0.1,
                        format="%.1f",
                        help="Standard deviation multiplier for band width",
                        key="bb_std_multiplier_input"
                    )
                    set_session_state("bb_std_multiplier", bb_std_multiplier)
                    
                    bb_sell_threshold = st.number_input(
                        "Sell Threshold", 
                        min_value=0.0,
                        max_value=0.3,
                        value=get_session_state("bb_sell_threshold", 0.0),
                        step=0.01,
                        format="%.2f",
                        help="Distance from upper band to trigger sell (0 = touch band)",
                        key="bb_sell_threshold_input"
                    )
                    set_session_state("bb_sell_threshold", bb_sell_threshold)
        
        elif strategy_type == "Stochastic Oscillator":
            with st.container():
                section_header("Stochastic Oscillator Parameters", level=3, divider=False)
                col1, col2 = st.columns(2)
                
                with col1:
                    stoch_k_period = st.number_input(
                        "%K Period",
                        min_value=5,
                        max_value=50,
                        value=get_session_state("stoch_k_period", 14),
                        help="Period for %K calculation (fast stochastic)",
                        key="stoch_k_period_input"
                    )
                    set_session_state("stoch_k_period", stoch_k_period)
                    
                    stoch_oversold = st.number_input(
                        "Oversold Level",
                        min_value=5.0,
                        max_value=40.0,
                        value=get_session_state("stoch_oversold", 20.0),
                        step=1.0,
                        format="%.1f",
                        help="Level below which asset is considered oversold",
                        key="stoch_oversold_input"
                    )
                    set_session_state("stoch_oversold", stoch_oversold)
                
                with col2:
                    stoch_d_period = st.number_input(
                        "%D Period",
                        min_value=1,
                        max_value=10,
                        value=get_session_state("stoch_d_period", 3),
                        help="Period for %D smoothing (slow stochastic)",
                        key="stoch_d_period_input"
                    )
                    set_session_state("stoch_d_period", stoch_d_period)
                    
                    stoch_overbought = st.number_input(
                        "Overbought Level",
                        min_value=60.0,
                        max_value=95.0,
                        value=get_session_state("stoch_overbought", 80.0),
                        step=1.0,
                        format="%.1f",
                        help="Level above which asset is considered overbought",
                        key="stoch_overbought_input"
                    )
                    set_session_state("stoch_overbought", stoch_overbought)
                
                if stoch_oversold >= stoch_overbought:
                    st.error("Oversold level must be less than overbought level")
        
        elif strategy_type == "Custom Strategy":
            render_custom_strategy_ui()
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_custom_strategy_ui():
    """Render the custom strategy configuration UI"""
    with st.container():
        section_header("Custom Strategy Builder", level=3, description="Create your own trading strategy using technical indicators and rules")
        
        # Strategy Name and Description
        col1, col2 = st.columns(2)
        with col1:
            strategy_name = st.text_input(
                "Strategy Name",
                value=get_session_state("custom_strategy_name", "My Custom Strategy"),
                help="Give your strategy a descriptive name",
                key="strategy_name_input"
            )
            set_session_state("custom_strategy_name", strategy_name)
        
        with col2:
            strategy_description = st.text_area(
                "Description",
                value=get_session_state("custom_strategy_description", "Custom trading strategy"),
                height=100,
                help="Describe what your strategy does",
                key="strategy_description_input"
            )
            set_session_state("custom_strategy_description", strategy_description)
        
        # Strategy Configuration
        section_header("Strategy Configuration", level=4, divider=False)
        
        # Get current strategy configuration
        default_strategy = {
            "name": strategy_name,
            "description": strategy_description,
            "indicators": {
                "rsi_14": {
                    "type": "rsi",
                    "window": 14,
                    "source": "Close"
                },
                "sma_20": {
                    "type": "sma",
                    "window": 20,
                    "source": "Close"
                }
            },
            "rules": [
                {
                    "conditions": [
                        {
                            "indicator": "rsi_14",
                            "operator": "<",
                            "value": 30
                        }
                    ],
                    "action": "buy"
                },
                {
                    "conditions": [
                        {
                            "indicator": "rsi_14",
                            "operator": ">",
                            "value": 70
                        }
                    ],
                    "action": "sell"
                }
            ]
        }
        
        current_json = get_session_state("custom_strategy_json", "")
        
        # If no JSON is stored, use the default strategy
        if not current_json.strip():
            current_json = json.dumps(default_strategy, indent=2)
            set_session_state("custom_strategy_json", current_json)
        
        # JSON Editor
        strategy_json = st.text_area(
            "Strategy Configuration (JSON)",
            value=current_json,
            height=400,
            help="Define your custom strategy using JSON format. See documentation below for examples.",
            key="strategy_json_editor"
        )
        
        # Validate and save JSON
        try:
            strategy_config = json.loads(strategy_json)
            if validate_custom_strategy_config(strategy_config):
                set_session_state("custom_strategy_json", strategy_json)
                st.success("Strategy configuration is valid!")
            else:
                st.error("Invalid strategy configuration. Please check the format.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
        
        # Documentation
        with st.expander("Strategy Configuration Guide", expanded=False):
            st.markdown("""
            **Strategy JSON Format:**
            ```json
            {
              "name": "Strategy Name",
              "description": "Strategy description",
              "indicators": {
                "indicator_name": {
                  "type": "sma|ema|rsi|macd|bollinger|atr",
                  "window": 14,
                  "source": "Close"
                }
              },
              "rules": [
                {
                  "conditions": [
                    {
                      "indicator": "rsi_14",
                      "operator": "<|>|<=|>=|==|!=",
                      "value": 30
                    }
                  ],
                  "action": "buy|sell|hold"
                }
              ]
            }
            ```
            
            **Available Indicators:**
            - `sma`: Simple Moving Average (params: window, source)
            - `ema`: Exponential Moving Average (params: window, source)
            - `rsi`: Relative Strength Index (params: window, source)
            - `macd`: MACD (params: fast_period, slow_period, signal_period, source)
            - `bollinger`: Bollinger Bands (params: window, num_std, source)
            - `atr`: Average True Range (params: window)
            
            **MACD Output Columns:**
            - `{name}_line`: MACD line
            - `{name}_signal`: Signal line
            - `{name}_hist`: Histogram
            
            **Bollinger Bands Output Columns:**
            - `{name}_upper`: Upper band
            - `{name}_middle`: Middle band (SMA)
            - `{name}_lower`: Lower band
            
            **Basic Price Data:**
            - `Open`, `High`, `Low`, `Close`, `Volume`
            
            **Example Strategies:**
            
            **RSI Mean Reversion:**
            ```json
            {
              "indicators": {
                "rsi_14": {"type": "rsi", "window": 14, "source": "Close"}
              },
              "rules": [
                {
                  "conditions": [{"indicator": "rsi_14", "operator": "<", "value": 30}],
                  "action": "buy"
                },
                {
                  "conditions": [{"indicator": "rsi_14", "operator": ">", "value": 70}],
                  "action": "sell"
                }
              ]
            }
            ```
            
            **Moving Average Crossover:**
            ```json
        }
        ```
        
        *Moving Average Crossover:*
        ```json
        {
          "indicators": {
            "sma_short": {"type": "sma", "window": 20, "source": "Close"},
            "sma_long": {"type": "sma", "window": 50, "source": "Close"}
          },
          "rules": [
            {
              "conditions": [{"indicator": "sma_short", "operator": ">", "value": "sma_long"}],
              "action": "buy"
            },
            {
              "conditions": [{"indicator": "sma_short", "operator": "<", "value": "sma_long"}],
              "action": "sell"
            }
          ]
        }
        ```
        """)

def validate_custom_strategy_config(config):
    """Validate custom strategy configuration"""
    try:
        # Check required fields
        required_fields = ["rules"]
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate rules
        if not isinstance(config["rules"], list) or not config["rules"]:
            return False
        
        for rule in config["rules"]:
            if not isinstance(rule, dict):
                return False
            
            if "conditions" not in rule or "action" not in rule:
                return False
            
            if rule["action"] not in ["buy", "sell", "hold"]:
                return False
            
            if not isinstance(rule["conditions"], list) or not rule["conditions"]:
                return False
            
            for condition in rule["conditions"]:
                if not isinstance(condition, dict):
                    return False
                
                required_cond_fields = ["indicator", "operator", "value"]
                for field in required_cond_fields:
                    if field not in condition:
                        return False
                
                if condition["operator"] not in [">", "<", ">=", "<=", "==", "!="]:
                    return False
        
        # Validate indicators if present
        if "indicators" in config:
            if not isinstance(config["indicators"], dict):
                return False
            
            for name, indicator in config["indicators"].items():
                if "type" not in indicator:
                    return False
                
                if indicator["type"] not in ["sma", "ema", "rsi", "macd", "bollinger", "atr"]:
                    return False
        
        return True
    
    except Exception:
        return False

def render_execution_settings():
    section_header("Execution Settings", level=2, divider=False)
    with st.container():
        # Trading Costs
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                transaction_cost = st.number_input(
                    "Transaction Cost (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=get_session_state("transaction_cost", 0.1),
                    step=0.01,
                    format="%.3f",
                    help="Transaction cost as percentage of trade value",
                    key="transaction_cost_input"
                ) / 100
                set_session_state("transaction_cost", transaction_cost)
            
            with col2:
                slippage = st.number_input(
                    "Slippage (%)",
                    min_value=0.0,
                    max_value=0.5,
                    value=get_session_state("slippage", 0.05),
                    step=0.01,
                    format="%.3f",
                    help="Price slippage as percentage of trade value",
                    key="slippage_input"
                ) / 100
                set_session_state("slippage", slippage)
        
        # Position Sizing
        with st.container():
            section_header("Position Sizing", level=4, divider=False)
            
            position_sizing = st.selectbox(
                "Position Sizing Method",
                options=["Fixed Fraction", "Fixed Size"],
                index=get_session_state("position_sizing_index", 0),
                help="Method for determining position sizes",
                key="position_sizing_select"
            )
            set_session_state("position_sizing", position_sizing)
            set_session_state("position_sizing_index", ["Fixed Fraction", "Fixed Size"].index(position_sizing))
            
            if position_sizing == "Fixed Fraction":
                position_size = st.slider(
                    "Position Size (% of Capital)",
                    min_value=1,
                    max_value=100,
                    value=get_session_state("position_size_pct", 10),
                    help="Percentage of capital to use per trade",
                    key="position_size_pct_slider"
                ) / 100
                set_session_state("position_size", position_size)
                set_session_state("position_size_pct", int(position_size * 100))
            else:
                initial_capital = get_session_state("initial_capital", 10000.0)
                position_size = st.number_input(
                    "Fixed Position Size ($)",
                    min_value=100.0,
                    max_value=float(initial_capital),
                    value=get_session_state("position_size_fixed", min(1000.0, initial_capital)),
                    step=100.0,
                    help="Fixed dollar amount per trade",
                    key="position_size_fixed_input"
                )
                set_session_state("position_size", position_size)
                set_session_state("position_size_fixed", position_size)

def run_backtest():
    try:
        # Define progress steps
        steps = [
            "Validating configuration",
            "Fetching historical data", 
            "Initializing strategy",
            "Generating trading signals",
            "Running simulation",
            "Calculating performance metrics",
            "Generating visualizations",
            "Finalizing results"
        ]
        
        # Initialize progress context
        with ProgressManager.progress_context(
            title="Running Backtest",
            description="Please wait while we execute your backtest...",
            total_steps=len(steps),
            show_spinner=True,
            show_progress_bar=True
        ) as update_progress:
            # Step 1: Validate configuration
            update_progress(1, "Validating configuration")
            config_valid, config_error = validate_backtest_config()
            if not config_valid:
                raise ValueError(f"Configuration error: {config_error}")
            
            # Step 2: Fetch historical data
            update_progress(2, "Fetching historical data")
            ticker = get_session_state("ticker", "AAPL")
            start_date = get_session_state("start_date", date(2024, 1, 1))
            end_date = get_session_state("end_date", date(2024, 6, 30))
            cache_manager = CacheManager()
            data_fetcher = YFinanceDataFetcher(cache_manager)
            
            try:
                data = data_fetcher.fetch_ohlcv(ticker, start_date, end_date)
                if data.empty:
                    raise ValueError(f"No data found for ticker {ticker}")
            except Exception as e:
                raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
            
            # Step 3: Initialize strategy
            update_progress(3, "Initializing strategy")
            strategy_type = get_session_state("strategy_type", "Moving Average Crossover")
            strategy = create_strategy_from_config()
            
            # Step 4: Generate trading signals
            update_progress(4, "Generating trading signals")
            try:
                signals = strategy.generate_signals(data)
                if signals.empty:
                    raise ValueError("Strategy generated no trading signals")
            except Exception as e:
                raise ValueError(f"Failed to generate signals: {str(e)}")
            
            # Step 5: Run simulation
            update_progress(5, "Running simulation")
            simulation_config = create_simulation_config()
            simulation_engine = SimulationEngine(simulation_config)
            try:
                simulation_result = simulation_engine.run_simulation(data, signals)
            except Exception as e:
                raise ValueError(f"Simulation failed: {str(e)}")
            
            # Step 6: Calculate performance metrics
            update_progress(6, "Calculating performance metrics")
            try:
                equity_curve_df = simulation_result.equity_curve
                if isinstance(equity_curve_df, pd.DataFrame):
                    equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
                else:
                    equity_series = equity_curve_df
                
                # Calculate performance metrics
                metrics = PerformanceMetrics.calculate_all_metrics(
                    equity_curve=equity_series,
                    trades=simulation_result.trades
                )
                
                # Store comprehensive results in session state
                set_session_state("backtest_results", {
                    "equity_curve": equity_series,
                    "trades": simulation_result.trades,
                    "metrics": metrics,
                    "signals": signals,
                    "data": data,
                    "strategy": strategy_type,  # Use the strategy_type variable instead
                    "simulation_config": simulation_config
                })
                
                # Step 7: Generate visualizations
                update_progress(7, "Generating visualizations")
                try:
                    # Store visualization data for charts tab
                    visualization_data = {
                        "data": data,
                        "signals": signals,
                        "equity_curve": equity_series,
                        "trades": simulation_result.trades,
                        "metrics": metrics
                    }
                    set_session_state("visualization_data", visualization_data)
                except Exception as viz_error:
                    st.warning(f"Visualization generation had issues: {str(viz_error)}")
                
                # Step 8: Finalize results
                update_progress(8, "Finalizing results")
                
                # Log successful completion
                st.session_state.last_backtest_time = datetime.now()
                
                # Store the final results in session state
                backtest_results = {
                    "status": "completed",
                    "equity_curve": equity_series,
                    "trades": simulation_result.trades,
                    "metrics": metrics,
                    "signals": signals,
                    "data": data,
                    "strategy": strategy_type,
                    "config": {
                        "ticker": ticker,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "strategy_type": strategy_type,
                        "strategy_params": strategy.get_parameters() if hasattr(strategy, 'get_parameters') else {},
                        "simulation_config": {
                            "initial_capital": simulation_config.initial_capital,
                            "leverage": simulation_config.leverage,
                            "transaction_cost": simulation_config.transaction_cost,
                            "slippage": simulation_config.slippage,
                            "position_sizing": simulation_config.position_sizing,
                            "position_size": simulation_config.position_size
                        }
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                set_session_state("backtest_results", backtest_results)
                
                # Log completion
                completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.last_backtest_time = completion_time
                
                # Return success with results
                return True, "Backtest completed successfully"
                
            except Exception as e:
                raise ValueError(f"Failed to calculate performance metrics: {str(e)}")
                
    except Exception as e:
        # Log the error for debugging
        error_msg = str(e)
        st.error(f"Backtest failed: {error_msg}")
        st.error("Debug Information:")
        st.code(traceback.format_exc())
        set_session_state("backtest_results", None)
        return False, error_msg


def validate_backtest_config():
    try:
        ticker = get_session_state("ticker", "")
        if not ticker or len(ticker.strip()) == 0:
            return False, "Ticker symbol is required"
        
        start_date = get_session_state("start_date", None)
        end_date = get_session_state("end_date", None)
        
        if not start_date or not end_date:
            return False, "Start and end dates are required"
        
        if start_date >= end_date:
            return False, "Start date must be before end date"
        
        if start_date.year < 2000:
            return False, "Start date must be from year 2000 onwards"
        
        if end_date > date.today():
            return False, "End date cannot be in the future"
        
        initial_capital = get_session_state("initial_capital", 0)
        if initial_capital <= 0:
            return False, "Initial capital must be greater than 0"
        
        leverage = get_session_state("leverage", 1.0)
        if leverage < 1.0 or leverage > 5.0:
            return False, "Leverage must be between 1.0 and 5.0"
        
        strategy_type = get_session_state("strategy_type", "")
        if strategy_type == "Moving Average Crossover":
            short_window = get_session_state("short_window", 0)
            long_window = get_session_state("long_window", 0)
            if short_window >= long_window:
                return False, "Short MA period must be less than Long MA period"
        
        return True, None
        
    except Exception as e:
        return False, f"Configuration validation error: {str(e)}"


def create_strategy_from_config():
    strategy_type = get_session_state("strategy_type", "Moving Average Crossover")
    if strategy_type == "Moving Average Crossover":
        short_window = get_session_state("short_window", 20)
        long_window = get_session_state("long_window", 50)
        return MovingAverageCrossoverStrategy(short_window, long_window)
    
    elif strategy_type == "RSI Strategy":
        rsi_period = get_session_state("rsi_period", 14)
        rsi_overbought = get_session_state("rsi_overbought", 70)
        rsi_oversold = get_session_state("rsi_oversold", 30)
        return RSIStrategy(rsi_period, rsi_overbought, rsi_oversold)
    
    elif strategy_type == "Bollinger Bands":
        bb_period = get_session_state("bb_period", 20)
        bb_std_multiplier = get_session_state("bb_std_multiplier", 2.0)
        bb_buy_threshold = get_session_state("bb_buy_threshold", 0.0)
        bb_sell_threshold = get_session_state("bb_sell_threshold", 0.0)
        return BollingerBandsStrategy(bb_period, bb_std_multiplier, bb_buy_threshold, bb_sell_threshold)
    
    elif strategy_type == "Stochastic Oscillator":
        stoch_k_period = get_session_state("stoch_k_period", 14)
        stoch_d_period = get_session_state("stoch_d_period", 3)
        stoch_oversold = get_session_state("stoch_oversold", 20.0)
        stoch_overbought = get_session_state("stoch_overbought", 80.0)
        return StochasticOscillatorStrategy(stoch_k_period, stoch_d_period, stoch_oversold, stoch_overbought)
    
    elif strategy_type == "Custom Strategy":
        # Get custom strategy configuration from JSON
        strategy_json = get_session_state("custom_strategy_json", "")
        
        # Provide a default configuration if JSON is empty or invalid
        default_config = {
            "rules": [
                {
                    "conditions": [
                        {
                            "indicator": "Close",
                            "operator": ">",
                            "value": 100
                        }
                    ],
                    "action": "buy"
                }
            ],
            "indicators": {}
        }
        
        try:
            if not strategy_json.strip():
                strategy_config = default_config
            else:
                strategy_config = json.loads(strategy_json)
            
            # Extract rules and indicators with fallbacks
            rules = strategy_config.get("rules", default_config["rules"])
            indicators = strategy_config.get("indicators", default_config["indicators"])
            
            # Validate that we have at least some rules
            if not rules:
                rules = default_config["rules"]
            
            # Create and configure custom strategy
            custom_strategy = CustomStrategy()
            custom_strategy.set_rules(rules)
            custom_strategy.set_indicators(indicators)
            
            return custom_strategy
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to a simple default strategy if configuration fails
            st.warning(f"Using default custom strategy due to configuration error: {str(e)}")
            custom_strategy = CustomStrategy()
            custom_strategy.set_rules(default_config["rules"])
            custom_strategy.set_indicators(default_config["indicators"])
            return custom_strategy
    
    elif strategy_type == "Buy and Hold":
        return MovingAverageCrossoverStrategy(1, 2)  # Will essentially buy and hold
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def create_simulation_config():
    initial_capital = get_session_state("initial_capital", 10000.0)
    leverage = get_session_state("leverage", 1.0)
    transaction_cost = get_session_state("transaction_cost", 0.001)
    slippage = get_session_state("slippage", 0.0005)
    position_sizing = get_session_state("position_sizing", "Fixed Fraction")
    position_sizing_method = "fixed_fraction" if position_sizing == "Fixed Fraction" else "fixed_size"
    position_size = get_session_state("position_size", 0.1)
    
    return SimulationConfig(
        initial_capital=initial_capital,
        leverage=leverage,
        transaction_cost=transaction_cost,
        slippage=slippage,
        position_sizing=position_sizing_method,
        position_size=position_size
    )

def render_results_tab():
    """Render the backtest results tab with performance metrics and summary"""
    results = get_session_state("backtest_results", None)
    
    if results is None:
        st.info("Configure your backtest settings and click 'Run Backtest' to see results here.")
        return
    
    if results.get("status") != "completed":
        st.warning("Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    # Main header
    section_header("Backtest Results", level=2)
    
    # Extract results data
    metrics = results.get("metrics", {})
    equity_curve = results.get("equity_curve")
    trades = results.get("trades", [])
    config = results.get("config", {})
    timestamp = results.get("timestamp", "N/A")
    
    # Display liquidation notice if applicable
    if metrics.get('final_liquidation_size'):
        size = metrics['final_liquidation_size']
        price = metrics['final_liquidation_price']
        value = metrics['final_liquidation_value']
        position_type = "BUY" if size > 0 else "SELL"
        
        with st.container():
            st.markdown("---")
            st.warning(
                f"End-of-Simulation Position Liquidation\n\n"
                f"At the end of the simulation, any open {position_type} position was automatically liquidated:\n"
                f"• Position Size: {abs(size):.6f} shares\n"
                f"• Liquidation Price: ${price:.2f}\n"
                f"• Total Value: ${value:.2f}\n\n"
                f"This is standard practice to close all positions and calculate final portfolio value.",
                icon=None
            )
            st.markdown("---")
    
    # Display equity curve and performance metrics
    equity_curve_df = results.get("equity_curve")
    if equity_curve_df is not None and len(equity_curve_df) > 0:
        if isinstance(equity_curve_df, pd.DataFrame):
            equity_curve = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
        else:
            equity_curve = equity_curve_df
            
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        if isinstance(initial_value, pd.Series):
            initial_value = initial_value.iloc[0] if len(initial_value) == 1 else initial_value.values[0]
        if isinstance(final_value, pd.Series):
            final_value = final_value.iloc[0] if len(final_value) == 1 else final_value.values[0]
            
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Calculate key metrics
        cagr = metrics.get("cagr", 0.0) * 100
        volatility = metrics.get("volatility", 0.0) * 100
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        sortino_ratio = metrics.get("sortino_ratio", 0.0)
        max_drawdown = metrics.get("max_drawdown", 0.0) * 100
        win_rate = metrics.get("win_rate", 0.0) * 100
        profit_factor = metrics.get("profit_factor", 0.0)
        avg_trade_return = metrics.get("avg_trade_return", 0.0) * 100
        max_consecutive_losses = metrics.get("max_consecutive_losses", 0)
        var_95 = metrics.get("var_95", 0.0)
        cvar_95 = metrics.get("cvar_95", 0.0)
        largest_win = metrics.get("largest_win", 0.0)
        largest_loss = metrics.get("largest_loss", 0.0)
        total_trades = len(trades)
        
        # Display results in the user's requested format
        with st.container():
            st.markdown("## Backtest Results")
            
            try:
                # Calculate buy and hold metrics for comparison
                data_df = results.get("data")
                if data_df is not None and len(data_df) >= 2:
                    initial_price = data_df['Close'].iloc[0]
                    final_price = data_df['Close'].iloc[-1]
                    bh_total_return = ((final_price / initial_price) - 1) * 100
                    
                    # Calculate buy and hold equity curve for additional metrics
                    bh_equity = (data_df['Close'] / initial_price) * initial_value
                    bh_returns = bh_equity.pct_change().dropna()
                    bh_sharpe = PerformanceMetrics.calculate_sharpe_ratio(bh_returns) if len(bh_returns) > 0 else 0
                else:
                    bh_total_return = 0
                    bh_sharpe = 0
                    
            except Exception:
                bh_total_return = 0
                bh_sharpe = 0
            
            # Strategy Return
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Strategy Return**")
            with col2:
                st.metric("", f"{total_return:.1f}%", f"{total_return:.1f}%")
            
            # Buy & Hold Return
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Buy & Hold Return**")
            with col2:
                st.metric("", f"{bh_total_return:.1f}%", f"{bh_total_return:.1f}%")
            
            # Outperformance
            outperformance = total_return - bh_total_return
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Outperformance**")
            with col2:
                st.metric("", f"{outperformance:.1f}%", f"{outperformance:.1f}%")
            
            # Sharpe Ratio with qualitative assessment
            def get_sharpe_assessment(sharpe):
                if sharpe >= 2.0:
                    return "Excellent"
                elif sharpe >= 1.0:
                    return "Good"
                elif sharpe >= 0.5:
                    return "Fair"
                else:
                    return "Poor"
            
            sharpe_assessment = get_sharpe_assessment(sharpe_ratio)
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Sharpe Ratio**")
            with col2:
                st.metric("", f"{sharpe_ratio:.2f}", sharpe_assessment)
            
            # Max Drawdown
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Max Drawdown**")
            with col2:
                st.metric("", f"{max_drawdown:.1f}%", f"-{max_drawdown:.1f}%")
            
            # Value at Risk (95%)
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Value at Risk (95%)**")
            with col2:
                st.metric("", f"{var_95:.2f}%", f"+{var_95:.2f}%")
            
            # Conditional VaR (95%)
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Conditional VaR (95%)**")
            with col2:
                st.metric("", f"{cvar_95:.2f}%", f"+{cvar_95:.2f}%")
            
            # Win Rate
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Win Rate**")
            with col2:
                st.metric("", f"{win_rate:.1f}%", f"{win_rate:.1f}%")
        
        # Additional detailed analysis in expanders
        with st.expander("📈 Detailed Performance Comparison"):
            
            try:
                # Calculate buy and hold metrics from the same data
                data_df = results.get("data")
                if data_df is not None and len(data_df) >= 2:
                    # Calculate buy and hold performance
                    initial_price = data_df['Close'].iloc[0]
                    final_price = data_df['Close'].iloc[-1]
                    bh_total_return = ((final_price / initial_price) - 1) * 100
                    
                    # Calculate buy and hold equity curve
                    bh_equity = (data_df['Close'] / initial_price) * initial_value
                    bh_returns = bh_equity.pct_change().dropna()
                    bh_volatility = PerformanceMetrics.calculate_volatility(bh_returns) * 100
                    bh_sharpe = PerformanceMetrics.calculate_sharpe_ratio(bh_returns)
                    bh_max_dd = PerformanceMetrics.calculate_max_drawdown(bh_equity) * 100
                    
                    periods_elapsed = len(data_df) - 1
                    years = periods_elapsed / 252
                    bh_cagr = ((final_price / initial_price) ** (1 / years) - 1) * 100 if years > 0 else 0
                    
                    # Display comparison
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Return", 
                            f"{total_return:.2f}%",
                            delta=f"{total_return - bh_total_return:.2f}% vs B&H"
                        )
                        st.metric(
                            "CAGR", 
                            f"{cagr:.2f}%",
                            delta=f"{cagr - bh_cagr:.2f}% vs B&H"
                        )
                    
                    with col2:
                        st.metric(
                            "Sharpe Ratio", 
                            f"{sharpe_ratio:.2f}",
                            delta=f"{sharpe_ratio - bh_sharpe:.2f} vs B&H"
                        )
                        st.metric(
                            "Volatility", 
                            f"{volatility:.2f}%",
                            delta=f"{volatility - bh_volatility:.2f}% vs B&H"
                        )
                    
                    with col3:
                        st.metric(
                            "Max Drawdown", 
                            f"{max_drawdown:.1f}%",
                            delta=f"{bh_max_dd - max_drawdown:.1f}% better" if max_drawdown < bh_max_dd else f"{max_drawdown - bh_max_dd:.1f}% worse"
                        )
                        
                        # Strategy advantage summary
                        strategy_advantage = total_return - bh_total_return
                        if strategy_advantage > 0:
                            st.success(f"Strategy outperformed by {strategy_advantage:.1f}%")
                        else:
                            st.error(f"Strategy underperformed by {abs(strategy_advantage):.1f}%")
                
                else:
                    st.warning("Buy & Hold comparison unavailable - insufficient data")
                    
            except Exception as e:
                st.warning(f"Buy & Hold comparison unavailable: {str(e)}")
        
        # Detailed metrics in expander
        with st.expander("📉 Advanced Risk Metrics"):
            section_header("Risk-Adjusted Returns", level=4, divider=False)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
            with col2:
                st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                st.metric("Volatility (Annualized)", f"{volatility:.2f}%")
            with col3:
                st.metric("VaR (95%)", f"{var_95:.2f}%")
                st.metric("CVaR (95%)", f"{cvar_95:.2f}%")
            
            section_header("Trade Statistics", level=4, divider=False)
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric("Total Trades", total_trades)
                st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
            
            with col5:
                st.metric("Avg Trade Return", f"{avg_trade_return:.2f}%")
                st.metric("Max Consecutive Losses", max_consecutive_losses)
            
            with col6:
                st.metric("Largest Win", f"${largest_win:,.2f}")
                st.metric("Largest Loss", f"${largest_loss:,.2f}")
        
        # Configuration details
        with st.expander("⚙️ View Backtest Configuration"):
            section_header("Backtest Configuration", level=4, divider=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                section_header("Data Settings", level=5, divider=False)
                st.write(f"**Ticker:** {config.get('ticker', get_session_state('ticker', 'N/A'))}")
                st.write(f"**Start Date:** {config.get('start_date', get_session_state('start_date', 'N/A'))}")
                st.write(f"**End Date:** {config.get('end_date', get_session_state('end_date', 'N/A'))}")
            
            with col2:
                section_header("Strategy Settings", level=5, divider=False)
                st.write(f"**Strategy:** {config.get('strategy_type', get_session_state('strategy_type', 'N/A'))}")
                strategy_params = config.get('strategy_params', {})
                if strategy_params:
                    st.write("**Parameters:**")
                    for param, value in strategy_params.items():
                        st.write(f"- {param.replace('_', ' ').title()}: {value}")
                else:
                    st.write("No strategy parameters configured.")
            
            section_header("Simulation Settings", level=5, divider=False)
            sim_config = config.get('simulation_config', {})
            col3, col4 = st.columns(2)
            
            with col3:
                initial_cap = sim_config.get('initial_capital', get_session_state('initial_capital', 0))
                leverage = sim_config.get('leverage', get_session_state('leverage', 1.0))
                st.write(f"**Initial Capital:** ${initial_cap:,.2f}")
                st.write(f"**Leverage:** {leverage:.1f}x")
            
            with col4:
                trans_cost = sim_config.get('transaction_cost', get_session_state('transaction_cost', 0.0))
                slip = sim_config.get('slippage', get_session_state('slippage', 0.0))
                st.write(f"**Transaction Cost:** {trans_cost*100:.3f}%")
                st.write(f"**Slippage:** {slip*100:.3f}%")

def render_charts_tab():
    """Render the charts tab with performance visualizations"""
    results = get_session_state("backtest_results", None)
    
    if results is None:
        st.info("Run a backtest first to see charts here.")
        return
    
    if results.get("status") != "completed":
        st.warning("Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    section_header("Performance Charts", level=2)
    
    # Equity Curve Section
    with st.container():
        section_header("Equity Curve", level=4, divider=False)
        
        equity_curve_df = results.get("equity_curve")
        if equity_curve_df is not None:
            if isinstance(equity_curve_df, pd.DataFrame):
                equity_curve = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
            else:
                equity_curve = equity_curve_df
                
            # Chart controls
            col1, col2 = st.columns([1, 2])
            with col1:
                show_benchmark = st.checkbox("Show Benchmark", value=False, key="benchmark_checkbox")
                show_annotations = st.checkbox("Show Annotations", value=True, key="annotations_checkbox")
                chart_height = st.slider("Chart Height (pixels)", 400, 1000, 600, 50, key="height_slider")
            
            # Generate the equity curve plot
            benchmark_data = None  # TODO: Implement benchmark data fetching if needed
            fig = Visualizer.plot_equity_curve(
                equity_curve, 
                title="Strategy Performance",
                show_annotations=show_annotations,
                height=chart_height,
                benchmark=benchmark_data if show_benchmark else None
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, key="equity_curve_chart")
            
            # Add download button for the chart
            col1, col2 = st.columns(2)
            with col1:
                try:
                    # Try to create PNG export
                    png_data = fig.to_image(format="png")
                    st.download_button(
                        label="Download Chart as PNG",
                        data=png_data,
                        file_name="equity_curve.png",
                        mime="image/png"
                    )
                except Exception as e:
                    # Fallback: provide HTML export or inform about missing package
                    if "kaleido" in str(e).lower():
                        st.info("📊 PNG export requires the 'kaleido' package. Install with: pip install kaleido")
                    else:
                        st.warning(f"Chart export unavailable: {e}")
                        
                    # Provide HTML export as alternative
                    html_data = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="Download Chart as HTML",
                        data=html_data,
                        file_name="equity_curve.html",
                        mime="text/html"
                    )
            with col2:
                st.download_button(
                    label="Download Data as CSV",
                    data=equity_curve_df.to_csv(index=True),
                    file_name="equity_curve.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No equity curve data available for the selected backtest.")
    
    # Drawdown Analysis Section
    with st.container():
        section_header("Drawdown Analysis", level=4, divider=True)
        
        if equity_curve_df is not None:
            if isinstance(equity_curve_df, pd.DataFrame):
                equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
            else:
                equity_series = equity_curve_df
                
            drawdown = calculate_drawdown_series(equity_series)
            fig_dd = Visualizer.plot_drawdown(
                drawdown, 
                title="Strategy Drawdown",
                height=500
            )
            st.plotly_chart(fig_dd, use_container_width=True, key="drawdown_chart")
        else:
            st.info("Drawdown analysis will be displayed after running a backtest")
    
    # Returns Distribution Section
    with st.container():
        section_header("Returns Distribution", level=4, divider=True)
        
        if equity_curve_df is not None:
            if isinstance(equity_curve_df, pd.DataFrame):
                equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
            else:
                equity_series = equity_curve_df
                
            returns = equity_series.pct_change().dropna()
            fig_dist = Visualizer.plot_returns_distribution(
                returns, 
                title="Daily Returns Distribution",
                height=500
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="returns_dist")
        else:
            st.info("Returns distribution will be displayed after running a backtest")
    
    # Rolling Performance Metrics Section
    with st.container():
        section_header("Rolling Performance Metrics", level=4, divider=True)
        
        if equity_curve_df is not None:
            if isinstance(equity_curve_df, pd.DataFrame):
                equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
            else:
                equity_series = equity_curve_df
            
            # Metrics configuration
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider(
                    "Rolling Window (days)", 
                    min_value=10, 
                    max_value=120, 
                    value=30,
                    step=5,
                    key="window_size_slider"
                )
            with col2:
                metrics_to_show = st.multiselect(
                    "Metrics to Display",
                    options=["returns", "volatility", "sharpe"],
                    default=["returns", "volatility", "sharpe"],
                    key="metrics_multiselect"
                )
            
            if metrics_to_show:
                fig_rolling = Visualizer.plot_rolling_metrics(
                    equity_series, 
                    window=window_size,
                    metrics=metrics_to_show,
                    title=f"{window_size}-Day Rolling Metrics",
                    height=500
                )
                st.plotly_chart(fig_rolling, use_container_width=True, key="rolling_metrics")
            else:
                st.info("Select at least one metric to display")


def calculate_drawdown_series(equity_curve):
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return -drawdown

def render_trade_log_tab():
    """Render the trade log tab with detailed trade information and filtering options"""
    results = get_session_state("backtest_results", None)
    
    if results is None:
        st.info("Run a backtest first to see the trade log.")
        return
    
    if results.get("status") != "completed":
        st.warning("Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    section_header("Trade Log", level=2)
    trades = results.get('trades', [])
    total_trades = len(trades)
    
    if total_trades > 0:
        trade_data = []
        validation_errors = []
        
        # Process and validate trades
        for trade in trades:
            try:
                pnl = getattr(trade, 'pnl', None)
                value = float(getattr(trade, 'value', 0.0))
                
                if pnl is not None and value > 0:
                    pnl_pct = (pnl / value) * 100
                    pnl_pct = max(min(pnl_pct, 50.0), -50.0)
                else:
                    pnl_pct = 0.0
                
                entry_price = float(trade.entry_price)
                exit_price = float(trade.exit_price if trade.exit_price is not None else 0.0)
                if entry_price > 0 and exit_price > 0:
                    if getattr(trade, 'type', 'Unknown') == 'buy':
                        actual_return_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        actual_return_pct = ((entry_price - exit_price) / entry_price) * 100
                else:
                    actual_return_pct = 0.0
                
                trade_type = getattr(trade, 'type', 'Unknown')
                if hasattr(trade_type, 'value'):
                    trade_type = trade_type.value
                
                duration = getattr(trade, 'duration', None)
                duration = int(duration) if duration is not None else 1
                
                # Filter out trades with minimal returns (this excludes BUY trades)
                if abs(actual_return_pct) < 0.0001:
                    continue
                    
                trade_dict = {
                    'Date': getattr(trade, 'timestamp', None),
                    'Type': trade_type,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Size': float(getattr(trade, 'size', 0.0)),
                    'Value': value,
                    'Return (%)': actual_return_pct,
                    'PnL': float(pnl) if pnl is not None else 0.0,
                    'Duration (days)': duration,
                    'Commission': float(getattr(trade, 'commission', 0.0)),
                    'Slippage': float(getattr(trade, 'slippage', 0.0))
                }
                
                # Validate trade data
                validation_issues = []
                if trade_dict['Date'] is None:
                    validation_issues.append('missing date')
                if trade_dict['Entry Price'] <= 0:
                    validation_issues.append('invalid entry price')
                if abs(trade_dict['Size']) < 0.000001:
                    validation_issues.append('invalid size')
                
                if validation_issues:
                    validation_errors.append(f"Trade validation failed: {', '.join(validation_issues)}")
                else:
                    trade_data.append(trade_dict)
                    
            except Exception as e:
                validation_errors.append(f"Error processing trade: {str(e)}")
                continue
        
        # Display validation errors if any
        if validation_errors:
            with st.expander("View Validation Warnings", expanded=False):
                for error in validation_errors:
                    st.warning(error)
        
        # Trade Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            winning_trades = len([t for t in trade_data if t['Return (%)'] > 0])
            st.metric("Winning Trades", f"{winning_trades}")
        
        with col2:
            losing_trades = len([t for t in trade_data if t['Return (%)'] < 0])
            st.metric("Losing Trades", f"{losing_trades}")
        
        with col3:
            winning_returns = [t['Return (%)'] for t in trade_data if t['Return (%)'] > 0]
            if winning_returns:
                avg_win = np.mean([r for r in winning_returns if abs(r) <= 100])
                st.metric("Avg Win", f"{avg_win:.2f}%")
            else:
                st.metric("Avg Win", "0.00%")
        
        with col4:
            losing_returns = [t['Return (%)'] for t in trade_data if t['Return (%)'] < 0]
            if losing_returns:
                avg_loss = np.mean([r for r in losing_returns if abs(r) <= 100])
                st.metric("Avg Loss", f"{avg_loss:.2f}%")
            else:
                st.metric("Avg Loss", "0.00%")
        
        st.subheader("Filter Trades")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_type_filter = st.selectbox(
                "Trade Type",
                options=["All", "Buy", "Sell"],
                index=0
            )
        
        with col2:
            profit_filter = st.selectbox(
                "Profit/Loss",
                options=["All", "Profitable", "Losing"],
                index=0
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=["Date", "Return (%)", "Size", "Duration"],
                index=0
            )
        
        filtered_data = apply_trade_filters(trade_data, trade_type_filter, profit_filter)
        if sort_by == "Date":
            filtered_data = sorted(filtered_data, key=lambda x: x['Date'])
        elif sort_by == "Return (%)":
            filtered_data = sorted(filtered_data, key=lambda x: x['Return (%)'], reverse=True)
        elif sort_by == "Size":
            filtered_data = sorted(filtered_data, key=lambda x: x['Size'], reverse=True)
        elif sort_by == "Duration":
            filtered_data = sorted(filtered_data, key=lambda x: x['Duration (days)'], reverse=True)
        
        st.subheader(f"📈 Trade Details ({len(filtered_data)} trades)")
        
        if filtered_data:
            # Create DataFrame with proper sanitization for Streamlit display
            df = pd.DataFrame(filtered_data)
            
            # Sanitize all columns to prevent ArrowTypeError with Timestamp objects
            for col in df.columns:
                try:
                    # Handle Date columns specifically
                    if 'Date' in col:
                        # Convert any Timestamp objects to string format
                        df[col] = df[col].apply(lambda x: 
                            x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') 
                            else str(x) if x is not None else '')
                    # Handle object columns that might contain Timestamp objects
                    elif df[col].dtype == 'object':
                        df[col] = df[col].apply(lambda x: 
                            x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') 
                            else str(x) if x is not None else '')
                    # Handle datetime columns
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Handle numeric columns with potential issues
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(0)
                        df[col] = df[col].replace([np.inf, -np.inf], 0)
                except Exception as col_error:
                    # If any column processing fails, convert to string
                    df[col] = df[col].astype(str)
            
            def style_returns(val):
                try:
                    # Ensure val is numeric for comparison
                    val_float = float(val) if not pd.isna(val) and val != '' else 0
                    color = 'green' if val_float > 0 else 'red' if val_float < 0 else 'black'
                    return f'color: {color}'
                except:
                    return 'color: black'
            
            # Apply styling with error handling
            try:
                styled_df = df.style.applymap(style_returns, subset=['Return (%)'])
            except Exception:
                # If styling fails, use unstyled dataframe
                styled_df = df
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Entry Price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                    "Exit Price": st.column_config.NumberColumn("Exit Price", format="$%.2f"),
                    "Size": st.column_config.NumberColumn("Size", format="%.0f"),
                    "Value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "Return (%)": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
                    "Duration (days)": st.column_config.NumberColumn("Duration", format="%.0f"),
                    "Commission": st.column_config.NumberColumn("Commission", format="$%.2f"),
                }
            )
            
            with st.expander("📈 Trade Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Performance Statistics**")
                    total_return = sum(t['Return (%)'] for t in filtered_data)
                    st.write(f"• Total Return: {total_return:.2f}%")
                    st.write(f"• Best Trade: {max(t['Return (%)'] for t in filtered_data):.2f}%")
                    st.write(f"• Worst Trade: {min(t['Return (%)'] for t in filtered_data):.2f}%")
                    st.write(f"• Average Trade: {np.mean([t['Return (%)'] for t in filtered_data]):.2f}%")
                
                with col2:
                    st.write("**Trade Characteristics**")
                    avg_duration = np.mean([t['Duration (days)'] for t in filtered_data])
                    st.write(f"• Average Duration: {avg_duration:.1f} days")
                    total_commission = sum(t['Commission'] for t in filtered_data)
                    st.write(f"• Total Commissions: ${total_commission:.2f}")
                    avg_size = np.mean([t['Size'] for t in filtered_data])
                    st.write(f"• Average Position Size: {avg_size:.0f} shares")
        else:
            st.info("No trades match the selected filters.")
    
    else:
        st.info("No trades were executed during this backtest period.")


def apply_trade_filters(trade_data, trade_type_filter, profit_filter):
    filtered_data = trade_data.copy()
    if trade_type_filter != "All":
        trade_type_lower = trade_type_filter.lower()
        filtered_data = [t for t in filtered_data if str(t['Type']).lower() == trade_type_lower]
    
    if profit_filter == "Profitable":
        filtered_data = [t for t in filtered_data if t['Return (%)'] > 0]
    elif profit_filter == "Losing":
        filtered_data = [t for t in filtered_data if t['Return (%)'] < 0]
    
    return filtered_data

def render_about_tab():
    """Render the About tab with application information and documentation"""
    section_header("About Trading Strategy Backtester", level=2)
    
    with st.container():
        section_header("Overview", level=3, divider=False)
        st.markdown("""
        This application provides a comprehensive platform for backtesting trading strategies
        using historical market data. It's designed to help traders and analysts evaluate
        the performance of their strategies before risking real capital in live markets.
        """)
    
    with st.container():
        section_header("Features", level=3, divider=True)
        
        with st.expander("Strategy Testing"):
            st.markdown("""
            - Multiple pre-built strategies (Moving Average, RSI, Bollinger Bands, Stochastic Oscillator, Buy & Hold)
            - Custom strategy definition capabilities
            - Configurable parameters for each strategy
            - Support for both long and short positions
            """)
        
        with st.expander("Risk Management"):
            st.markdown("""
            - Adjustable position sizing
            - Stop-loss and take-profit order support
            - Transaction costs and slippage simulation
            - Margin requirements and leverage control
            """)
        
        with st.expander("Performance Analysis"):
            st.markdown("""
            - Comprehensive performance metrics (Sharpe, Sortino, Calmar ratios)
            - Risk metrics (Max Drawdown, Volatility, Value at Risk)
            - Trade statistics and analytics
            - Benchmark comparison capabilities
            """)
        
        with st.expander("Visualization"):
            st.markdown("""
            - Interactive equity curve with drawdown visualization
            - Returns distribution and rolling metrics
            - Trade execution markers on price charts
            - Exportable charts and reports
            """)
    
    with st.container():
        section_header("Getting Started", level=3, divider=True)
        
        st.markdown("### Step-by-Step Guide")
        
        steps = [
            "1. **Configure**: Set up your backtest parameters in the Configuration tab",
            "2. **Select Strategy**: Choose from built-in strategies or define a custom one",
            "3. **Run Backtest**: Execute the backtest with your chosen settings",
            "4. **Analyze Results**: Review performance metrics and visualizations",
            "5. **Optimize**: Adjust parameters and re-run to improve strategy performance"
        ]
        
        for step in steps:
            st.markdown(step)
    
    with st.container():
        section_header("Technical Details", level=3, divider=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data")
            st.markdown("""
            - **Data Sources**: Historical OHLCV data from multiple providers
            - **Timeframes**: Daily, hourly, and minute data support
            - **Asset Classes**: Equities, Forex
            - **Data Quality**: Cleaned and adjusted for corporate actions
            """)
        
        with col2:
            st.markdown("### Technology Stack")
            st.markdown("""
            - **Backend**: Python 3.9+
            - **Data Processing**: Pandas, NumPy
            - **Visualization**: Plotly, Streamlit
            - **Performance**: Numba-accelerated calculations
            - **Storage**: Local SQLite for results caching
            """)
    
    with st.container():
        section_header("Disclaimer", level=3, divider=True)
        
        st.warning("""
        **Important Notice**: 
        
        This application is provided for educational and research purposes only. 
        
        - Past performance is not indicative of future results.
        - Backtested results may not reflect actual trading conditions.
        - Always conduct thorough analysis and consider consulting with financial professionals 
          before making any investment decisions.
        - The developers assume no liability for any financial losses incurred through the use of this application.
        """)

def main():
    """Main application entry point"""
    try:
        # Initialize application state and configuration
        initialize_session_state()
        
        # Configure page (moved to main.py, but keeping the function call for backward compatibility)
        configure_page()
        
        # Render UI components
        try:
            render_header()
            tabs = render_navigation()
            
            # Tab routing with error handling for each tab
            tab_handlers = [
                (0, "Strategy Builder", render_configuration_tab),
                (1, "Backtest Results", render_results_tab),
                (2, "Performance Charts", render_charts_tab),
                (3, "Trade Analysis", render_trade_log_tab),
                (4, "Market News", render_alpha_news_tab),
                (5, "AI Strategy Advisor", render_enhanced_recommendation_tab),
                (6, "About", render_about_tab)
            ]
            
            for tab_index, tab_name, tab_handler in tab_handlers:
                try:
                    with tabs[tab_index]:
                        tab_handler()
                except Exception as tab_error:
                    st.error(f"Error in {tab_name} tab: {str(tab_error)}")
                    st.exception(tab_error)
            
            # Footer with version and copyright
            render_footer()
            
        except Exception as ui_error:
            st.error("An error occurred while rendering the UI")
            st.exception(ui_error)
            
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {str(e)}")
        st.exception(e)

def apply_global_styles():
    """Apply global CSS styles for consistent theming"""
    st.markdown("""
    <style>
    /* Global styles */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
        }
        
        /* Consistent section headers */
        h1, h2, h3, h4, h5, h6 {
            color: #1E3F66;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Form elements */
        .stSelectbox, .stNumberInput, .stTextInput, .stDateInput {
            margin-bottom: 1rem;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #1E3F66;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #2A5A8C;
            color: white;
        }
        
        /* Tables */
        .stDataFrame {
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #1E3F66;
        }
    </style>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the application footer"""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d; padding: 1rem 0; font-size: 0.9rem;'>"
        "<a href='https://shravankumar05.github.io/' target='_blank' style='color: #1E3F66; text-decoration: none;'>"
        "My Website"
        "</a> | "
        "<a href='https://github.com/ShravanKumar05/Strategy-Backtester' target='_blank' style='color: #1E3F66; text-decoration: none;'>"
        "GitHub Repository  | "
        "<a href='mailto:shravankumar.murki@gmail.com' target='_blank' style='color: #1E3F66; text-decoration: none;'>"
        "My Email"
        "</b>"
        "<t> | This is my personal project – not financial advice or recommendation </t>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()