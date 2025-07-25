import streamlit as st
import json
from datetime import date
from typing import Dict, Any, Optional, List

def render_input_controls() -> Dict[str, Any]:
    config = {}
    
    st.subheader("📅 Data Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        config['ticker'] = st.text_input("Stock Ticker", value="AAPL", help="Enter a valid stock ticker symbol").upper()
    
    with col2:
        config['data_source'] = st.selectbox("Data Source", options=["Yahoo Finance"], index=0, help="Data provider for historical prices")

    col1, col2 = st.columns(2)
    with col1:
        config['start_date'] = st.date_input("Start Date", value=date(2024, 1, 1), min_value=date(2024, 1, 1), max_value=date(2024, 12, 31))
    
    with col2:
        config['end_date'] = st.date_input("End Date", value=date(2024, 6, 30), min_value=date(2024, 1, 1), max_value=date(2024, 12, 31))
    
    if config['start_date'] >= config['end_date']:
        st.error("Start date must be before end date")
        return None
    
    st.markdown("---")
    st.subheader("📈 Strategy Configuration")
    config['strategy_type'] = st.selectbox("Strategy Type", options=["Moving Average Crossover", "RSI Strategy", "Buy and Hold", "Custom Strategy"], help="Select the trading strategy to test")
    
    if config['strategy_type'] == "Moving Average Crossover":
        config.update(render_ma_crossover_params())
    
    elif config['strategy_type'] == "RSI Strategy":
        config.update(render_rsi_strategy_params())
    
    elif config['strategy_type'] == "Custom Strategy":
        custom_config = render_custom_strategy_editor()
        if custom_config is None:
            return None
        config.update(custom_config)
    
    st.markdown("---")
    st.subheader("💰 Capital & Risk Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        config['initial_capital'] = st.number_input("Initial Capital ($)", min_value=1000.0, max_value=10000000.0, value=10000.0, step=1000.0)
    
    with col2:
        config['leverage'] = st.slider("Leverage", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        config['transaction_cost'] = st.number_input("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.3f") / 100
    
    with col2:
        config['slippage'] = st.number_input("Slippage (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.3f") / 100
    
    config['position_sizing'] = st.selectbox("Position Sizing Method", options=["Fixed Fraction", "Fixed Size"])
    
    if config['position_sizing'] == "Fixed Fraction":
        config['position_size'] = st.slider("Position Size (% of capital)", min_value=1, max_value=100, value=10) / 100
    else:
        config['position_size'] = st.number_input("Position Size ($)", min_value=100.0, max_value=config['initial_capital'], value=1000.0, step=100.0)
    
    return config


def render_ma_crossover_params() -> Dict[str, Any]:
    params = {}
    col1, col2 = st.columns(2)
    with col1:
        params['short_ma'] = st.number_input("Short MA Period", min_value=5, max_value=50, value=20, help="Period for short moving average")
    
    with col2:
        params['long_ma'] = st.number_input("Long MA Period", min_value=20, max_value=200, value=50, help="Period for long moving average")
    
    if params['short_ma'] >= params['long_ma']:
        st.error("Short MA period must be less than Long MA period")
        return {}
    
    return params


def render_rsi_strategy_params() -> Dict[str, Any]:
    params = {}
    col1, col2, col3 = st.columns(3)
    with col1:
        params['rsi_period'] = st.number_input("RSI Period", min_value=5, max_value=30, value=14, help="Period for RSI calculation")
    
    with col2:
        params['rsi_overbought'] = st.number_input("Overbought Level", min_value=60, max_value=90, value=70, help="RSI level considered overbought")
    
    with col3:
        params['rsi_oversold'] = st.number_input("Oversold Level", min_value=10, max_value=40, value=30, help="RSI level considered oversold")
    
    if params['rsi_oversold'] >= params['rsi_overbought']:
        st.error("Oversold level must be less than overbought level")
        return {}
    
    return params


def render_custom_strategy_editor() -> Optional[Dict[str, Any]]:
    st.info("📝 Define your custom trading strategy using JSON rules")
    
    default_strategy = {
        "name": "My Custom Strategy",
        "description": "Custom strategy with buy/sell rules",
        "buy_rules": [
            {
                "indicator": "rsi_14",
                "operator": "<",
                "value": 30,
                "description": "Buy when RSI is oversold"
            }
        ],
        "sell_rules": [
            {
                "indicator": "rsi_14", 
                "operator": ">",
                "value": 70,
                "description": "Sell when RSI is overbought"
            }
        ]
    }
    
    with st.expander("📖 Strategy Format Documentation"):
        st.markdown("""
        **Strategy JSON Format:**
        
        ```json
        {
            "name": "Strategy Name",
            "description": "Strategy description",
            "buy_rules": [
                {
                    "indicator": "rsi_14",
                    "operator": "<",
                    "value": 30,
                    "description": "Buy condition description"
                }
            ],
            "sell_rules": [
                {
                    "indicator": "rsi_14",
                    "operator": ">", 
                    "value": 70,
                    "description": "Sell condition description"
                }
            ]
        }
        ```
        
        **Available Indicators:**
        - `close`: Closing price
        - `open`: Opening price
        - `high`: High price
        - `low`: Low price
        - `volume`: Trading volume
        - `rsi_14`: 14-period RSI
        - `sma_20`: 20-period Simple Moving Average
        - `sma_50`: 50-period Simple Moving Average
        - `ema_12`: 12-period Exponential Moving Average
        - `ema_26`: 26-period Exponential Moving Average
        
        **Available Operators:**
        - `>`: Greater than
        - `<`: Less than
        - `>=`: Greater than or equal
        - `<=`: Less than or equal
        - `==`: Equal to
        - `!=`: Not equal to
        """)
    
    tab1, tab2 = st.tabs(["📝 JSON Editor", "🔧 Visual Builder"])
    with tab1:
        strategy_json = st.text_area("Strategy JSON", value=json.dumps(default_strategy, indent=2), height=400, help="Enter your strategy as JSON")
        
        try:
            strategy_config = json.loads(strategy_json)
            validation_result = validate_custom_strategy(strategy_config)
            if validation_result["valid"]:
                st.success("✅ Strategy JSON is valid!")
                return {"custom_strategy": strategy_config}
            else:
                st.error(f"❌ Strategy validation failed: {validation_result['error']}")
                return None
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {str(e)}")
            return None
    
    with tab2:
        st.info("🚧 Visual strategy builder coming soon! Use JSON editor for now.")
        return None


def validate_custom_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = ["name", "buy_rules", "sell_rules"]
    for field in required_fields:
        if field not in strategy:
            return {"valid": False, "error": f"Missing required field: {field}"}
    
    for rule_type in ["buy_rules", "sell_rules"]:
        rules = strategy[rule_type]
        if not isinstance(rules, list):
            return {"valid": False, "error": f"{rule_type} must be a list"}
        
        if len(rules) == 0:
            return {"valid": False, "error": f"{rule_type} cannot be empty"}
        
        for i, rule in enumerate(rules):
            validation = validate_rule(rule, f"{rule_type}[{i}]")
            if not validation["valid"]:
                return validation
    
    return {"valid": True, "error": None}


def validate_rule(rule: Dict[str, Any], rule_path: str) -> Dict[str, Any]:
    required_fields = ["indicator", "operator", "value"]
    for field in required_fields:
        if field not in rule:
            return {"valid": False, "error": f"{rule_path}: Missing required field '{field}'"}
    
    valid_indicators = ["close", "open", "high", "low", "volume", "rsi_14", "sma_20", "sma_50", "ema_12", "ema_26"]
    
    if rule["indicator"] not in valid_indicators:
        return {
            "valid": False, 
            "error": f"{rule_path}: Invalid indicator '{rule['indicator']}'. Valid options: {', '.join(valid_indicators)}"
        }
    
    valid_operators = [">", "<", ">=", "<=", "==", "!="]
    if rule["operator"] not in valid_operators:
        return {
            "valid": False,
            "error": f"{rule_path}: Invalid operator '{rule['operator']}'. Valid options: {', '.join(valid_operators)}"
        }
    
    if not isinstance(rule["value"], (int, float)):
        return {"valid": False, "error": f"{rule_path}: Value must be numeric"}
    
    return {"valid": True, "error": None}


def render_advanced_settings() -> Dict[str, Any]:
    advanced_config = {}
    with st.expander("⚙️ Advanced Settings"):
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        with col1:
            advanced_config['stop_loss'] = st.number_input("Stop Loss (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, help="Stop loss percentage (0 = disabled)")
        
        with col2:
            advanced_config['take_profit'] = st.number_input("Take Profit (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, help="Take profit percentage (0 = disabled)")
        
        st.subheader("Execution Settings")
        col1, col2 = st.columns(2)
        with col1:
            advanced_config['max_positions'] = st.number_input("Max Concurrent Positions", min_value=1, max_value=10, value=1, help="Maximum number of concurrent positions")
        
        with col2:
            advanced_config['rebalance_frequency'] = st.selectbox("Rebalance Frequency", options=["Daily", "Weekly", "Monthly"], index=0, help="How often to rebalance positions")
        
        st.subheader("Data Settings")
        
        advanced_config['warm_up_period'] = st.number_input("Warm-up Period (days)", min_value=0, max_value=252, value=50, help="Number of days for indicator warm-up")
        advanced_config['benchmark'] = st.selectbox("Benchmark", options=["SPY", "QQQ", "IWM", "None"], index=0, help="Benchmark for comparison")
    return advanced_config


def get_ticker_suggestions() -> List[str]:
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "JPM", "BAC", "WFC", "GS", "MS", "C", "JNJ", "PFE", "UNH", "ABBV", "MRK", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SPY", "QQQ", "IWM", "VTI", "VOO"]


def render_ticker_input_with_suggestions() -> str:
    suggestions = get_ticker_suggestions()
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter a valid stock ticker symbol").upper()
    
    with col2:
        st.write("**Popular:**")
        selected_suggestion = st.selectbox("Quick Select", options=[""] + suggestions, index=0, label_visibility="collapsed")
        
        if selected_suggestion:
            ticker = selected_suggestion
    return ticker