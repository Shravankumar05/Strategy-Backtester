import streamlit as st
import json
from datetime import date
from typing import Dict, Any, Optional, List

def render_input_controls() -> Dict[str, Any]:
    # Apply custom styles
    st.markdown("""
    <style>
    .section-header {
        color: #1e3c72;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e9ecef;
    }
    
    .form-group {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .stDateInput, .stTextInput, .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    .stDateInput>div>div>input,
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div>div,
    .stNumberInput>div>div>input {
        border: 1px solid #ced4da;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
    }
    
    .stDateInput>div>div>input:focus,
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div>div:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #1e3c72;
        box-shadow: 0 0 0 2px rgba(30, 60, 114, 0.2);
    }
    
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e9ecef, transparent);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    config = {}
    
    # Data Configuration Section
    st.markdown("<div class='section-header'>Data Configuration</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='form-group'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            config['ticker'] = st.text_input(
                "Stock Ticker", 
                value="AAPL", 
                help="Enter a valid stock ticker symbol"
            ).upper()
        
        with col2:
            config['data_source'] = st.selectbox(
                "Data Source", 
                options=["Yahoo Finance"], 
                index=0, 
                help="Data provider for historical prices"
            )

        col1, col2 = st.columns(2)
        with col1:
            config['start_date'] = st.date_input(
                "Start Date", 
                value=date(2024, 1, 1), 
                min_value=date(2020, 1, 1), 
                max_value=date(2024, 12, 31)
            )
        
        with col2:
            config['end_date'] = st.date_input(
                "End Date", 
                value=date(2024, 6, 30), 
                min_value=date(2020, 1, 1), 
                max_value=date(2024, 12, 31)
            )
        
        if config['start_date'] >= config['end_date']:
            st.error("Start date must be before end date")
            return None
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Strategy Configuration Section
    st.markdown("<div class='section-header'>Strategy Configuration</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='form-group'>", unsafe_allow_html=True)
        
        config['strategy_type'] = st.selectbox(
            "Strategy Type", 
            options=["Moving Average Crossover", "RSI Strategy", "Buy and Hold", "Custom Strategy"], 
            help="Select the trading strategy to test"
        )
    
    if config['strategy_type'] == "Moving Average Crossover":
        ma_config = render_ma_crossover_params()
        if not ma_config:
            return None
        config.update(ma_config)
    
    elif config['strategy_type'] == "RSI Strategy":
        rsi_config = render_rsi_strategy_params()
        if not rsi_config:
            return None
        config.update(rsi_config)
    
    elif config['strategy_type'] == "Custom Strategy":
        custom_config = render_custom_strategy_editor()
        if custom_config is None:
            return None
        config.update(custom_config)
    
    # Close the form group div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Capital & Risk Configuration Section
    st.markdown("<div class='section-header'>Capital & Risk Configuration</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='form-group'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            config['initial_capital'] = st.number_input(
                "Initial Capital ($)", 
                min_value=1000.0, 
                max_value=10000000.0, 
                value=10000.0, 
                step=1000.0,
                help="Initial capital to start the backtest with"
            )
        
        with col2:
            config['leverage'] = st.slider(
                "Leverage", 
                min_value=1.0, 
                max_value=10.0, 
                value=1.0, 
                step=0.1,
                help="Leverage multiplier for positions"
            )
        
        st.markdown("<div style='margin: 1rem 0; font-size: 0.9rem; color: #4a5568;'>Transaction Settings</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            config['transaction_cost'] = st.number_input(
                "Transaction Cost (%)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.1, 
                step=0.01, 
                format="%.3f",
                help="Transaction cost as a percentage of trade value"
            ) / 100
        
        with col2:
            config['slippage'] = st.number_input(
                "Slippage (%)", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.05, 
                step=0.01, 
                format="%.3f",
                help="Expected slippage as a percentage of trade value"
            ) / 100
        
        st.markdown("<div style='margin: 1rem 0; font-size: 0.9rem; color: #4a5568;'>Position Sizing</div>", unsafe_allow_html=True)
        
        config['position_sizing'] = st.selectbox(
            "Position Sizing Method", 
            options=["Fixed Fraction", "Fixed Size"],
            help="Method to determine position sizing"
        )
        
        if config['position_sizing'] == "Fixed Fraction":
            config['position_size'] = st.slider(
                "Position Size (% of capital)", 
                min_value=1, 
                max_value=100, 
                value=10,
                help="Percentage of capital to risk per trade"
            ) / 100
        else:
            config['position_size'] = st.number_input(
                "Position Size ($)", 
                min_value=100.0, 
                max_value=config['initial_capital'], 
                value=min(1000.0, config['initial_capital']), 
                step=100.0,
                help="Fixed dollar amount to risk per trade"
            )
        
        # Close the form group div
        st.markdown("</div>", unsafe_allow_html=True)
    
    return config


def render_ma_crossover_params() -> Dict[str, Any]:
    params = {}
    st.markdown("<div class='form-group'>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 1rem; color: #4a5568;'>Moving Average Crossover Strategy Parameters</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        params['short_ma'] = st.number_input(
            "Short MA Period", 
            min_value=5, 
            max_value=50, 
            value=20, 
            help="Period for short moving average (shorter-term trend)",
            key="short_ma_input"
        )
    
    with col2:
        params['long_ma'] = st.number_input(
            "Long MA Period", 
            min_value=20, 
            max_value=200, 
            value=50, 
            help="Period for long moving average (longer-term trend)",
            key="long_ma_input"
        )
    
    if params['short_ma'] >= params['long_ma']:
        st.error("Short MA period must be less than Long MA period")
        st.markdown("</div>", unsafe_allow_html=True)
        return {}
    
    # Add some strategy description
    st.markdown("""
    <div style="margin-top: 1rem; padding: 1rem; background-color: #f0f4f8; border-radius: 6px; font-size: 0.9rem; color: #4a5568;">
        <strong>Strategy Logic:</strong> Buy when Short MA crosses above Long MA, sell when it crosses below.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    return params


def render_rsi_strategy_params() -> Dict[str, Any]:
    params = {}
    st.markdown("<div class='form-group'>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 1rem; color: #4a5568;'>RSI Strategy Parameters</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        params['rsi_period'] = st.number_input(
            "RSI Period", 
            min_value=5, 
            max_value=30, 
            value=14, 
            help="Lookback period for RSI calculation",
            key="rsi_period_input"
        )
    
    with col2:
        params['rsi_overbought'] = st.number_input(
            "Overbought Level", 
            min_value=60, 
            max_value=90, 
            value=70, 
            help="RSI level considered overbought (sell signal)",
            key="rsi_overbought_input"
        )
    
    with col3:
        params['rsi_oversold'] = st.number_input(
            "Oversold Level", 
            min_value=10, 
            max_value=40, 
            value=30, 
            help="RSI level considered oversold (buy signal)",
            key="rsi_oversold_input"
        )
    
    if params['rsi_oversold'] >= params['rsi_overbought']:
        st.error("Oversold level must be less than overbought level")
        st.markdown("</div>", unsafe_allow_html=True)
        return {}
    
    # Add some strategy description
    st.markdown(f"""
    <div style="margin-top: 1rem; padding: 1rem; background-color: #f0f4f8; border-radius: 6px; font-size: 0.9rem; color: #4a5568;">
        <strong>Strategy Logic:</strong> Buy when RSI (period={params['rsi_period']}) falls below {params['rsi_oversold']} 
        (oversold) and sell when it rises above {params['rsi_overbought']} (overbought).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    return params


def render_custom_strategy_editor() -> Optional[Dict[str, Any]]:
    st.markdown("<div class='form-group' style='padding: 1.5rem;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h3 style="color: #1e3c72; margin-bottom: 0.5rem;">Custom Strategy Configuration</h3>
        <p style="color: #4a5568; margin: 0;">Define your custom trading strategy using JSON rules</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    with st.expander("Strategy Format Documentation", expanded=False):
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 6px; border-left: 4px solid #1e3c72;">
            <h4 style="color: #1e3c72; margin-top: 0;">Strategy JSON Format</h4>
            <pre style="background-color: #f1f5f9; padding: 1rem; border-radius: 4px; overflow-x: auto;">
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
            </pre>
            
            <div style="margin-top: 1.5rem;">
                <h5 style="color: #1e3c72; margin-bottom: 0.5rem;">Available Indicators</h5>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-bottom: 1rem;">
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">close</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">open</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">high</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">low</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">volume</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">rsi_14</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">sma_20</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">sma_50</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">ema_12</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">ema_26</div>
                </div>
                
                <h5 style="color: #1e3c72; margin-bottom: 0.5rem;">Available Operators</h5>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">></div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;"><</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">>=</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;"><=</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">==</div>
                    <div style="background: white; padding: 0.5rem; border-radius: 4px; font-family: monospace;">!=</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["JSON Editor", "Visual Builder"])
    
    with tab1:
        strategy_json = st.text_area(
            "Strategy JSON", 
            value=json.dumps(default_strategy, indent=2), 
            height=400, 
            help="Enter your strategy as JSON",
            key="strategy_json_editor"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Validate Strategy", use_container_width=True):
                try:
                    strategy_config = json.loads(strategy_json)
                    validation_result = validate_custom_strategy(strategy_config)
                    if validation_result["valid"]:
                        st.session_state.last_valid_strategy = strategy_config
                        st.session_state.strategy_validation = {"valid": True, "message": "Strategy is valid!"}
                    else:
                        st.session_state.strategy_validation = {"valid": False, "message": f"Validation failed: {validation_result['error']}"}
                except json.JSONDecodeError as e:
                    st.session_state.strategy_validation = {"valid": False, "message": f"Invalid JSON: {str(e)}"}
        
        # Show validation result
        if hasattr(st.session_state, 'strategy_validation'):
            if st.session_state.strategy_validation["valid"]:
                st.markdown(f"""
                <div style="background-color: #d4edda; color: #155724; padding: 0.75rem; border-radius: 6px; margin-top: 1rem; display: flex; align-items: center;">
                    <svg style="width: 20px; height: 20px; margin-right: 0.5rem;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    {st.session_state.strategy_validation["message"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8d7da; color: #721c24; padding: 0.75rem; border-radius: 6px; margin-top: 1rem; display: flex; align-items: center;">
                    <svg style="width: 20px; height: 20px; margin-right: 0.5rem;" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                    {st.session_state.strategy_validation["message"]}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if hasattr(st.session_state, 'last_valid_strategy') and st.session_state.last_valid_strategy:
                if st.button("Use Valid Strategy", use_container_width=True):
                    st.session_state.strategy_json = json.dumps(st.session_state.last_valid_strategy, indent=2)
                    st.rerun()
        
        if hasattr(st.session_state, 'last_valid_strategy') and st.session_state.last_valid_strategy:
            strategy_config = st.session_state.last_valid_strategy
            return {"custom_strategy": strategy_config}
        
        return None
    
    with tab2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
            <h3 style="color: #1e3c72; margin: 0 0 1rem 0;">Visual Strategy Builder</h3>
            <p style="color: #6c757d;">We're working on an intuitive visual interface for building trading strategies without writing JSON.</p>
            <p>For now, please use the JSON Editor tab to define your strategy.</p>
        </div>
        """, unsafe_allow_html=True)
        return None
    
    # Close the form group div
    st.markdown("</div>", unsafe_allow_html=True)


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