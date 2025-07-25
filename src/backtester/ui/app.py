import streamlit as st
import pandas as pd
import numpy as np
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
    from backtester.simulation.engine import SimulationEngine
    from backtester.simulation.config import SimulationConfig
    from backtester.ui.components.progress_indicators import ProgressManager
    from backtester.ui.utils.session_state import initialize_session_state, get_session_state, set_session_state
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please ensure all required modules are properly installed and the application is run from the correct directory.")
    st.stop()

def configure_page():
    st.set_page_config(
        page_title="Trading Strategy Backtester",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Shravankumar05/Strategy-Backtester',
            'Report a bug': 'https://github.com/Shravankumar05/Strategy-Backtester/issues',
            'About': """
            # Trading Strategy Backtester
            
            A comprehensive backtesting platform for trading strategies.
            
            **Features:**
            - Historical data analysis
            - Multiple strategy types
            - Comprehensive performance metrics
            - Interactive visualizations
            - Risk analysis tools
            
            Built with Python, Streamlit, and Plotly.
            """
        }
    )
    
    from backtester.ui.components.responsive_layout import apply_global_responsive_styles
    apply_global_responsive_styles()
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #1f77b4;
    }
    
    .results-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="main-header">
        📈 Trading Strategy Backtester
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6c757d; margin-bottom: 2rem;">
            Analyze and optimize your trading strategies with comprehensive backtesting
        </div>
        """, unsafe_allow_html=True)

def render_navigation():
    tabs = st.tabs([
        "🔧 Configuration", 
        "📊 Results", 
        "📈 Charts", 
        "📋 Trade Log",
        "ℹ️ About"
    ])
    return tabs

def render_configuration_tab():
    st.header("Backtest Configuration")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📅 Data Settings")
        render_data_settings()
        
        st.subheader("💰 Capital Settings")
        render_capital_settings()
    
    with col2:
        st.subheader("📈 Strategy Settings")
        render_strategy_settings()
        
        st.subheader("⚙️ Execution Settings")
        render_execution_settings()
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            run_backtest()

def render_data_settings():
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        ticker = st.text_input(
            "Stock Ticker",
            value=get_session_state("ticker", "AAPL"),
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        )
        set_session_state("ticker", ticker)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=get_session_state("start_date", date(2020, 1, 1)),
                min_value=date(2000, 1, 1),
                max_value=date.today(),
                help="Select the start date for backtesting"
            )
            set_session_state("start_date", start_date)
        with col2:
            end_date = st.date_input(
                "End Date",
                value=get_session_state("end_date", date(2023, 12, 31)),
                min_value=date(2000, 1, 1),
                max_value=date.today(),
                help="Select the end date for backtesting"
            )
            set_session_state("end_date", end_date)
        if start_date >= end_date:
            st.error("Start date must be before end date")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_capital_settings():
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=get_session_state("initial_capital", 10000.0),
            step=1000.0,
            help="Starting capital for the backtest"
        )
        set_session_state("initial_capital", initial_capital)
        
        leverage = st.slider(
            "Leverage",
            min_value=1.0,
            max_value=5.0,
            value=get_session_state("leverage", 1.0),
            step=0.1,
            help="Maximum leverage to use (1.0 = no leverage)"
        )
        set_session_state("leverage", leverage)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_strategy_settings():
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        strategy_type = st.selectbox(
            "Strategy Type",
            options=["Moving Average Crossover", "RSI Strategy", "Buy and Hold"],
            index=get_session_state("strategy_index", 0),
            help="Select the trading strategy to backtest"
        )
        set_session_state("strategy_type", strategy_type)
        set_session_state("strategy_index", ["Moving Average Crossover", "RSI Strategy", "Buy and Hold"].index(strategy_type))
        
        if strategy_type == "Moving Average Crossover":
            col1, col2 = st.columns(2)
            with col1:
                short_window = st.number_input(
                    "Short MA Period",
                    min_value=5,
                    max_value=50,
                    value=get_session_state("short_window", 20),
                    help="Short moving average period"
                )
                set_session_state("short_window", short_window)
            
            with col2:
                long_window = st.number_input(
                    "Long MA Period",
                    min_value=20,
                    max_value=200,
                    value=get_session_state("long_window", 50),
                    help="Long moving average period"
                )
                set_session_state("long_window", long_window)
            
            if short_window >= long_window:
                st.error("Short MA period must be less than Long MA period")
        
        elif strategy_type == "RSI Strategy":
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.number_input(
                    "RSI Period",
                    min_value=5,
                    max_value=30,
                    value=get_session_state("rsi_period", 14),
                    help="RSI calculation period"
                )
                set_session_state("rsi_period", rsi_period)
            
            with col2:
                rsi_overbought = st.number_input(
                    "Overbought Level",
                    min_value=60,
                    max_value=90,
                    value=get_session_state("rsi_overbought", 70),
                    help="RSI overbought threshold"
                )
                set_session_state("rsi_overbought", rsi_overbought)
            
            with col3:
                rsi_oversold = st.number_input(
                    "Oversold Level",
                    min_value=10,
                    max_value=40,
                    value=get_session_state("rsi_oversold", 30),
                    help="RSI oversold threshold"
                )
                set_session_state("rsi_oversold", rsi_oversold)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_execution_settings():
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        transaction_cost = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=get_session_state("transaction_cost", 0.1),
            step=0.01,
            format="%.3f",
            help="Transaction cost as percentage of trade value"
        ) / 100
        set_session_state("transaction_cost", transaction_cost)
        
        slippage = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=0.5,
            value=get_session_state("slippage", 0.05),
            step=0.01,
            format="%.3f",
            help="Price slippage as percentage of trade value"
        ) / 100
        set_session_state("slippage", slippage)
        
        position_sizing = st.selectbox(
            "Position Sizing",
            options=["Fixed Fraction", "Fixed Size"],
            index=get_session_state("position_sizing_index", 0),
            help="Method for determining position sizes"
        )
        set_session_state("position_sizing", position_sizing)
        set_session_state("position_sizing_index", ["Fixed Fraction", "Fixed Size"].index(position_sizing))
        
        if position_sizing == "Fixed Fraction":
            position_size = st.slider(
                "Position Size (%)",
                min_value=1,
                max_value=100,
                value=get_session_state("position_size_pct", 10),
                help="Percentage of capital to use per trade"
            ) / 100
            set_session_state("position_size", position_size)
            set_session_state("position_size_pct", int(position_size * 100))
        else:
            initial_capital = get_session_state("initial_capital", 10000.0)
            position_size = st.number_input(
                "Position Size ($)",
                min_value=100.0,
                max_value=float(initial_capital),
                value=get_session_state("position_size_fixed", min(1000.0, initial_capital)),
                step=100.0,
                help="Fixed dollar amount per trade"
            )
            set_session_state("position_size", position_size)
            set_session_state("position_size_fixed", position_size)
        
        st.markdown('</div>', unsafe_allow_html=True)

def run_backtest():
    try:
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
        
        with ProgressManager.progress_context(
            "Running Backtest",
            total_steps=len(steps),
            show_spinner=True,
            show_progress_bar=True
        ) as update_progress:
            update_progress(1, "Validating configuration")
            config_valid, config_error = validate_backtest_config()
            if not config_valid:
                raise ValueError(f"Configuration error: {config_error}")
            
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
            
            update_progress(3, "Initializing strategy")
            strategy = create_strategy_from_config()
            update_progress(4, "Generating trading signals")
            try:
                signals = strategy.generate_signals(data)
                if signals.empty:
                    raise ValueError("Strategy generated no trading signals")
            except Exception as e:
                raise ValueError(f"Failed to generate signals: {str(e)}")
            
            update_progress(5, "Running simulation")
            simulation_config = create_simulation_config()
            simulation_engine = SimulationEngine(simulation_config)
            try:
                simulation_result = simulation_engine.run_simulation(data, signals)
            except Exception as e:
                raise ValueError(f"Simulation failed: {str(e)}")
            
            update_progress(6, "Calculating performance metrics")
            try:
                equity_curve_df = simulation_result.equity_curve
                if isinstance(equity_curve_df, pd.DataFrame):
                    equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
                else:
                    equity_series = equity_curve_df
                    
                metrics = PerformanceMetrics.calculate_all_metrics(
                    equity_series,
                    simulation_result.trades
                )
            except Exception as e:
                raise ValueError(f"Failed to calculate metrics: {str(e)}")
            
            update_progress(7, "Generating visualizations")
            update_progress(8, "Finalizing results")
            backtest_results = {
                "equity_curve": simulation_result.equity_curve,
                "trades": simulation_result.trades,
                "raw_data": data,
                "signals": signals,
                "metrics": metrics,
                "config": {
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "strategy_type": get_session_state("strategy_type"),
                    "strategy_params": strategy.get_parameters(),
                    "simulation_config": simulation_config.__dict__
                },
                "status": "completed",
                "timestamp": datetime.now()
            }
            
            set_session_state("backtest_results", backtest_results)
        
        st.success("✅ Backtest completed successfully!")
        st.balloons()
        
    except Exception as e:
        error_msg = f"❌ Backtest failed: {str(e)}"
        st.error(error_msg)
        st.error("**Debug Information:**")
        st.code(traceback.format_exc())
        set_session_state("backtest_results", None)


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
    results = get_session_state("backtest_results", None)
    
    if results is None:
        st.info("👆 Configure your backtest settings and click 'Run Backtest' to see results here.")
        return
    
    if results.get("status") != "completed":
        st.warning("⏳ Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    st.header("📊 Backtest Results")
    
    metrics = results.get("metrics", {})
    equity_curve = results.get("equity_curve")
    trades = results.get("trades", [])
    config = results.get("config", {})
    
    if metrics.get('final_liquidation_size'):
        size = metrics['final_liquidation_size']
        price = metrics['final_liquidation_price']
        value = metrics['final_liquidation_value']
        st.info(f"🔄 Final Position Liquidation: {size:.2f} shares @ ${price:.2f} " + 
                f"(Value: ${value:.2f})", icon="ℹ️")
    
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
    else:
        total_return = 0.0
    
    raw_data = results.get("raw_data")
    benchmark_return = 0.0
    if raw_data is not None and len(raw_data) > 0:
        initial_price = raw_data['Close'].iloc[0]
        final_price = raw_data['Close'].iloc[-1]
        
        if isinstance(initial_price, pd.Series):
            initial_price = initial_price.iloc[0] if len(initial_price) == 1 else initial_price.values[0]
        if isinstance(final_price, pd.Series):
            final_price = final_price.iloc[0] if len(final_price) == 1 else final_price.values[0]
            
        benchmark_return = ((final_price / initial_price) - 1) * 100
    
    outperformance = total_return - benchmark_return
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Strategy Return",
            f"{total_return:.1f}%",
            delta=f"{total_return:.1f}%"
        )
    
    with col2:
        st.metric(
            "Buy & Hold Return",
            f"{benchmark_return:.1f}%",
            delta=f"{benchmark_return:.1f}%"
        )
    
    with col3:
        st.metric(
            "Outperformance",
            f"{outperformance:.1f}%",
            delta=f"{outperformance:.1f}%",
            delta_color="normal" if outperformance >= 0 else "inverse"
        )
    
    with col4:
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio:.2f}",
            delta="Good" if sharpe_ratio > 1.0 else "Poor"
        )
    
    with col5:
        max_drawdown = metrics.get("max_drawdown", 0.0) * 100  # Convert to percentage
        st.metric(
            "Max Drawdown",
            f"{max_drawdown:.1f}%",
            delta=f"-{max_drawdown:.1f}%"
        )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        win_rate = metrics.get("win_rate", 0.0) * 100  # Convert to percentage
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{win_rate:.1f}%"
        )
    
    with st.expander("📈 Detailed Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Return Metrics**")
            st.write(f"• Total Return: {total_return:.2f}%")
            cagr = metrics.get("cagr", 0.0) * 100
            st.write(f"• CAGR: {cagr:.2f}%")
            volatility = metrics.get("volatility", 0.0) * 100
            st.write(f"• Volatility: {volatility:.2f}%")
        
        with col2:
            st.write("**Risk Metrics**")
            st.write(f"• Sharpe Ratio: {sharpe_ratio:.2f}")
            sortino_ratio = metrics.get("sortino_ratio", 0.0)
            st.write(f"• Sortino Ratio: {sortino_ratio:.2f}")
            st.write(f"• Max Drawdown: {max_drawdown:.2f}%")
        
        # Trade statistics
        st.write("**Trade Statistics**")
        col3, col4 = st.columns(2)
        
        with col3:
            total_trades = len(trades)
            st.write(f"• Total Trades: {total_trades}")
            profit_factor = metrics.get("profit_factor", 0.0)
            st.write(f"• Profit Factor: {profit_factor:.2f}")
        
        with col4:
            avg_trade_return = metrics.get("avg_trade_return", 0.0) * 100
            st.write(f"• Avg Trade Return: {avg_trade_return:.2f}%")
            max_consecutive_losses = metrics.get("max_consecutive_losses", 0)
            st.write(f"• Max Consecutive Losses: {max_consecutive_losses}")
    
    with st.expander("⚙️ Backtest Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Settings**")
            st.write(f"• Ticker: {config.get('ticker', 'N/A')}")
            st.write(f"• Start Date: {config.get('start_date', 'N/A')}")
            st.write(f"• End Date: {config.get('end_date', 'N/A')}")
        
        with col2:
            st.write("**Strategy Settings**")
            st.write(f"• Strategy: {config.get('strategy_type', 'N/A')}")
            strategy_params = config.get('strategy_params', {})
            for param, value in strategy_params.items():
                st.write(f"• {param}: {value}")
        
        st.write("**Simulation Settings**")
        sim_config = config.get('simulation_config', {})
        col3, col4 = st.columns(2)
        
        with col3:
            st.write(f"• Initial Capital: ${sim_config.get('initial_capital', 0):,.2f}")
            st.write(f"• Leverage: {sim_config.get('leverage', 1.0):.1f}x")
        
        with col4:
            st.write(f"• Transaction Cost: {sim_config.get('transaction_cost', 0.0)*100:.3f}%")
            st.write(f"• Slippage: {sim_config.get('slippage', 0.0)*100:.3f}%")

def render_charts_tab():
    results = get_session_state("backtest_results", None)
    
    if results is None:
        st.info("👆 Run a backtest first to see charts here.")
        return
    
    if results.get("status") != "completed":
        st.warning("⏳ Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    st.header("📈 Performance Charts")
    
    st.subheader("Equity Curve")
    equity_curve_df = results.get("equity_curve")
    if equity_curve_df is not None:
        if isinstance(equity_curve_df, pd.DataFrame):
            equity_curve = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
        else:
            equity_curve = equity_curve_df
            
        col1, col2, col3 = st.columns(3)
        with col1:
            show_benchmark = st.checkbox("Show Benchmark", value=False, key="benchmark_checkbox")
        with col2:
            show_annotations = st.checkbox("Show Annotations", value=True, key="annotations_checkbox")
        with col3:
            chart_height = st.slider("Chart Height", 400, 800, 600, key="height_slider")
        
        fig = Visualizer.plot_equity_curve(
            equity_curve, 
            title="Strategy Performance",
            show_annotations=show_annotations,
            height=chart_height
        )
        st.plotly_chart(fig, use_container_width=True, key="equity_curve_chart")
    else:
        st.error("No equity curve data available")
    
    st.subheader("📉 Drawdown Analysis")
    if equity_curve_df is not None:
        if isinstance(equity_curve_df, pd.DataFrame):
            equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
        else:
            equity_series = equity_curve_df
            
        drawdown = calculate_drawdown_series(equity_series)
        fig_dd = Visualizer.plot_drawdown(drawdown, title="Strategy Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True, key="drawdown_chart")
    else:
        st.info("Drawdown chart will be displayed here after running a backtest")
    
    st.subheader("📊 Returns Distribution")
    if equity_curve_df is not None:
        if isinstance(equity_curve_df, pd.DataFrame):
            equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
        else:
            equity_series = equity_curve_df
            
        returns = equity_series.pct_change().dropna()
        fig_dist = Visualizer.plot_returns_distribution(returns, title="Daily Returns Distribution")
        st.plotly_chart(fig_dist, use_container_width=True, key="returns_dist")
    else:
        st.info("Returns distribution will be displayed here after running a backtest")
    
    st.subheader("📈 Rolling Performance Metrics")
    if equity_curve_df is not None:
        if isinstance(equity_curve_df, pd.DataFrame):
            equity_series = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
        else:
            equity_series = equity_curve_df
            
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider("Rolling Window (days)", 10, 60, 30, key="window_size_slider")
        with col2:
            metrics_to_show = st.multiselect(
                "Metrics to Display",
                ["returns", "volatility", "sharpe"],
                default=["returns", "volatility", "sharpe"],
                key="metrics_multiselect"
            )
        
        if metrics_to_show:
            fig_rolling = Visualizer.plot_rolling_metrics(
                equity_series, 
                window=window_size,
                metrics=metrics_to_show,
                title=f"Rolling Metrics ({window_size}d window)"
            )
            st.plotly_chart(fig_rolling, use_container_width=True, key="rolling_metrics")
    else:
        st.info("Rolling metrics will be displayed here after running a backtest")


def calculate_drawdown_series(equity_curve):
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return -drawdown

def render_trade_log_tab():
    results = get_session_state("backtest_results", None)
    if results is None:
        st.info("👆 Run a backtest first to see trade details here.")
        return
    
    if results.get("status") != "completed":
        st.warning("⏳ Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    st.header("📋 Trade Log")
    trades = results.get('trades', [])
    total_trades = len(trades)
    
    if total_trades > 0:
        trade_data = []
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
                if hasattr(trade_type, 'value'):  # Handle Enum values
                    trade_type = trade_type.value
                
                duration = getattr(trade, 'duration', None)
                duration = int(duration) if duration is not None else 1
                if abs(actual_return_pct) < 0.0001:  # Small epsilon to handle floats
                    continue
                    
                trade_dict = {
                    'Date': getattr(trade, 'timestamp', None),
                    'Type': trade_type,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Size': float(getattr(trade, 'size', 0.0)),
                    'Value': value,
                    'Return (%)': actual_return_pct,
                    'Duration (days)': duration,
                    'Commission': float(getattr(trade, 'commission', 0.0))
                }
                missing_fields = []
                invalid_fields = []
                
                if trade_dict['Date'] is None:
                    missing_fields.append('Date')
                if trade_dict['Entry Price'] <= 0:
                    invalid_fields.append('Entry Price')
                if abs(trade_dict['Size']) < 0.000001:
                    invalid_fields.append('Size')    
                if not missing_fields and not invalid_fields:
                    trade_data.append(trade_dict)
                else:
                    error_msg = []
                    if missing_fields:
                        error_msg.append(f"Missing fields: {', '.join(missing_fields)}")
                    if invalid_fields:
                        error_msg.append(f"Invalid values for: {', '.join(invalid_fields)}")
                    st.error(f"Trade validation failed: {' | '.join(error_msg)}")
            except Exception as e:
                st.error(f"Error processing trade: {str(e)}\nTrade data: {str(trade_dict)}")
                continue
        
        
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
                avg_win = np.mean([r for r in winning_returns if abs(r) <= 100])  # Cap at 100%
                st.metric("Avg Win", f"{avg_win:.2f}%")
            else:
                st.metric("Avg Win", "0.00%")
        
        with col4:
            losing_returns = [t['Return (%)'] for t in trade_data if t['Return (%)'] < 0]
            if losing_returns:
                avg_loss = np.mean([r for r in losing_returns if abs(r) <= 100])  # Cap at 100%
                st.metric("Avg Loss", f"{avg_loss:.2f}%")
            else:
                st.metric("Avg Loss", "0.00%")
        
        st.subheader("🔍 Filter Trades")
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
        
        st.subheader(f"📊 Trade Details ({len(filtered_data)} trades)")
        
        if filtered_data:
            df = pd.DataFrame(filtered_data)
            def style_returns(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            
            styled_df = df.style.applymap(style_returns, subset=['Return (%)'])
            
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
    st.header("ℹ️ About Trading Strategy Backtester")
    st.markdown("""
    ## Overview
    
    This application provides a comprehensive platform for backtesting trading strategies
    using historical market data. It's designed to help traders and analysts evaluate
    the performance of their strategies before risking real capital.
    
    ## Features
    
    ### 📊 **Strategy Testing**
    - Multiple pre-built strategies (Moving Average, RSI, Buy & Hold)
    - Custom strategy definition capabilities
    - Configurable parameters for each strategy
    
    ### 💰 **Risk Management**
    - Leverage control and margin modeling
    - Transaction costs and slippage simulation
    - Position sizing options (fixed fraction or fixed size)
    
    ### 📈 **Performance Analysis**
    - Comprehensive performance metrics
    - Risk-adjusted returns (Sharpe, Sortino ratios)
    - Drawdown analysis and visualization
    - Trade-by-trade analysis
    
    ### 🎨 **Visualization**
    - Interactive equity curve charts
    - Drawdown visualization
    - Returns distribution analysis
    - Rolling performance metrics
    
    ## How to Use
    
    1. **Configure**: Set up your backtest parameters in the Configuration tab
    2. **Run**: Execute the backtest with your chosen settings
    3. **Analyze**: Review results, charts, and trade details
    4. **Iterate**: Adjust parameters and re-run to optimize your strategy
    
    ## Technical Details
    
    - **Data Source**: Historical OHLCV data (2024 only in this version)
    - **Simulation Engine**: Event-driven backtesting with realistic execution modeling
    - **Performance Metrics**: Industry-standard risk and return calculations
    - **Visualization**: Interactive charts powered by Plotly
    
    ## Disclaimer
    
    ⚠️ **Important**: This tool is for educational and research purposes only. 
    Past performance does not guarantee future results. Always conduct thorough 
    analysis and consider consulting with financial professionals before making 
    investment decisions.
    """)

def main():
    initialize_session_state()
    configure_page()
    render_header()
    tabs = render_navigation()
    
    with tabs[0]:  # Config
        render_configuration_tab()
    
    with tabs[1]:  # Results
        render_results_tab()
    
    with tabs[2]:  # Charts
        render_charts_tab()
    
    with tabs[3]:  # Trade Log
        render_trade_log_tab()
    
    with tabs[4]:  # About
        render_about_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        Trading Strategy Backtester v1.0 | https://shravankumar05.github.io/
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()