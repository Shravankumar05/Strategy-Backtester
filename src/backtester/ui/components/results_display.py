import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
from datetime import datetime
from ...visualization.visualizer import Visualizer
from ..utils.session_state import get_session_state

def render_results(results: Dict[str, Any]) -> None:
    if not results or results.get("status") != "completed":
        render_no_results()
        return
    
    render_metrics_overview(results)
    render_detailed_metrics(results)

def render_no_results() -> None:
    st.info("👆 Configure your backtest settings and click 'Run Backtest' to see results here.")
    st.subheader("📊 Results Preview")
    st.write("After running a backtest, you'll see:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", "---", delta="---")
    with col2:
        st.metric("Sharpe Ratio", "---", delta="---")
    with col3:
        st.metric("Max Drawdown", "---", delta="---")
    with col4:
        st.metric("Win Rate", "---", delta="---")
    st.write("📈 Interactive charts and detailed trade analysis will appear here.")

def render_metrics_overview(results: Dict[str, Any]) -> None:
    st.subheader("📊 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_return = results.get("total_return", 0)
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric("Total Return", f"{total_return:.1f}%", delta=f"{total_return:.1f}%", delta_color=delta_color)
    
    with col2:
        sharpe_ratio = results.get("sharpe_ratio", 0)
        delta_text = "Excellent" if sharpe_ratio > 2 else "Good" if sharpe_ratio > 1 else "Poor"
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta=delta_text)
    
    with col3:
        max_drawdown = results.get("max_drawdown", 0)
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%", delta=f"-{max_drawdown:.1f}%", delta_color="inverse")
    
    with col4:
        win_rate = results.get("win_rate", 0)
        delta_color = "normal" if win_rate >= 50 else "inverse"
        st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{win_rate:.1f}%", delta_color=delta_color)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_trades = results.get("total_trades", 0)
        st.metric("Total Trades", f"{total_trades:,}")
    with col2:
        profit_factor = results.get("profit_factor", 0)
        delta_color = "normal" if profit_factor > 1 else "inverse"
        st.metric("Profit Factor", f"{profit_factor:.2f}", delta="Profitable" if profit_factor > 1 else "Unprofitable", delta_color=delta_color)
    with col3:
        avg_trade = results.get("avg_trade_return", 0)
        delta_color = "normal" if avg_trade >= 0 else "inverse"
        st.metric("Avg Trade", f"{avg_trade:.2f}%", delta=f"{avg_trade:.2f}%", delta_color=delta_color)
    with col4:
        volatility = results.get("volatility", 0)
        st.metric("Volatility", f"{volatility:.1f}%")
    
    # Add VaR and CVaR metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        var_95 = results.get("var_95", 0)
        st.metric("Value at Risk (95%)", f"{var_95:.2f}%", delta=f"-{var_95:.2f}%", delta_color="inverse")
    with col2:
        cvar_95 = results.get("cvar_95", 0)
        st.metric("Conditional VaR (95%)", f"{cvar_95:.2f}%", delta=f"-{cvar_95:.2f}%", delta_color="inverse")
    with col3:
        pass  # Empty column for alignment
    with col4:
        pass  # Empty column for alignment

def render_detailed_metrics(results: Dict[str, Any]) -> None:
    with st.expander("📈 Return Metrics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Absolute Returns**")
            st.write(f"• Total Return: {results.get('total_return', 0):.2f}%")
            st.write(f"• CAGR: {results.get('cagr', 0):.2f}%")
            st.write(f"• Best Month: {results.get('best_month', 0):.2f}%")
            st.write(f"• Worst Month: {results.get('worst_month', 0):.2f}%")
        with col2:
            st.write("**Risk-Adjusted Returns**")
            st.write(f"• Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            st.write(f"• Sortino Ratio: {results.get('sortino_ratio', 0):.2f}")
            st.write(f"• Calmar Ratio: {results.get('calmar_ratio', 0):.2f}")
            st.write(f"• Information Ratio: {results.get('information_ratio', 0):.2f}")
    
    with st.expander("⚠️ Risk Metrics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Drawdown Analysis**")
            st.write(f"• Maximum Drawdown: {results.get('max_drawdown', 0):.2f}%")
            st.write(f"• Avg Drawdown: {results.get('avg_drawdown', 0):.2f}%")
            st.write(f"• Max Drawdown Duration: {results.get('max_dd_duration', 0)} days")
            st.write(f"• Recovery Factor: {results.get('recovery_factor', 0):.2f}")
        with col2:
            st.write("**Volatility Analysis**")
            st.write(f"• Annualized Volatility: {results.get('volatility', 0):.2f}%")
            st.write(f"• Downside Deviation: {results.get('downside_deviation', 0):.2f}%")
            st.write(f"• VaR (95%): {results.get('var_95', 0):.2f}%")
            st.write(f"• CVaR (95%): {results.get('cvar_95', 0):.2f}%")
    
    with st.expander("📊 Trade Statistics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Trade Performance**")
            st.write(f"• Total Trades: {results.get('total_trades', 0):,}")
            st.write(f"• Winning Trades: {results.get('winning_trades', 0):,}")
            st.write(f"• Losing Trades: {results.get('losing_trades', 0):,}")
            st.write(f"• Win Rate: {results.get('win_rate', 0):.1f}%")
        with col2:
            st.write("**Trade Analysis**")
            st.write(f"• Profit Factor: {results.get('profit_factor', 0):.2f}")
            st.write(f"• Avg Win: {results.get('avg_win', 0):.2f}%")
            st.write(f"• Avg Loss: {results.get('avg_loss', 0):.2f}%")
            st.write(f"• Largest Win: {results.get('largest_win', 0):.2f}%")
            st.write(f"• Largest Loss: {results.get('largest_loss', 0):.2f}%")
    
    if results.get("leverage", 1.0) > 1.0:
        with st.expander("⚡ Leverage Analysis", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Leverage Usage**")
                st.write(f"• Max Leverage: {results.get('max_leverage_used', 0):.1f}x")
                st.write(f"• Avg Leverage: {results.get('avg_leverage_used', 0):.1f}x")
                st.write(f"• Leverage Utilization: {results.get('leverage_utilization', 0):.1f}%")
            with col2:
                st.write("**Margin Analysis**")
                st.write(f"• Margin Calls: {results.get('margin_calls', 0)}")
                st.write(f"• Min Margin Ratio: {results.get('min_margin_ratio', 0):.2f}")
                st.write(f"• Avg Margin Ratio: {results.get('avg_margin_ratio', 0):.2f}")


def render_performance_summary(results: Dict[str, Any]) -> None:
    score = calculate_performance_score(results)
    grade, color = get_performance_grade(score)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20, {color}10);
        border: 2px solid {color};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    ">
        <h3 style="color: {color}; margin: 0;">Strategy Performance Grade</h3>
        <h1 style="color: {color}; margin: 0.5rem 0; font-size: 3rem;">{grade}</h1>
        <p style="margin: 0; color: #666;">Score: {score:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)


def calculate_performance_score(results: Dict[str, Any]) -> float:
    score = 0
    
    total_return = results.get("total_return", 0)
    if total_return > 20:
        score += 30
    elif total_return > 10:
        score += 25
    elif total_return > 5:
        score += 20
    elif total_return > 0:
        score += 15
    
    sharpe = results.get("sharpe_ratio", 0)
    if sharpe > 2:
        score += 25
    elif sharpe > 1.5:
        score += 20
    elif sharpe > 1:
        score += 15
    elif sharpe > 0.5:
        score += 10
    
    max_dd = results.get("max_drawdown", 100)
    if max_dd < 5:
        score += 25
    elif max_dd < 10:
        score += 20
    elif max_dd < 15:
        score += 15
    elif max_dd < 25:
        score += 10
    
    win_rate = results.get("win_rate", 0)
    if win_rate > 70:
        score += 20
    elif win_rate > 60:
        score += 15
    elif win_rate > 50:
        score += 10
    elif win_rate > 40:
        score += 5

    return min(score, 100)


def get_performance_grade(score: float) -> tuple[str, str]:
    if score >= 90:
        return "A+", "#28a745"
    elif score >= 80:
        return "A", "#28a745"
    elif score >= 70:
        return "B", "#17a2b8"
    elif score >= 60:
        return "C", "#ffc107"
    elif score >= 50:
        return "D", "#fd7e14"
    else:
        return "F", "#dc3545"


def render_comparison_table(results: Dict[str, Any]) -> None:
    st.subheader("📊 Performance Comparison")
    comparison_data = {
        "Metric": ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Volatility (%)", "Profit Factor"],
        "Your Strategy": [f"{results.get('total_return', 0):.1f}", f"{results.get('sharpe_ratio', 0):.2f}", f"{results.get('max_drawdown', 0):.1f}", f"{results.get('win_rate', 0):.1f}", f"{results.get('volatility', 0):.1f}", f"{results.get('profit_factor', 0):.2f}"],
        "Buy & Hold": ["8.5", "0.85", "12.3", "100.0", "16.2", "N/A"],
        "Good Strategy": ["> 15.0", "> 1.5", "< 10.0", "> 60.0", "< 20.0", "> 1.5"]
    }
    
    df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Your Strategy": st.column_config.TextColumn("Your Strategy", width="small"),
            "Buy & Hold": st.column_config.TextColumn("Buy & Hold", width="small"),
            "Good Strategy": st.column_config.TextColumn("Good Strategy", width="small"),
        }
    )

def render_risk_warning(results: Dict[str, Any]) -> None:
    warnings = []
    max_dd = results.get("max_drawdown", 0)
    if max_dd > 20:
        warnings.append(f"⚠️ High maximum drawdown ({max_dd:.1f}%) indicates significant risk")
    sharpe = results.get("sharpe_ratio", 0)
    if sharpe < 0.5:
        warnings.append(f"⚠️ Low Sharpe ratio ({sharpe:.2f}) suggests poor risk-adjusted returns")
    win_rate = results.get("win_rate", 0)
    if win_rate < 40:
        warnings.append(f"⚠️ Low win rate ({win_rate:.1f}%) may indicate inconsistent strategy")
    volatility = results.get("volatility", 0)
    if volatility > 30:
        warnings.append(f"⚠️ High volatility ({volatility:.1f}%) indicates unstable returns")
    if warnings:
        st.warning("**Risk Warnings:**\n\n" + "\n\n".join(warnings))

def format_currency(value: float) -> str:
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    return f"{value:.2f}%"

def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"