import streamlit as st
from ..utils.session_state import get_session_state

def render_sidebar():
    with st.sidebar:
        st.title("🎛️ Quick Controls")
        results = get_session_state("backtest_results", None)
        if results:
            if results.get("status") == "completed":
                st.success("✅ Backtest Complete")
                st.metric("Total Return", f"{results['total_return']:.1f}%")
            else:
                st.warning("⏳ Backtest Running")
        else:
            st.info("🔧 Configure Backtest")
        
        st.markdown("---")
        st.subheader("⚡ Quick Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
        auto_refresh = st.checkbox("Auto-refresh charts", value=False)
        st.subheader("📤 Export")
        if results and results.get("status") == "completed":
            if st.button("📊 Export Results"):
                st.info("Export functionality coming soon!")
            
            if st.button("📈 Export Charts"):
                st.info("Chart export functionality coming soon!")
        else:
            st.info("Run a backtest to enable export options")
        
        st.markdown("---")
        st.subheader("❓ Help")
        with st.expander("Quick Tips"):
            st.markdown("""
            **Getting Started:**
            1. Select a ticker symbol
            2. Choose date range (2024 only)
            3. Configure strategy parameters
            4. Click 'Run Backtest'
            
            **Tips:**
            - Higher leverage = higher risk
            - Consider transaction costs
            - Test multiple strategies
            - Review drawdown carefully
            """)
        
        with st.expander("System Info"):
            st.markdown("""
            **Version:** 1.0.0  
            **Data Source:** Yahoo Finance  
            **Update Frequency:** Daily  
            **Supported Assets:** US Stocks  
            """)