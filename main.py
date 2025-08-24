import sys
import os
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Shravankumar05/Strategy-Backtester',
        'Report a bug': 'https://github.com/Shravankumar05/Strategy-Backtester/issues',
        'About': """
        # Trading Strategy Backtester
        
        A comprehensive backtesting platform for trading strategies.
        """
    }
)

current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
os.environ['PYTHONPATH'] = str(src_dir)

if __name__ == "__main__":
    try:
        from backtester.ui.app import main
        main()
    except ImportError as e:
        st.error(f"Import Error: {e}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Python path: {sys.path}")
        st.error(f"Src directory: {src_dir}")
        st.error("Please ensure you're running this from the Strategy-Backtester directory")
        sys.exit(1)