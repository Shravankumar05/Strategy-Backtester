import streamlit as st
import time
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from ..utils.session_state import get_session_state, set_session_state, update_backtest_status

class ProgressManager:
    @staticmethod
    @contextmanager
    def progress_context(title: str, total_steps: int = 100, show_spinner: bool = True, show_progress_bar: bool = True):
        progress_state = {
            "current_step": 0,
            "total_steps": total_steps,
            "status": "running",
            "message": title
        }
        
        status_container = st.empty()
        progress_container = st.empty()
        message_container = st.empty()
        
        try:
            with status_container.container():
                if show_spinner:
                    st.info(f"🔄 {title}")
            
            if show_progress_bar:
                progress_bar = progress_container.progress(0)
            else:
                progress_bar = None
            
            def update_progress(step: int, message: Optional[str] = None):
                progress_state["current_step"] = step
                if message:
                    progress_state["message"] = message
                
                if progress_bar:
                    progress_value = min(step / total_steps, 1.0)
                    progress_bar.progress(progress_value)
                
                current_message = progress_state["message"]
                percentage = int((step / total_steps) * 100)
                message_container.text(f"{current_message} ({percentage}%)")
            
            yield update_progress
            progress_state["status"] = "completed"
            status_container.success(f"✅ {title} completed successfully!")
            if progress_bar:
                progress_bar.progress(1.0)
            
            message_container.text("Operation completed (100%)")
            
        except Exception as e:
            progress_state["status"] = "failed"
            status_container.error(f"❌ {title} failed: {str(e)}")
            message_container.text(f"Error: {str(e)}")
            raise
        
        finally:
            time.sleep(1)
            status_container.empty()
            progress_container.empty()
            message_container.empty()
    
    @staticmethod
    def show_loading_spinner(message: str = "Loading..."):
        return st.spinner(message)
    
    @staticmethod
    def show_progress_bar(value: float, text: Optional[str] = None, format_string: str = "%d%%"):
        progress_bar = st.progress(value)
        if text:
            st.text(text)
        
        return progress_bar
    
    @staticmethod
    def show_step_progress(current_step: int, total_steps: int, step_descriptions: Optional[Dict[int, str]] = None):
        st.subheader(f"Progress: Step {current_step} of {total_steps}")
        progress_value = (current_step - 1) / total_steps
        st.progress(progress_value)
        if step_descriptions:
            for step_num in range(1, total_steps + 1):
                if step_num < current_step:
                    st.write(f"✅ Step {step_num}: {step_descriptions.get(step_num, f'Step {step_num}')}")
                elif step_num == current_step:
                    st.write(f"🔄 Step {step_num}: {step_descriptions.get(step_num, f'Step {step_num}')} (In Progress)")
                else:
                    st.write(f"⏳ Step {step_num}: {step_descriptions.get(step_num, f'Step {step_num}')}")
    
    @staticmethod
    def show_backtest_progress():
        status = get_session_state("backtest_status", "not_started")
        progress = get_session_state("backtest_progress", 0.0)
        message = get_session_state("backtest_message", "")
        
        if status == "not_started":
            st.info("🔧 Configure your backtest and click 'Run Backtest' to begin")
            return
        
        elif status == "running":
            st.info("🔄 Backtest in progress...")
            progress_bar = st.progress(progress / 100.0)
            st.text(f"{message} ({progress:.1f}%)")
            if progress > 0:
                estimated_total_time = 30 # rough time
                elapsed_ratio = progress / 100.0
                remaining_time = estimated_total_time * (1 - elapsed_ratio)
                st.text(f"Estimated time remaining: {remaining_time:.0f} seconds") # Botched progress bar method lol
        
        elif status == "completed":
            st.success("✅ Backtest completed successfully!")
            st.progress(1.0)
            
        elif status == "failed":
            st.error(f"❌ Backtest failed: {message}")
        
        else:
            st.warning(f"⚠️ Unknown status: {status}")


def simulate_backtest_with_progress():
    steps = ["Fetching historical data", "Validating data quality", "Initializing strategy", "Generating trading signals", "Running simulation", "Calculating performance metrics", "Generating visualizations", "Finalizing results"]
    
    with ProgressManager.progress_context("Running Backtest", total_steps=len(steps), show_spinner=True, show_progress_bar=True
    ) as update_progress:
        for i, step_description in enumerate(steps, 1):
            update_progress(i, step_description)
            time.sleep(0.5)  # Simulate processing time lol
            progress_percentage = (i / len(steps)) * 100
            update_backtest_status("running", progress_percentage, step_description)


def show_data_loading_progress(ticker: str, start_date, end_date):
    with st.spinner(f"Fetching data for {ticker} from {start_date} to {end_date}..."):
        steps = ["Connecting to data source", "Requesting historical data", "Processing OHLCV data", "Validating data quality", "Caching results"]
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(steps, 1):
            progress_value = i / len(steps)
            progress_bar.progress(progress_value)
            status_text.text(f"{step}... ({int(progress_value * 100)}%)")
            time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()


def show_strategy_validation_progress(strategy_config: Dict[str, Any]):
    validation_steps = ["Parsing strategy configuration", "Validating rule syntax",  "Checking indicator availability", "Testing rule logic", "Finalizing strategy"]
    progress_container = st.container()
    with progress_container:
        st.subheader("🔍 Validating Strategy")
        for i, step in enumerate(validation_steps, 1):
            progress_value = (i - 1) / len(validation_steps)
            st.progress(progress_value)
            st.text(f"Step {i}/{len(validation_steps)}: {step}")
            time.sleep(0.2)
            if i == len(validation_steps):
                st.progress(1.0)
                st.success("✅ Strategy validation completed!")


def show_metric_calculation_progress():
    metrics = ["Total Return", "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Win Rate", "Profit Factor", "Volatility", "CAGR"]
    st.subheader("📊 Calculating Performance Metrics")
    progress_bar = st.progress(0)
    metric_status = st.empty()
    
    for i, metric in enumerate(metrics, 1):
        progress_value = i / len(metrics)
        progress_bar.progress(progress_value)
        metric_status.text(f"Calculating {metric}... ({int(progress_value * 100)}%)")
        time.sleep(0.1)
    
    progress_bar.progress(1.0)
    metric_status.text("All metrics calculated successfully!")
    time.sleep(0.5)
    progress_bar.empty()
    metric_status.empty()


class ProgressIndicatorStyles:
    @staticmethod
    def apply_custom_progress_styles():
        st.markdown("""
        <style>
        /* Custom progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #1f77b4 0%, #17a2b8 100%);
        }
        /* Loading spinner styling */
        .stSpinner > div {
            border-top-color: #1f77b4 !important;
        }
        /* Status message styling */
        .progress-status {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #666;
            text-align: center;
            margin: 0.5rem 0;
        }
        /* Step indicator styling */
        .step-indicator {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
            padding: 0.5rem;
            border-radius: 5px;
            background: #f8f9fa;
        }
        .step-indicator.completed {
            background: #d4edda;
            color: #155724;
        }
        .step-indicator.current {
            background: #d1ecf1;
            color: #0c5460;
        }
        .step-indicator.pending {
            background: #f8f9fa;
            color: #6c757d;
        }
        </style>
        """, unsafe_allow_html=True)


def with_progress(func: Callable, *args, **kwargs) -> Any:
    operation_name = kwargs.pop('_progress_title', func.__name__)
    with st.spinner(f"Executing {operation_name}..."):
        return func(*args, **kwargs)


def show_operation_progress(operation_name: str, steps: list, step_functions: list, show_details: bool = True):
    if len(steps) != len(step_functions):
        raise ValueError("Number of steps must match number of step functions")
    
    st.subheader(f"🔄 {operation_name}")
    overall_progress = st.progress(0)
    current_step_text = st.empty()
    results = []
    try:
        for i, (step_desc, step_func) in enumerate(zip(steps, step_functions), 1):
            progress_value = (i - 1) / len(steps)
            overall_progress.progress(progress_value)
            current_step_text.text(f"Step {i}/{len(steps)}: {step_desc}")
            
            if show_details:
                with st.expander(f"Details: {step_desc}", expanded=False):
                    result = step_func()
                    results.append(result)
            else:
                result = step_func()
                results.append(result)
        
        overall_progress.progress(1.0)
        current_step_text.text(f"✅ {operation_name} completed successfully!")
        return results
        
    except Exception as e:
        current_step_text.text(f"❌ {operation_name} failed: {str(e)}")
        raise