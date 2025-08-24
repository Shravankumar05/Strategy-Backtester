import streamlit as st
import time
from typing import Optional, Callable, Any, Dict, List
from contextlib import contextmanager
from ..utils.session_state import get_session_state, set_session_state, update_backtest_status

class ProgressManager:
    @staticmethod
    @contextmanager
    def progress_context(title: str, description: str = "", total_steps: int = 100, show_spinner: bool = True, show_progress_bar: bool = True):
        progress_state = {
            "current_step": 0,
            "total_steps": total_steps,
            "status": "running",
            "message": title
        }
        
        status_container = st.empty()
        progress_container = st.empty()
        message_container = st.empty()
        
        # Apply custom styles
        st.markdown("""
        <style>
        .progress-container {
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #1e3c72;
        }
        .progress-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #1e3c72;
        }
        .progress-message {
            color: #495057;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        try:
            with status_container.container():
                if show_spinner:
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-title">{title}</div>
                        <div class="progress-message">Initializing...</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if show_progress_bar:
                progress_bar = progress_container.progress(0, text=title)
            else:
                progress_bar = None
            
            def update_progress(step: int, message: Optional[str] = None):
                progress_state["current_step"] = step
                if message:
                    progress_state["message"] = message
                
                if progress_bar:
                    progress_value = min(step / total_steps, 1.0)
                    progress_bar.progress(progress_value, text=f"{title}: {int(progress_value * 100)}%")
                
                current_message = progress_state["message"]
                percentage = int((step / total_steps) * 100)
                message_container.markdown(
                    f"<div class='progress-message'>{current_message} ({percentage}%)</div>", 
                    unsafe_allow_html=True
                )
            
            yield update_progress
            progress_state["status"] = "completed"
            
            # Update to success state
            status_container.markdown(f"""
            <div class="progress-container" style="border-left-color: #28a745; background-color: #d4edda;">
                <div class="progress-title">{title} completed successfully</div>
                <div class="progress-message">Operation completed successfully</div>
            </div>
            """, unsafe_allow_html=True)
            
            if progress_bar:
                progress_bar.progress(1.0, text=f"{title}: 100% Complete")
            
            message_container.text("Operation completed (100%)")
            
        except Exception as e:
            progress_state["status"] = "failed"
            status_container.markdown(f"""
            <div class="progress-container" style="border-left-color: #dc3545; background-color: #f8d7da;">
                <div class="progress-title">{title} failed</div>
                <div class="progress-message">Error: {str(e)}</div>
            </div>
            """, unsafe_allow_html=True)
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
        if text:
            st.markdown(f"<div class='progress-message'>{text}</div>", unsafe_allow_html=True)
        return st.progress(value, text=text)
    
    @staticmethod
    def show_step_progress(current_step: int, total_steps: int, step_descriptions: Optional[Dict[int, str]] = None):
        st.markdown(f"<div class='progress-title'>Progress: Step {current_step} of {total_steps}</div>", unsafe_allow_html=True)
        progress_value = (current_step - 1) / total_steps
        st.progress(progress_value)
        
        if step_descriptions:
            for step_num in range(1, total_steps + 1):
                step_class = "completed" if step_num < current_step else ("current" if step_num == current_step else "pending")
                step_icon = "✓" if step_num < current_step else ("→" if step_num == current_step else "○")
                step_text = step_descriptions.get(step_num, f'Step {step_num}')
                
                st.markdown(f"""
                <div class="step-indicator {step_class}">
                    <span style="margin-right: 10px; font-weight: bold;">{step_icon}</span>
                    <span>{step_text}</span>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def show_backtest_progress():
        status = get_session_state("backtest_status", "not_started")
        progress = get_session_state("backtest_progress", 0.0)
        message = get_session_state("backtest_message", "")
        
        if status == "not_started":
            st.markdown("""
            <div class="progress-container" style="background-color: #e2e3e5; border-left-color: #6c757d;">
                <div class="progress-title">Ready to Start</div>
                <div class="progress-message">Configure your backtest and click 'Run Backtest' to begin</div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        elif status == "running":
            st.markdown(f"""
            <div class="progress-container">
                <div class="progress-title">Backtest in Progress</div>
                <div class="progress-message">{message} ({progress:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(progress / 100.0, text=f"{message} ({progress:.1f}%)")
            
            if progress > 0:
                estimated_total_time = 30  # rough estimate
                elapsed_ratio = progress / 100.0
                remaining_time = estimated_total_time * (1 - elapsed_ratio)
                st.markdown(f"<div class='progress-message'>Estimated time remaining: {remaining_time:.0f} seconds</div>", 
                          unsafe_allow_html=True)
        
        elif status == "completed":
            st.markdown("""
            <div class="progress-container" style="border-left-color: #28a745; background-color: #d4edda;">
                <div class="progress-title">Backtest Completed</div>
                <div class="progress-message">Analysis completed successfully</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(1.0, text="Analysis complete (100%)")
            
        elif status == "failed":
            st.markdown(f"""
            <div class="progress-container" style="border-left-color: #dc3545; background-color: #f8d7da;">
                <div class="progress-title">Backtest Failed</div>
                <div class="progress-message">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class="progress-container" style="border-left-color: #ffc107; background-color: #fff3cd;">
                <div class="progress-title">Unknown Status</div>
                <div class="progress-message">Current status: {status}</div>
            </div>
            """, unsafe_allow_html=True)


def simulate_backtest_with_progress():
    steps = ["Fetching historical data", "Validating data quality", "Initializing strategy", "Generating trading signals", "Running simulation", "Calculating performance metrics", "Generating visualizations", "Finalizing results"]
    
    with ProgressManager.progress_context(
        "Running Backtest", 
        description="Processing backtest simulation",
        total_steps=len(steps), 
        show_spinner=True, 
        show_progress_bar=True
    ) as update_progress:
        for i, step_description in enumerate(steps, 1):
            update_progress(i, step_description)
            time.sleep(0.5)  # Simulate processing time
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


def show_validation_progress():
    validation_steps = ["Parsing strategy configuration", "Validating rule syntax",  "Checking indicator availability", "Testing rule logic", "Finalizing strategy"]
    progress_container = st.container()
    with progress_container:
        st.subheader("Validating Strategy")
        for i, step in enumerate(validation_steps, 1):
            progress_value = (i - 1) / len(validation_steps)
            st.progress(progress_value)
            st.text(f"Step {i}/{len(validation_steps)}: {step}")
            time.sleep(0.2)
        
        st.progress(1.0)
        st.success("Strategy validation completed!")


def show_metric_calculation_progress():
    metrics = ["Total Return", "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Win Rate", "Profit Factor", "Volatility", "CAGR"]
    st.subheader("Calculating Performance Metrics")
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
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 4px;
            height: 8px;
        }
        /* Loading spinner styling */
        .stSpinner > div {
            border-top-color: #1e3c72 !important;
            width: 2.5rem;
            height: 2.5rem;
        }
        /* Status message styling */
        .progress-status {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.9rem;
            color: #495057;
            text-align: left;
            margin: 0.5rem 0;
            line-height: 1.5;
        }
        /* Step indicator styling */
        .step-indicator {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            background: #f8f9fa;
            border-left: 3px solid #dee2e6;
            transition: all 0.3s ease;
        }
        .step-indicator.completed {
            background: #f0f9f0;
            border-left-color: #28a745;
            color: #155724;
        }
        .step-indicator.current {
            background: #f0f7ff;
            border-left-color: #1e3c72;
            color: #0a3d6b;
            font-weight: 500;
        }
        .step-indicator.pending {
            background: #f8f9fa;
            color: #6c757d;
            opacity: 0.8;
        }
        /* Progress container styles */
        .progress-container {
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #1e3c72;
        }
        .progress-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #1e3c72;
            font-size: 1.1rem;
        }
        .progress-message {
            color: #495057;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            line-height: 1.5;
        }
        </style>
        """, unsafe_allow_html=True)


def with_progress(func: Callable, *args, **kwargs) -> Any:
    operation_name = kwargs.pop('_progress_title', func.__name__)
    with st.spinner(f"Executing {operation_name}..."):
        return func(*args, **kwargs)


def execute_with_progress(steps: List[str], step_functions: List[Callable], operation_name: str = "Processing") -> List[Any]:
    """Execute a series of steps with a progress indicator"""
    if len(steps) != len(step_functions):
        raise ValueError("Number of steps must match number of step functions")
    
    st.subheader(f"{operation_name}")
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