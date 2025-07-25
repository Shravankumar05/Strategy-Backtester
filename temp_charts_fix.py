def render_charts_tab():
    results = get_session_state("backtest_results", None)
    if results is None:
        st.info("Run a backtest first to see charts here.")
        return
    
    if results.get("status") != "completed":
        st.warning("⏳ Backtest is still running or failed. Please check the Configuration tab.")
        return
    
    st.header("📈 Performance Charts")
    equity_curve_df = results.get("equity_curve")
    equity_curve = None
    
    if equity_curve_df is not None:
        if isinstance(equity_curve_df, pd.DataFrame):
            equity_curve = equity_curve_df['equity'] if 'equity' in equity_curve_df.columns else equity_curve_df.iloc[:, 0]
        else:
            equity_curve = equity_curve_df
    
    st.subheader("Equity Curve")
    if equity_curve is not None:
        fig = Visualizer.plot_equity_curve(equity_curve, title="Strategy Performance")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            show_benchmark = st.checkbox("Show Benchmark", value=False)
        
        with col2:
            show_annotations = st.checkbox("Show Annotations", value=True)
        
        with col3:
            chart_height = st.slider("Chart Height", 400, 800, 600)
        
        if show_benchmark or not show_annotations or chart_height != 600:
            fig = Visualizer.plot_equity_curve(
                equity_curve, 
                title="Strategy Performance",
                show_annotations=show_annotations,
                height=chart_height
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No equity curve data available")
    chart_tabs = st.tabs(["📉 Drawdown", "📊 Returns Distribution", "📈 Rolling Metrics"])
    
    with chart_tabs[0]:
        st.subheader("Drawdown Analysis")
        if equity_curve is not None:
            drawdown = calculate_drawdown_series(equity_curve)
            fig_dd = Visualizer.plot_drawdown(drawdown, title="Strategy Drawdown")
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("Drawdown chart will be displayed here after running a backtest")
    
    with chart_tabs[1]:
        st.subheader("Returns Distribution")
        if equity_curve is not None:
            returns = equity_curve.pct_change().dropna()
            fig_dist = Visualizer.plot_returns_distribution(returns, title="Daily Returns Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Returns distribution will be displayed here after running a backtest")
    
    with chart_tabs[2]:
        st.subheader("Rolling Performance Metrics")
        if equity_curve is not None:
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Rolling Window (days)", 10, 60, 30)
            with col2:
                metrics_to_show = st.multiselect(
                    "Metrics to Display",
                    ["returns", "volatility", "sharpe"],
                    default=["returns", "volatility", "sharpe"]
                )
            
            if metrics_to_show:
                fig_rolling = Visualizer.plot_rolling_metrics(
                    equity_curve, 
                    window=window_size,
                    metrics=metrics_to_show,
                    title=f"Rolling Metrics ({window_size}d window)"
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
        else:
            st.info("Rolling metrics will be displayed here after running a backtest")