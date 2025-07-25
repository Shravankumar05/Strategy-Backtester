import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from datetime import datetime
import warnings

class Visualizer:
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'dark': '#343a40',
        'light': '#f8f9fa',
        'background': '#ffffff',
        'grid': '#e6e6e6'
    }
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve", show_annotations: bool = True, benchmark: Optional[pd.Series] = None, key_events: Optional[List[Dict]] = None, width: int = 1000, height: int = 600) -> go.Figure:
        if len(equity_curve) == 0:
            fig = go.Figure()
            fig.update_layout(title="No Data Available", width=width, height=height)
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Portfolio Value', line=dict(color=Visualizer.COLORS['primary'], width=2), hovertemplate='<b>Date:</b> %{x}<br>' + '<b>Portfolio Value:</b> $%{y:,.2f}<br>' + '<extra></extra>'))
        
        if benchmark is not None and len(benchmark) > 0:
            fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, mode='lines', name='Benchmark', line=dict(color=Visualizer.COLORS['secondary'], width=2, dash='dash'), hovertemplate='<b>Date:</b> %{x}<br>' + '<b>Benchmark Value:</b> $%{y:,.2f}<br>' + '<extra></extra>'))
        
        if show_annotations and key_events:
            for event in key_events:
                if 'date' in event and 'text' in event:
                    event_date = event['date']
                    event_text = event['text']
                    event_value = event.get('value', None)
                    
                    if event_value is None and event_date in equity_curve.index:
                        event_value = equity_curve.loc[event_date]
                    elif event_value is None:
                        closest_idx = equity_curve.index.get_indexer([event_date], method='nearest')[0]
                        event_value = equity_curve.iloc[closest_idx]
                    
                    fig.add_annotation(x=event_date, y=event_value, text=event_text, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=Visualizer.COLORS['danger'], bgcolor=Visualizer.COLORS['light'], bordercolor=Visualizer.COLORS['danger'], borderwidth=1, font=dict(size=10))
        
        if len(equity_curve) >= 2:
            initial_value = equity_curve.iloc[0]
            final_value = equity_curve.iloc[-1]
            
            if isinstance(initial_value, pd.Series):
                initial_value = initial_value.iloc[0] if len(initial_value) == 1 else initial_value.values[0]
            if isinstance(final_value, pd.Series):
                final_value = final_value.iloc[0] if len(final_value) == 1 else final_value.values[0]
                
            total_return = (final_value / initial_value - 1) * 100
            returns = equity_curve.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            if isinstance(volatility, pd.Series):
                volatility = volatility.iloc[0] if len(volatility) == 1 else volatility.values[0]
            
            subtitle = f"Total Return: {total_return:.1f}% | Volatility: {volatility:.1f}%"
        else:
            subtitle = "Insufficient data for statistics"
        
        fig.update_layout(title=dict(text=f"<b>{title}</b><br><sub>{subtitle}</sub>", x=0.5, font=dict(size=16)),
            xaxis=dict(title="Date", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid'], showline=True, linewidth=1, linecolor=Visualizer.COLORS['dark']),
            yaxis=dict(title="Portfolio Value ($)", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid'], showline=True, linewidth=1, linecolor=Visualizer.COLORS['dark'], tickformat='$,.0f'),
            plot_bgcolor=Visualizer.COLORS['background'],
            paper_bgcolor=Visualizer.COLORS['background'],
            width=width,
            height=height,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)", bordercolor=Visualizer.COLORS['dark'], borderwidth=1),
            margin=dict(l=60, r=30, t=80, b=60)
        )
        
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all", label="All")])
                ),
                rangeslider=dict(visible=False), type="date"
            )
        )
        return fig
    
    @staticmethod
    def create_equity_curve_with_trades(equity_curve: pd.Series, trades: List, title: str = "Equity Curve with Trade Markers") -> go.Figure:
        fig = Visualizer.plot_equity_curve(equity_curve, title=title, show_annotations=False)
        
        if not trades:
            return fig
        buy_trades = [t for t in trades if hasattr(t, 'type') and t.type.lower() == 'buy']
        sell_trades = [t for t in trades if hasattr(t, 'type') and t.type.lower() == 'sell']
        
        if buy_trades:
            buy_dates = [t.timestamp for t in buy_trades]
            buy_values = []
            
            for trade in buy_trades:
                if trade.timestamp in equity_curve.index:
                    buy_values.append(equity_curve.loc[trade.timestamp])
                else:
                    closest_idx = equity_curve.index.get_indexer([trade.timestamp], method='nearest')[0]
                    buy_values.append(equity_curve.iloc[closest_idx])
            
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_values, mode='markers', name='Buy Trades',
                marker=dict(symbol='triangle-up', size=10, color=Visualizer.COLORS['success'], line=dict(width=1, color='white')),
                hovertemplate='<b>Buy Trade</b><br>' + '<b>Date:</b> %{x}<br>' + '<b>Portfolio Value:</b> $%{y:,.2f}<br>' + '<extra></extra>'))
        
        if sell_trades:
            sell_dates = [t.timestamp for t in sell_trades]
            sell_values = []
            
            for trade in sell_trades:
                if trade.timestamp in equity_curve.index:
                    sell_values.append(equity_curve.loc[trade.timestamp])
                else:
                    closest_idx = equity_curve.index.get_indexer([trade.timestamp], method='nearest')[0]
                    sell_values.append(equity_curve.iloc[closest_idx])
            
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_values, mode='markers', name='Sell Trades',
                marker=dict(symbol='triangle-down', size=10, color=Visualizer.COLORS['danger'], line=dict(width=1, color='white')),
                hovertemplate='<b>Sell Trade</b><br>' + '<b>Date:</b> %{x}<br>' + '<b>Portfolio Value:</b> $%{y:,.2f}<br>' + '<extra></extra>'))
        return fig
    
    @staticmethod
    def create_multi_strategy_comparison(equity_curves: Dict[str, pd.Series], title: str = "Strategy Comparison") -> go.Figure:
        if not equity_curves:
            fig = go.Figure()
            fig.update_layout(title="No Data Available")
            return fig
        
        fig = go.Figure()
        
        colors = [
            Visualizer.COLORS['primary'],
            Visualizer.COLORS['secondary'], 
            Visualizer.COLORS['success'],
            Visualizer.COLORS['danger'],
            Visualizer.COLORS['info'],
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22'
        ]
        
        for i, (strategy_name, equity_curve) in enumerate(equity_curves.items()):
            if len(equity_curve) == 0:
                continue
                
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name=strategy_name, line=dict(color=color, width=2),
                hovertemplate=f'<b>{strategy_name}</b><br>' + '<b>Date:</b> %{x}<br>' + '<b>Value:</b> $%{y:,.2f}<br>' + '<extra></extra>'))
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=16)),
            xaxis=dict(title="Date", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid']),
            yaxis=dict(title="Portfolio Value ($)", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid'], tickformat='$,.0f'),
            plot_bgcolor=Visualizer.COLORS['background'],
            paper_bgcolor=Visualizer.COLORS['background'],
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)", bordercolor=Visualizer.COLORS['dark'], borderwidth=1)
        )
        return fig   
 
    @staticmethod
    def plot_drawdown(drawdown_series: pd.Series, title: str = "Drawdown Chart", highlight_major: bool = True, major_threshold: float = 0.1, width: int = 1000, height: int = 400) -> go.Figure:
        if len(drawdown_series) == 0:
            fig = go.Figure()
            fig.update_layout(title="No Drawdown Data Available", width=width, height=height)
            return fig
        
        fig = go.Figure()
        
        drawdown_pct = drawdown_series * 100
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=-drawdown_pct, fill='tonexty', mode='lines', name='Drawdown', line=dict(color=Visualizer.COLORS['danger'], width=1), fillcolor=f"rgba(214, 39, 40, 0.3)", hovertemplate='<b>Date:</b> %{x}<br>' + '<b>Drawdown:</b> %{y:.2f}%<br>' + '<extra></extra>'))
        
        fig.add_hline(y=0,  line_dash="dash", line_color=Visualizer.COLORS['dark'], annotation_text="No Drawdown")
        
        if highlight_major and major_threshold > 0:
            major_drawdowns = drawdown_series[drawdown_series >= major_threshold]
            if len(major_drawdowns) > 0:
                fig.add_trace(go.Scatter(x=major_drawdowns.index, y=-major_drawdowns * 100, mode='markers',
                    name=f'Major Drawdowns (>{major_threshold*100:.0f}%)',
                    marker=dict(color=Visualizer.COLORS['danger'], size=8, symbol='circle', line=dict(width=2, color='white')),
                    hovertemplate='<b>Major Drawdown</b><br>' + '<b>Date:</b> %{x}<br>' + '<b>Drawdown:</b> %{y:.2f}%<br>' + '<extra></extra>')
                    )
        
        max_drawdown = drawdown_series.max() * 100
        if isinstance(max_drawdown, pd.Series):
            max_drawdown = max_drawdown.iloc[0] if len(max_drawdown) == 1 else max_drawdown.values[0]
        
        positive_drawdowns = drawdown_series[drawdown_series > 0]
        if len(positive_drawdowns) > 0:
            avg_drawdown = positive_drawdowns.mean() * 100
            if isinstance(avg_drawdown, pd.Series):
                avg_drawdown = avg_drawdown.iloc[0] if len(avg_drawdown) == 1 else avg_drawdown.values[0]
        else:
            avg_drawdown = 0
        
        subtitle = f"Max Drawdown: {max_drawdown:.1f}% | Avg Drawdown: {avg_drawdown:.1f}%"
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><sub>{subtitle}</sub>", x=0.5, font=dict(size=16)),
            xaxis=dict(title="Date", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid']),
            yaxis=dict(title="Drawdown (%)", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid'], tickformat='.1f'),
            plot_bgcolor=Visualizer.COLORS['background'], paper_bgcolor=Visualizer.COLORS['background'], width=width, height=height, hovermode='x unified', showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, title: str = "Returns Distribution", bins: int = 50, show_normal: bool = True, width: int = 800, height: int = 500) -> go.Figure:
        if len(returns) == 0:
            fig = go.Figure()
            fig.update_layout(title="No Returns Data Available", width=width, height=height)
            return fig
        
        fig = go.Figure()
        returns_pct = returns * 100
        fig.add_trace(go.Histogram(x=returns_pct, nbinsx=bins, name='Returns', marker_color=Visualizer.COLORS['primary'], opacity=0.7, hovertemplate='<b>Return Range:</b> %{x}%<br>' + '<b>Frequency:</b> %{y}<br>' + '<extra></extra>'))
        
        if show_normal and len(returns) > 1:
            mean_return = returns_pct.mean()
            std_return = returns_pct.std()
            if isinstance(mean_return, pd.Series):
                mean_return = mean_return.iloc[0] if len(mean_return) == 1 else mean_return.values[0]
            if isinstance(std_return, pd.Series):
                std_return = std_return.iloc[0] if len(std_return) == 1 else std_return.values[0]
            min_return = returns_pct.min()
            max_return = returns_pct.max()
            if isinstance(min_return, pd.Series):
                min_return = min_return.iloc[0] if len(min_return) == 1 else min_return.values[0]
            if isinstance(max_return, pd.Series):
                max_return = max_return.iloc[0] if len(max_return) == 1 else max_return.values[0]
            
            x_range = np.linspace(min_return, max_return, 100)
            normal_curve = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
            
            normal_curve = normal_curve * len(returns) * (max_return - min_return) / bins
            fig.add_trace(go.Scatter(x=x_range, y=normal_curve, mode='lines', name='Normal Distribution', line=dict(color=Visualizer.COLORS['danger'], width=2, dash='dash'), hovertemplate='<b>Normal Distribution</b><br>' + '<b>Return:</b> %{x:.2f}%<br>' + '<b>Density:</b> %{y:.4f}<br>' + '<extra></extra>'))
        
        mean_return = returns_pct.mean()
        std_return = returns_pct.std()
        skewness = returns_pct.skew() if len(returns) > 2 else 0
        if isinstance(mean_return, pd.Series):
            mean_return = mean_return.iloc[0] if len(mean_return) == 1 else mean_return.values[0]
        if isinstance(std_return, pd.Series):
            std_return = std_return.iloc[0] if len(std_return) == 1 else std_return.values[0]
        if isinstance(skewness, pd.Series):
            skewness = skewness.iloc[0] if len(skewness) == 1 else skewness.values[0]
        
        subtitle = f"Mean: {mean_return:.2f}% | Std: {std_return:.2f}% | Skew: {skewness:.2f}"
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><sub>{subtitle}</sub>", x=0.5, font=dict(size=16)),
            xaxis=dict(title="Return (%)", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid']),
            yaxis=dict(title="Frequency", showgrid=True, gridwidth=1, gridcolor=Visualizer.COLORS['grid']),
            plot_bgcolor=Visualizer.COLORS['background'], paper_bgcolor=Visualizer.COLORS['background'], width=width, height=height, showlegend=True, bargap=0.1
        )
        return fig
    
    @staticmethod
    def plot_rolling_metrics(equity_curve: pd.Series, window: int = 30, metrics: List[str] = None, title: str = "Rolling Performance Metrics", width: int = 1000, height: int = 600) -> go.Figure:
        if len(equity_curve) < window:
            fig = go.Figure()
            fig.update_layout(title="Insufficient Data for Rolling Metrics", width=width, height=height)
            return fig
        
        if metrics is None:
            metrics = ['returns', 'volatility', 'sharpe']
        
        returns = equity_curve.pct_change().dropna()
        n_metrics = len(metrics)
        fig = make_subplots(rows=n_metrics, cols=1, subplot_titles=[f"Rolling {metric.title()} ({window}d)" for metric in metrics], vertical_spacing=0.08)
        
        colors = [Visualizer.COLORS['primary'], Visualizer.COLORS['secondary'],  Visualizer.COLORS['success'], Visualizer.COLORS['danger']]
        
        for i, metric in enumerate(metrics):
            color = colors[i % len(colors)]
            
            if metric.lower() == 'returns':
                rolling_returns = returns.rolling(window).mean() * 252 * 100
                y_data = rolling_returns
                y_title = "Annualized Return (%)"
                
            elif metric.lower() == 'volatility':
                rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
                y_data = rolling_vol
                y_title = "Volatility (%)"
                
            elif metric.lower() == 'sharpe':
                rolling_mean = returns.rolling(window).mean()
                rolling_std = returns.rolling(window).std()
                rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
                y_data = rolling_sharpe
                y_title = "Sharpe Ratio"
                
            else:
                continue
            
            fig.add_trace(
                go.Scatter(x=y_data.index, y=y_data, mode='lines', name=f'Rolling {metric.title()}', line=dict(color=color, width=2), hovertemplate=f'<b>Date:</b> %{{x}}<br>' + f'<b>{metric.title()}:</b> %{{y:.2f}}<br>' + '<extra></extra>'),
                row=i+1, col=1
            )
            
            fig.update_yaxes(title_text=y_title, row=i+1, col=1)
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=16)), plot_bgcolor=Visualizer.COLORS['background'],
            paper_bgcolor=Visualizer.COLORS['background'], width=width,
            height=height, showlegend=False, hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=n_metrics, col=1)
        return fig
    
    @staticmethod
    def create_trade_log_table(trades: List, title: str = "Trade Log", max_rows: int = 100) -> go.Figure:
        if not trades:
            fig = go.Figure()
            fig.add_annotation(text="No trades to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(title=title)
            return fig
        
        trade_data = []
        for trade in trades[:max_rows]:
            trade_info = {
                'Date': trade.timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(trade.timestamp, 'strftime') else str(trade.timestamp),
                'Type': trade.type.upper() if hasattr(trade, 'type') else 'N/A',
                'Price': f"${trade.price:.2f}" if hasattr(trade, 'price') else 'N/A',
                'Size': f"{trade.size:.4f}" if hasattr(trade, 'size') else 'N/A',
                'Value': f"${trade.value:.2f}" if hasattr(trade, 'value') else 'N/A',
                'Commission': f"${trade.commission:.2f}" if hasattr(trade, 'commission') else 'N/A',
                'Slippage': f"${trade.slippage:.2f}" if hasattr(trade, 'slippage') else 'N/A',
                'PnL': f"${trade.pnl:.2f}" if hasattr(trade, 'pnl') and trade.pnl is not None else 'N/A'
            }
            trade_data.append(trade_info)
        
        if not trade_data:
            fig = go.Figure()
            fig.add_annotation(text="No valid trade data to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=title)
            return fig
        
        df = pd.DataFrame(trade_data)
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color=Visualizer.COLORS['primary'], font=dict(color='white', size=12), align="center", height=40),
            cells=dict(values=[df[col] for col in df.columns], fill_color=[['white', '#f8f9fa'] * len(df)], align="center", height=30, font=dict(size=11))
        )])
        
        trades_with_pnl = [t for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
        if trades_with_pnl:
            total_pnl = sum(t.pnl for t in trades_with_pnl)
            win_rate = sum(1 for t in trades_with_pnl if t.pnl > 0) / len(trades_with_pnl) * 100
            subtitle = f"Total Trades: {len(trades)} | Total PnL: ${total_pnl:.2f} | Win Rate: {win_rate:.1f}%"
        else:
            subtitle = f"Total Trades: {len(trades)}"
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b><br><sub>{subtitle}</sub>", x=0.5, font=dict(size=16)),
            height=min(600, 100 + len(trade_data) * 35),  # Dynamic height based on rows
            margin=dict(l=20, r=20, t=80, b=20)
        )
        return fig