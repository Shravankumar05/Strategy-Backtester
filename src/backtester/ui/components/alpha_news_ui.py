import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional
import time

try:
    from backtester.news.news import News, AlphaNewsError
except ImportError as e:
    st.error(f"❌ News module import error: {e}")
    st.error("Please ensure the Alpha Vantage news module is properly installed.")


def convert_sentiment_to_score(sentiment_value: float) -> int:
    score = int((sentiment_value + 1.0) * 50)
    return max(0, min(100, score))

def get_sentiment_color(sentiment_label: str) -> str:
    if sentiment_label == "Bullish":
        return "#28a745"
    elif sentiment_label == "Somewhat-Bullish":
        return "#20c997"
    elif sentiment_label == "Bearish":
        return "#dc3545"
    elif sentiment_label == "Somewhat-Bearish":
        return "#fd7e14"
    else:
        return "#add8e6"

def get_sentiment_emoji(sentiment_label: str) -> str:
    if sentiment_label in ["Bullish", "Somewhat-Bullish"]:
        return "🟢"
    elif sentiment_label in ["Bearish", "Somewhat-Bearish"]:
        return "🔴"
    else:
        return "🔵"


def format_date(date_string: str) -> str:
    if not date_string or date_string == "Unknown time":
        return "Unknown time"
    
    try:
        if 'T' in date_string and len(date_string) >= 15:
            date_part = date_string[:8]
            time_part = date_string[9:15]
            
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            dt = datetime(year, month, day, hour, minute)
            return dt.strftime("%b %d, %Y at %I:%M %p")
        else:
            return date_string
    except (ValueError, IndexError):
        return date_string


def calculate_overall_sentiment_score(articles: List[Dict]) -> tuple:
    if not articles:
        return 50, {"Neutral": 100}
    
    sentiment_scores = []
    sentiment_counts = {}
    
    for article in articles:
        ticker_sentiment = None
        ticker_sentiments = article.get("ticker_sentiment", [])
        for ts in ticker_sentiments:
            if ts.get("ticker_sentiment_score") is not None:
                ticker_sentiment = ts.get("ticker_sentiment_score")
                break
        
        if ticker_sentiment is not None:
            sentiment_scores.append(ticker_sentiment)
        elif article.get("overall_sentiment_score") is not None:
            sentiment_scores.append(article.get("overall_sentiment_score"))
        
        label = article.get("overall_sentiment_label", "Neutral")
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
    
    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        overall_score = convert_sentiment_to_score(avg_score)
    else:
        overall_score = 50
    
    total_articles = len(articles)
    sentiment_percentages = {
        label: (count / total_articles) * 100 
        for label, count in sentiment_counts.items()
    }
    
    return overall_score, sentiment_percentages


def render_sentiment_distribution_chart(sentiment_percentages: Dict[str, float]):
    if not sentiment_percentages:
        st.info("No sentiment data available")
        return
    
    labels = list(sentiment_percentages.keys())
    values = list(sentiment_percentages.values())
    colors = [get_sentiment_color(label) for label in labels]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), textinfo='label+percent', textposition='inside')])
    fig.update_layout(title="Sentiment Distribution", height=400, showlegend=True)
    return fig


def render_overall_sentiment_card(symbol: str, score: int, confidence: float = 0.8):
    if score >= 70:
        color = "#28a745"
        label = "Very Bullish"
        emoji = "🌙"
    elif score >= 55:
        color = "#17a2b8"
        label = "Bullish"
        emoji = "🟢"
    elif score >= 45:
        color = "#add8e6"
        label = "Neutral"
        emoji = "🔵"
    elif score >= 30:
        color = "#fd7e14"
        label = "Bearish"
        emoji = "🔴"
    else:
        color = "#dc3545"
        label = "Very Bearish"
        emoji = "📉"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20, {color}10);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h1 style="color: {color}; margin: 0; font-size: 3rem; font-weight: bold;">
            {emoji} {score}/100
        </h1>
        <h3 style="color: {color}; margin: 0.5rem 0; font-size: 1.5rem;">
            {label}
        </h3>
        <p style="color: #666; margin: 0; font-size: 1rem;">
            Overall News Sentiment for {symbol}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_news_articles_list(articles: List[Dict], ticker: str):
    """Render a list of news articles with sentiment indicators"""
    if not articles:
        st.info("No news articles available")
        return
    
    st.subheader(f"📰 Latest News for {ticker}")
    st.write(f"**{len(articles)} articles found**")
    
    for i, article in enumerate(articles, 1):
        try:
            title = article.get("title", "No title")
            url = article.get("url", "")
            source = article.get("source", "Unknown source")
            time_published = format_date(article.get("time_published", "Unknown time"))
            overall_sentiment_label = article.get("overall_sentiment_label", "Neutral")
            overall_sentiment_score = article.get("overall_sentiment_score")
            
            if overall_sentiment_score is not None:
                try:
                    overall_sentiment_score = float(overall_sentiment_score)
                except (ValueError, TypeError):
                    overall_sentiment_score = None
            
            ticker_sentiment_label = None
            ticker_sentiment_score = None
            relevance_score = None
            ticker_sentiments = article.get("ticker_sentiment", [])
            for ts in ticker_sentiments:
                if ts.get("ticker") == ticker:
                    ticker_sentiment_label = ts.get("ticker_sentiment_label")
                    ticker_sentiment_score = ts.get("ticker_sentiment_score")
                    relevance_score = ts.get("relevance_score")
                    if relevance_score is not None:
                        try:
                            relevance_score = float(relevance_score)
                        except (ValueError, TypeError):
                            relevance_score = None
                    break
            
            display_sentiment_label = ticker_sentiment_label or overall_sentiment_label
            display_sentiment_score = ticker_sentiment_score or overall_sentiment_score
            if display_sentiment_score is not None:
                try:
                    display_sentiment_score = float(display_sentiment_score)
                except (ValueError, TypeError):
                    display_sentiment_score = None
            
            sentiment_color = get_sentiment_color(display_sentiment_label)
            sentiment_emoji = get_sentiment_emoji(display_sentiment_label)
            with st.container():
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.markdown(f"""
                    <div style="
                        background-color: {sentiment_color}20;
                        border-left: 4px solid {sentiment_color};
                        padding: 1rem;
                        text-align: center;
                        border-radius: 8px;
                        height: 100%;
                    ">
                        <div style="font-size: 2rem;">{sentiment_emoji}</div>
                        <div style="color: {sentiment_color}; font-weight: bold; font-size: 0.9rem;">
                            {display_sentiment_label}
                        </div>
                        {f'<div style="color: #666; font-size: 0.8rem;">Score: {display_sentiment_score:.3f}</div>' if display_sentiment_score is not None else ''}
                        {f'<div style="color: #666; font-size: 0.8rem;">Relevance: {relevance_score:.2f}</div>' if relevance_score is not None else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"### {i}. {title}")
                    st.markdown(f"🏢 **{source}** | ⏰ {time_published}")
                    
                    if url:
                        st.markdown(f"🔗 [Read full article]({url})")
                    
                    # Show summary if available
                    summary = article.get("summary")
                    if summary and len(summary) > 50:
                        with st.expander("📄 Article Summary"):
                            st.write(summary)
            
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Error processing article {i}: {str(e)}")
            continue


def render_alpha_news_tab():
    st.header("📰 Stock News & Sentiment Analysis")
    st.markdown("Get the latest news and sentiment analysis for any stock using Alpha Vantage")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, GOOGL, TSLA, NVDA)",
            key="alpha_news_symbol"
        ).upper()
    
    with col2:
        limit = st.selectbox(
            "Number of Articles",
            options=[10, 20, 30, 50],
            index=1,
            help="Number of articles to fetch",
            key="alpha_news_limit"
        )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button(
            "📈News & Sentiment",
            type="primary",
            use_container_width=True,
            key="alpha_news_btn"
        )
    
    if analyze_button and symbol:
        with st.spinner(f"Fetching news for {symbol}..."):
            try:
                news_api = News(symbol)
                result = news_api.get_news(limit=limit)
                articles = result.get("feed", [])
                if len(articles) > limit:
                    articles = articles[:limit]
                
                if not articles:
                    st.warning(f"No articles found for {symbol}. Please try a different symbol.")
                    return
                
                overall_score, sentiment_percentages = calculate_overall_sentiment_score(articles)
                render_overall_sentiment_card(symbol, overall_score)
                tab1, tab2 = st.tabs(["📰 News Articles", "📊 Sentiment Summary"])
                
                with tab1:
                    st.markdown(f"**Found {len(articles)} articles for {symbol}** (requested: {limit})")
                    render_news_articles_list(articles, symbol)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Sentiment Summary")
                        
                        for sentiment, percentage in sentiment_percentages.items():
                            color = get_sentiment_color(sentiment)
                            emoji = get_sentiment_emoji(sentiment)
                            count = int((percentage / 100) * len(articles))
                            
                            st.markdown(f"""
                            <div style="
                                background-color: {color}20;
                                border-left: 4px solid {color};
                                padding: 0.8rem;
                                margin: 0.5rem 0;
                                border-radius: 8px;
                            ">
                                {emoji} <strong>{sentiment}</strong>: {count} articles ({percentage:.1f}%)
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("📈 Distribution Chart")
                        chart = render_sentiment_distribution_chart(sentiment_percentages)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                
                st.session_state[f'alpha_news_data_{symbol}'] = {
                    'score': overall_score,
                    'articles': articles,
                    'sentiment_percentages': sentiment_percentages,
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ Analysis completed! Found {len(articles)} articles with overall sentiment: {overall_score}/100")
                
            except AlphaNewsError as e:
                # Sanitize error message to remove API key
                error_msg = str(e)
                
                # Remove API key from error message
                import re
                # Pattern to match API key (alphanumeric string that looks like an API key)
                error_msg = re.sub(r'\b[A-Z0-9]{16}\b', '[API_KEY_HIDDEN]', error_msg)
                error_msg = re.sub(r'your API key as [A-Z0-9]+', 'your API key', error_msg)
                
                # Show generic error messages for common issues
                if "rate limit" in error_msg.lower():
                    st.error("❌ API Rate Limit Exceeded")
                    st.error("You have reached the daily API request limit. Please try again tomorrow or upgrade your API plan.")
                elif "api key" in error_msg.lower():
                    st.error("❌ API Key Issue")
                    st.error("There is an issue with the API key configuration. Please check your settings.")
                elif "invalid" in error_msg.lower():
                    st.error("❌ Invalid Request")
                    st.error("The request was invalid. Please check the stock symbol and try again.")
                else:
                    st.error("❌ API Error")
                    st.error("An error occurred while fetching news data. Please try again later.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.error("Please check your API key and internet connection.")
    
    elif analyze_button:
        st.error("Please enter a stock symbol")

def render_sentiment_analysis_tab():
    render_alpha_news_tab()