from __future__ import annotations
import os
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Try multiple possible paths for .env file
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir.parent.parent.parent / '.env',  # Strategy-Backtester/.env
        current_dir.parent.parent.parent.parent / '.env',  # One level up
        Path('.env'),  # Current working directory
    ]
    
    env_loaded = False
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(env_path)
            env_loaded = True
            break
    
    if not env_loaded:
        print("⚠️ Could not find .env file in any expected location")
        
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

DEFAULT_SENTIMENT_SCORE_DEF = (
    "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; "
    "-0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish"
)
DEFAULT_RELEVANCE_SCORE_DEF = "0 < x <= 1, with a higher score indicating higher relevance."


class AlphaNewsError(Exception):
    pass


class News:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, ticker: str, apikey: Optional[str] = None) -> None:
        self.ticker = ticker
        
        if apikey:
            self.apikey = apikey
        else:
            try:
                import streamlit as st
                self.apikey = st.secrets.get("ALPHAVANTAGE_API_KEY")
            except (ImportError, AttributeError, KeyError):
                self.apikey = os.getenv("ALPHAVANTAGE_API_KEY")
        
        if not self.apikey:
            raise ValueError(
                "API key missing. Please set ALPHAVANTAGE_API_KEY in Streamlit secrets "
                "(for production) or environment variables (for local development)."
            )

    def _call_api(self, limit: Optional[int] = None, time_from: Optional[str] = None, time_to: Optional[str] = None, topics: Optional[str] = None, sort: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.ticker,
            "apikey": self.apikey,
        }
        if limit is not None:
            params["limit"] = int(limit)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        if topics:
            params["topics"] = topics
        if sort:
            params["sort"] = sort

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
        except requests.RequestException as e:
            raise AlphaNewsError(f"Network error during API call: {e}") from e

        if resp.status_code != 200:
            raise AlphaNewsError(f"AlphaVantage returned HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as e:
            raise AlphaNewsError("Response is not valid JSON") from e

        for key in ("Note", "Information", "Error Message"):
            if key in data:
                error_msg = str(data[key])
                import re
                error_msg = re.sub(r'\b[A-Z0-9]{16}\b', '[API_KEY_HIDDEN]', error_msg)
                error_msg = re.sub(r'your API key as [A-Z0-9]+', 'your API key', error_msg)
                if "rate limit" in error_msg.lower():
                    raise AlphaNewsError("API rate limit exceeded. Please try again later or upgrade your plan.")
                elif "api key" in error_msg.lower():
                    raise AlphaNewsError("API key configuration issue. Please check your settings.")
                else:
                    raise AlphaNewsError(f"API error: {error_msg}")

        return data

    def _score_to_label(self, x: float) -> str:
        if x <= -0.35:
            return "Bearish"
        if -0.35 < x <= -0.15:
            return "Somewhat-Bearish"
        if -0.15 < x < 0.15:
            return "Neutral"
        if 0.15 <= x < 0.35:
            return "Somewhat-Bullish"
        return "Bullish"

    def _safe_float(self, value) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    def _parse_feed(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        feed_raw = raw.get("feed", [])
        items_val = raw.get("items")
        if items_val is None:
            items_val = str(len(feed_raw))
        else:
            items_val = str(items_val)

        parsed_feed: List[Dict[str, Any]] = []
        for article in feed_raw:
            title = article.get("title")
            url = article.get("url")
            time_published = article.get("time_published")
            authors = article.get("authors", []) or []
            summary = article.get("summary")
            banner_image = article.get("banner_image")
            source = article.get("source")
            category_within_source = article.get("category_within_source")
            source_domain = article.get("source_domain")
            topics = article.get("topics", []) or []
            overall_score = self._safe_float(article.get("overall_sentiment_score", None))
            if overall_score is None:
                overall_score = self._safe_float(article.get("overall_sentiment_score"))
            overall_label = (
                article.get("overall_sentiment_label")
                or (self._score_to_label(overall_score) if overall_score is not None else None)
            )

            ticker_sent = []
            for t in article.get("ticker_sentiment", []) or []:
                ts_score = t.get("ticker_sentiment_score")
                ts_score_num = self._safe_float(ts_score)
                ticker_sent.append(
                    {
                        "ticker": t.get("ticker"),
                        "relevance_score": t.get("relevance_score"),
                        "ticker_sentiment_score": ts_score_num if ts_score_num is not None else t.get("ticker_sentiment_score"),
                        "ticker_sentiment_label": t.get("ticker_sentiment_label"),
                    }
                )

            parsed_feed.append(
                {
                    "title": title,
                    "url": url,
                    "time_published": time_published,
                    "authors": authors,
                    "summary": summary,
                    "banner_image": banner_image,
                    "source": source,
                    "category_within_source": category_within_source,
                    "source_domain": source_domain,
                    "topics": topics,
                    "overall_sentiment_score": overall_score,
                    "overall_sentiment_label": overall_label,
                    "ticker_sentiment": ticker_sent,
                }
            )

        return {
            "items": items_val,
            "sentiment_score_definition": DEFAULT_SENTIMENT_SCORE_DEF,
            "relevance_score_definition": DEFAULT_RELEVANCE_SCORE_DEF,
            "feed": parsed_feed,
        }

    def get_news(self, limit: Optional[int] = 50, time_from: Optional[str] = None, time_to: Optional[str] = None, topics: Optional[str] = None, sort: Optional[str] = "LATEST") -> Dict[str, Any]:
        raw = self._call_api(limit=limit, time_from=time_from, time_to=time_to, topics=topics, sort=sort)
        parsed = self._parse_feed(raw)
        return parsed

    def headlines(self, **kwargs) -> List[Dict[str, Any]]:
        parsed = self.get_news(**kwargs)
        return [
            {
                "title": item["title"],
                "url": item["url"],
                "time_published": item["time_published"],
                "overall_sentiment_label": item["overall_sentiment_label"],
                "overall_sentiment_score": item["overall_sentiment_score"],
            }
            for item in parsed["feed"]
        ]