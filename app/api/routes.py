"""API routes for the trading dashboard."""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query, HTTPException
import asyncio
import pandas as pd

from ..data import MarketDataFetcher, NewsDataFetcher, SocialDataFetcher
from ..data.tradingview import TradingViewIntegration
from ..analysis import FibonacciAnalyzer, CorrelationAnalyzer, SentimentAnalyzer

router = APIRouter()

# Initialize components
market_fetcher = MarketDataFetcher()
news_fetcher = NewsDataFetcher()
social_fetcher = SocialDataFetcher()
tradingview = TradingViewIntegration()
fibonacci_analyzer = FibonacciAnalyzer()
correlation_analyzer = CorrelationAnalyzer()
sentiment_analyzer = SentimentAnalyzer()


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/api/market/stocks")
async def get_stock_data(
    symbols: Optional[str] = Query(None, description="Comma-separated stock symbols"),
    period: str = Query("1mo", description="Data period (1d, 5d, 1mo, 3mo, 1y)"),
    interval: str = Query("1d", description="Data interval (1m, 5m, 1h, 1d)")
) -> Dict[str, Any]:
    """Get stock market data."""
    symbol_list = symbols.split(",") if symbols else None

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(
        None,
        lambda: market_fetcher.get_stock_data(symbol_list, period, interval)
    )

    return {
        "market": "stocks",
        "data": {
            symbol: {
                "symbol": md.symbol,
                "name": md.name,
                "price": md.current_price,
                "change_24h": md.change_24h,
                "change_percent_24h": md.change_percent_24h,
                "high_24h": md.high_24h,
                "low_24h": md.low_24h,
                "volume_24h": md.volume_24h,
                "last_updated": md.last_updated.isoformat()
            }
            for symbol, md in data.items()
        }
    }


@router.get("/api/market/crypto")
async def get_crypto_data(
    coins: Optional[str] = Query(None, description="Comma-separated coin IDs"),
    days: int = Query(30, description="Days of historical data")
) -> Dict[str, Any]:
    """Get cryptocurrency market data."""
    coin_list = coins.split(",") if coins else None
    data = await market_fetcher.get_crypto_data(coin_list, days)

    return {
        "market": "crypto",
        "data": {
            coin: {
                "symbol": md.symbol,
                "name": md.name,
                "price": md.current_price,
                "change_24h": md.change_24h,
                "change_percent_24h": md.change_percent_24h,
                "high_24h": md.high_24h,
                "low_24h": md.low_24h,
                "volume_24h": md.volume_24h,
                "last_updated": md.last_updated.isoformat()
            }
            for coin, md in data.items()
        }
    }


@router.get("/api/market/forex")
async def get_forex_data(
    currencies: Optional[str] = Query(None, description="Comma-separated currency codes"),
    base: str = Query("USD", description="Base currency")
) -> Dict[str, Any]:
    """Get forex market data."""
    currency_list = currencies.split(",") if currencies else None
    data = await market_fetcher.get_forex_data(currency_list, base)

    return {
        "market": "forex",
        "base": base,
        "data": {
            currency: {
                "symbol": md.symbol,
                "name": md.name,
                "rate": md.current_price,
                "last_updated": md.last_updated.isoformat()
            }
            for currency, md in data.items()
        }
    }


@router.get("/api/market/all")
async def get_all_market_data() -> Dict[str, Any]:
    """Get all market data (stocks, crypto, forex)."""
    data = await market_fetcher.get_all_market_data()

    result = {}
    for market, market_data in data.items():
        result[market] = {
            symbol: {
                "symbol": md.symbol,
                "name": md.name,
                "price": md.current_price,
                "change_24h": md.change_24h,
                "change_percent_24h": md.change_percent_24h
            }
            for symbol, md in market_data.items()
        }

    return result


@router.get("/api/analysis/fibonacci/{symbol}")
async def get_fibonacci_analysis(
    symbol: str,
    period: str = Query("1mo", description="Data period")
) -> Dict[str, Any]:
    """Get Fibonacci analysis for a symbol."""
    # Fetch stock data
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(
        None,
        lambda: market_fetcher.get_stock_data([symbol], period)
    )

    if symbol not in data:
        # Try crypto
        crypto_data = await market_fetcher.get_crypto_data([symbol.lower()])
        if symbol.lower() in crypto_data:
            data = {symbol: crypto_data[symbol.lower()]}
        else:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    market_data = data[symbol]
    price_series = market_fetcher.get_price_series(market_data)

    if len(price_series) < 5:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")

    analysis = fibonacci_analyzer.analyze(price_series, symbol)
    return fibonacci_analyzer.get_levels_as_dict(analysis)


@router.get("/api/analysis/correlation")
async def get_correlation_analysis(
    assets: Optional[str] = Query(None, description="Comma-separated asset symbols"),
    period: str = Query("1mo", description="Data period")
) -> Dict[str, Any]:
    """Get correlation matrix for multiple assets."""
    asset_list = assets.split(",") if assets else ["AAPL", "GOOGL", "MSFT", "AMZN"]

    # Fetch data for all assets
    loop = asyncio.get_event_loop()
    stock_data = await loop.run_in_executor(
        None,
        lambda: market_fetcher.get_stock_data(asset_list, period)
    )

    if len(stock_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 assets for correlation")

    # Build price series dict
    price_data = {}
    for symbol, md in stock_data.items():
        series = market_fetcher.get_price_series(md)
        if len(series) > 0:
            price_data[symbol] = series

    if len(price_data) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data for correlation")

    matrix = correlation_analyzer.build_correlation_matrix(price_data, period)
    return correlation_analyzer.matrix_to_dict(matrix)


@router.get("/api/analysis/sentiment/{asset}")
async def get_asset_sentiment(asset: str) -> Dict[str, Any]:
    """Get sentiment analysis for an asset."""
    # Fetch news and social data
    news_data = await news_fetcher.get_asset_news(asset)
    social_data = await social_fetcher.get_asset_sentiment(asset)

    # Analyze sentiment
    news_texts = [a.title + " " + a.description for a in news_data.articles]
    social_texts = [p.text for p in social_data.posts]

    sentiment = sentiment_analyzer.get_market_sentiment(news_texts, social_texts)

    return {
        "asset": asset,
        "sentiment": sentiment,
        "news_count": len(news_data.articles),
        "social_count": len(social_data.posts),
        "social_trending_score": social_data.trending_score
    }


@router.get("/api/analysis/market-sentiment")
async def get_market_sentiment(
    market: str = Query("all", description="Market type (stocks, crypto, forex, all)")
) -> Dict[str, Any]:
    """Get overall market sentiment."""
    news_data = await news_fetcher.get_market_news(market)
    social_data = await social_fetcher.get_market_sentiment(market)

    news_texts = [a.title + " " + a.description for a in news_data.articles]
    social_texts = [p.text for p in social_data.posts]

    sentiment = sentiment_analyzer.get_market_sentiment(news_texts, social_texts)

    return {
        "market": market,
        "sentiment": sentiment,
        "trending_score": social_data.trending_score
    }


@router.get("/api/news")
async def get_news(
    market: str = Query("all", description="Market type"),
    limit: int = Query(20, description="Number of articles")
) -> Dict[str, Any]:
    """Get financial news."""
    news_data = await news_fetcher.get_market_news(market)

    articles = []
    for article in news_data.articles[:limit]:
        # Analyze sentiment for each article
        sentiment = sentiment_analyzer.analyze_text(article.title + " " + article.description)

        articles.append({
            "title": article.title,
            "description": article.description,
            "source": article.source,
            "url": article.url,
            "published_at": article.published_at.isoformat(),
            "categories": article.categories,
            "sentiment": {
                "score": sentiment.score,
                "label": sentiment.label.value
            }
        })

    return {
        "market": market,
        "articles": articles,
        "total": len(articles)
    }


@router.get("/api/social/{asset}")
async def get_social_data(
    asset: str,
    limit: int = Query(50, description="Number of posts")
) -> Dict[str, Any]:
    """Get social media sentiment for an asset."""
    social_data = await social_fetcher.get_asset_sentiment(asset)

    posts = []
    for post in social_data.posts[:limit]:
        posts.append({
            "id": post.id,
            "text": post.text,
            "author": post.author,
            "created_at": post.created_at.isoformat(),
            "likes": post.likes,
            "retweets": post.retweets,
            "sentiment_score": post.sentiment_score
        })

    return {
        "asset": asset,
        "posts": posts,
        "total_posts": social_data.total_posts,
        "avg_sentiment": social_data.avg_sentiment,
        "sentiment_distribution": social_data.sentiment_distribution,
        "trending_score": social_data.trending_score
    }


@router.get("/api/widgets/chart")
async def get_chart_widget(
    symbol: str,
    height: str = Query("500", description="Widget height"),
    interval: str = Query("D", description="Time interval"),
    studies: Optional[str] = Query(None, description="Comma-separated indicator names")
) -> Dict[str, str]:
    """Get TradingView chart widget HTML."""
    study_list = studies.split(",") if studies else ["volume", "macd"]

    html = tradingview.generate_chart_widget(
        symbol=symbol,
        height=height,
        interval=interval,
        studies=study_list
    )

    return {"html": html}


@router.get("/api/widgets/ticker")
async def get_ticker_widget(
    symbols: Optional[str] = Query(None, description="Comma-separated symbols")
) -> Dict[str, str]:
    """Get TradingView ticker tape widget HTML."""
    symbol_list = symbols.split(",") if symbols else None
    html = tradingview.generate_ticker_tape(symbol_list)
    return {"html": html}


@router.get("/api/widgets/overview")
async def get_overview_widget() -> Dict[str, str]:
    """Get TradingView market overview widget HTML."""
    html = tradingview.generate_market_overview()
    return {"html": html}


@router.get("/api/widgets/technical/{symbol}")
async def get_technical_widget(
    symbol: str,
    height: str = Query("400", description="Widget height")
) -> Dict[str, str]:
    """Get TradingView technical analysis widget HTML."""
    html = tradingview.generate_technical_analysis(symbol, height=height)
    return {"html": html}


@router.get("/api/dashboard/summary")
async def get_dashboard_summary() -> Dict[str, Any]:
    """Get complete dashboard summary data."""
    # Fetch all data concurrently
    market_task = market_fetcher.get_all_market_data()
    news_task = news_fetcher.get_all_news()
    social_task = social_fetcher.get_all_sentiments()

    market_data, news_data, social_data = await asyncio.gather(
        market_task, news_task, social_task
    )

    # Aggregate sentiment
    all_news_texts = []
    for market_news in news_data.values():
        all_news_texts.extend([a.title for a in market_news.articles])

    all_social_texts = []
    for asset_social in social_data.values():
        all_social_texts.extend([p.text for p in asset_social.posts])

    overall_sentiment = sentiment_analyzer.get_market_sentiment(
        all_news_texts, all_social_texts
    )

    # Build summary
    summary = {
        "markets": {},
        "sentiment": overall_sentiment,
        "top_movers": [],
        "news_highlights": []
    }

    # Process market data
    for market, data in market_data.items():
        summary["markets"][market] = {
            "count": len(data),
            "assets": [
                {
                    "symbol": md.symbol,
                    "price": md.current_price,
                    "change_percent": md.change_percent_24h
                }
                for md in list(data.values())[:5]
            ]
        }

        # Find top movers
        for md in data.values():
            summary["top_movers"].append({
                "symbol": md.symbol,
                "market": market,
                "change_percent": md.change_percent_24h
            })

    # Sort top movers
    summary["top_movers"] = sorted(
        summary["top_movers"],
        key=lambda x: abs(x["change_percent"]),
        reverse=True
    )[:10]

    # News highlights
    for market, news in news_data.items():
        for article in news.articles[:2]:
            summary["news_highlights"].append({
                "title": article.title,
                "source": article.source,
                "market": market
            })

    return summary
