"""API routes for the trading dashboard."""
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query, HTTPException
import asyncio
import pandas as pd

from ..data import MarketDataFetcher, NewsDataFetcher, SocialDataFetcher
from ..data.tradingview import TradingViewIntegration
from ..data.precious_metals import PreciousMetalsDataFetcher
from ..analysis import FibonacciAnalyzer, CorrelationAnalyzer, SentimentAnalyzer
from ..analysis.manipulation import ManipulationDetector
from ..analysis.signals import SignalGenerator
from ..analysis.backtesting import BacktestEngine
from ..trading import PositionSizer, PnLTracker

router = APIRouter()

# Initialize components
market_fetcher = MarketDataFetcher()
news_fetcher = NewsDataFetcher()
social_fetcher = SocialDataFetcher()
tradingview = TradingViewIntegration()
fibonacci_analyzer = FibonacciAnalyzer()
correlation_analyzer = CorrelationAnalyzer()
sentiment_analyzer = SentimentAnalyzer()

# Precious metals components
metals_fetcher = PreciousMetalsDataFetcher()
manipulation_detector = ManipulationDetector()
signal_generator = SignalGenerator()
backtest_engine = BacktestEngine()
position_sizer = PositionSizer()
pnl_tracker = PnLTracker()


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


# ==================== PRECIOUS METALS API ENDPOINTS ====================

@router.get("/api/metals/prices")
async def get_metals_prices() -> Dict[str, Any]:
    """Get current precious metals prices (XAU/USD, XAG/USD)."""
    try:
        loop = asyncio.get_event_loop()
        metals_data = await loop.run_in_executor(None, metals_fetcher.get_all_metals)
        session = metals_fetcher.get_current_session()

        result = {
            "session": {
                "name": session.name,
                "manipulation_risk": session.manipulation_risk,
                "is_active": session.is_active,
                "time_remaining": str(session.time_remaining) if session.time_remaining else None
            }
        }

        for symbol, data in metals_data.items():
            key = symbol.split("/")[0].lower()
            result[key] = {
                "symbol": data.symbol,
                "name": data.name,
                "price": data.current_price.price,
                "change": data.current_price.change,
                "change_percent": data.current_price.change_percent,
                "high_24h": data.current_price.high_24h,
                "low_24h": data.current_price.low_24h,
                "volume": data.current_price.volume,
                "timestamp": data.current_price.timestamp.isoformat()
            }

            if data.volume_analysis:
                result[key]["volume_analysis"] = {
                    "current": data.volume_analysis.current_volume,
                    "average": data.volume_analysis.avg_volume,
                    "ratio": data.volume_analysis.volume_ratio,
                    "trend": data.volume_analysis.volume_trend
                }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/metals/refresh")
async def refresh_metals_data() -> Dict[str, Any]:
    """Force refresh all metals data by clearing the cache."""
    try:
        # Clear the cache to force fresh data fetch
        metals_fetcher._cache.clear()

        # Fetch fresh data
        loop = asyncio.get_event_loop()
        metals_data = await loop.run_in_executor(None, metals_fetcher.get_all_metals)
        session = metals_fetcher.get_current_session()

        result = {
            "status": "refreshed",
            "timestamp": datetime.now().isoformat(),
            "session": {
                "name": session.name,
                "manipulation_risk": session.manipulation_risk,
                "is_active": session.is_active,
                "time_remaining": str(session.time_remaining) if session.time_remaining else None
            }
        }

        for symbol, data in metals_data.items():
            key = symbol.split("/")[0].lower()
            result[key] = {
                "symbol": data.symbol,
                "name": data.name,
                "price": data.current_price.price,
                "change": data.current_price.change,
                "change_percent": data.current_price.change_percent,
                "high_24h": data.current_price.high_24h,
                "low_24h": data.current_price.low_24h,
                "volume": data.current_price.volume,
                "timestamp": data.current_price.timestamp.isoformat()
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/signals")
async def get_metals_signals(
    symbol: str = Query("XAU/USD", description="Metal symbol (XAU/USD or XAG/USD)")
) -> Dict[str, Any]:
    """Get trade signals for precious metals."""
    try:
        loop = asyncio.get_event_loop()
        metal_data = await loop.run_in_executor(
            None,
            lambda: metals_fetcher.get_metal_data(symbol)
        )

        # Get OHLCV DataFrame for analysis (ETF data - GLD/SLV)
        df = metals_fetcher.get_ohlcv_dataframe(symbol, "5m")

        if df.empty:
            return {
                "symbol": symbol,
                "has_signal": False,
                "message": "Insufficient data for signal generation"
            }

        # Get spot price to calculate scale factor (ETF prices are ~1/10th of spot gold)
        spot_price = metal_data.current_price.price
        etf_price = df['close'].iloc[-1] if not df.empty else 1
        scale_factor = spot_price / etf_price if etf_price > 0 else 1

        # Get manipulation alerts first
        manipulation = manipulation_detector.analyze(df, symbol)

        # Generate signals
        signal_summary = signal_generator.get_signal_summary(df, symbol)

        # Scale top_signal prices to spot prices
        scaled_top_signal = None
        if signal_summary["top_signal"]:
            ts = signal_summary["top_signal"]
            scaled_top_signal = {
                "direction": ts["direction"],
                "entry": round(ts["entry"] * scale_factor, 2) if ts["entry"] else None,
                "stop_loss": round(ts["stop_loss"] * scale_factor, 2) if ts["stop_loss"] else None,
                "take_profit_1": round(ts["take_profit_1"] * scale_factor, 2) if ts["take_profit_1"] else None,
                "take_profit_2": round(ts["take_profit_2"] * scale_factor, 2) if ts["take_profit_2"] else None,
                "position_size": ts["position_size"],
                "confidence": ts["confidence"],
                "reasoning": ts["reasoning"],
                "risk_reward_1": ts["risk_reward_1"],
                "risk_reward_2": ts["risk_reward_2"],
            }

        # Scale indicator EMAs to spot prices
        scaled_indicators = {
            "rsi": signal_summary["indicators"]["rsi"],
            "macd": round(signal_summary["indicators"]["macd"] * scale_factor, 4),
            "macd_signal": round(signal_summary["indicators"]["macd_signal"] * scale_factor, 4),
            "ema_9": round(signal_summary["indicators"]["ema_9"] * scale_factor, 2),
            "ema_21": round(signal_summary["indicators"]["ema_21"] * scale_factor, 2),
        }

        return {
            "symbol": symbol,
            "has_signal": signal_summary["has_signal"],
            "market_bias": signal_summary["market_bias"],
            "overall_confidence": signal_summary["overall_confidence"],
            "top_signal": scaled_top_signal,
            "indicators": scaled_indicators,
            "signal_count": signal_summary["signal_count"],
            "manipulation_score": manipulation.overall_manipulation_score
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/manipulation")
async def get_manipulation_analysis(
    symbol: str = Query("XAU/USD", description="Metal symbol")
) -> Dict[str, Any]:
    """Get manipulation detection analysis for precious metals."""
    try:
        loop = asyncio.get_event_loop()

        # Get OHLCV data (from ETF - GLD/SLV)
        df = await loop.run_in_executor(
            None,
            lambda: metals_fetcher.get_ohlcv_dataframe(symbol, "5m")
        )

        if df.empty:
            return {
                "symbol": symbol,
                "manipulation_score": 0,
                "session_risk": "UNKNOWN",
                "active_alerts": [],
                "key_levels": [],
                "recommendation": "No data available"
            }

        # Get current spot price to calculate scale factor
        metal_data = await loop.run_in_executor(
            None,
            lambda: metals_fetcher.get_metal_data(symbol)
        )
        spot_price = metal_data.current_price.price

        # Calculate scale factor: ETF prices are ~1/10th of spot gold, ~1/3rd of spot silver
        etf_price = df['close'].iloc[-1] if not df.empty else 1
        scale_factor = spot_price / etf_price if etf_price > 0 else 1

        # Run manipulation analysis
        analysis = manipulation_detector.analyze(df, symbol)

        return {
            "symbol": symbol,
            "manipulation_score": analysis.overall_manipulation_score,
            "session_risk": analysis.session_risk,
            "active_alerts": [
                {
                    "type": alert.manipulation_type.value,
                    "severity": alert.severity.value,
                    "description": alert.description,
                    "key_level": alert.key_level_involved * scale_factor if alert.key_level_involved else None,
                    "expected_reversal": alert.expected_reversal,
                    "confidence": alert.confidence
                }
                for alert in analysis.active_alerts
            ],
            "key_levels": [
                {
                    "price": level.price * scale_factor,
                    "type": level.level_type,
                    "strength": level.strength,
                    "touches": level.touches
                }
                for level in analysis.key_levels
            ],
            "order_blocks": [
                {
                    "high": ob.price_high * scale_factor,
                    "low": ob.price_low * scale_factor,
                    "type": ob.block_type,
                    "strength": ob.strength,
                    "is_tested": ob.is_tested
                }
                for ob in analysis.order_blocks
            ],
            "recommendation": analysis.recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/correlations")
async def get_metals_correlations() -> Dict[str, Any]:
    """Get correlation data for precious metals with other assets."""
    try:
        loop = asyncio.get_event_loop()
        correlated = await loop.run_in_executor(
            None,
            metals_fetcher.get_correlated_assets
        )

        return {
            "correlations": [
                {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "price": asset.price,
                    "change_percent": asset.change_percent,
                    "correlation_type": asset.correlation_type
                }
                for asset in correlated
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/backtest")
async def run_metals_backtest(
    symbol: str = Query("XAU/USD", description="Metal symbol"),
    strategy: str = Query("default", description="Strategy name")
) -> Dict[str, Any]:
    """Run backtest on precious metals trading strategy."""
    try:
        loop = asyncio.get_event_loop()

        # Get historical data
        df = await loop.run_in_executor(
            None,
            lambda: metals_fetcher.get_ohlcv_dataframe(symbol, "1h")
        )

        if df.empty or len(df) < 100:
            return {
                "symbol": symbol,
                "error": "Insufficient historical data for backtesting"
            }

        # Run backtest
        result = backtest_engine.run_backtest(
            df=df,
            signal_generator=signal_generator.generate_signals,
            symbol=symbol,
            strategy_name=strategy
        )

        return backtest_engine.get_backtest_summary(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/pnl")
async def get_metals_pnl() -> Dict[str, Any]:
    """Get P&L summary for metals trading."""
    try:
        return pnl_tracker.get_pnl_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/position-size")
async def calculate_position_size(
    symbol: str = Query("XAU/USD", description="Metal symbol"),
    entry_price: float = Query(..., description="Entry price"),
    stop_loss: float = Query(..., description="Stop loss price"),
    take_profit_1: float = Query(..., description="First take profit"),
    take_profit_2: float = Query(..., description="Second take profit"),
    risk_percent: Optional[float] = Query(None, description="Custom risk percentage")
) -> Dict[str, Any]:
    """Calculate optimal position size for a trade."""
    try:
        position = position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            symbol=symbol,
            custom_risk_percent=risk_percent
        )

        return position_sizer.get_position_summary(position)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/session")
async def get_trading_session() -> Dict[str, Any]:
    """Get current trading session information."""
    session = metals_fetcher.get_current_session()

    return {
        "session": session.session.value,
        "name": session.name,
        "start_hour": session.start_hour,
        "end_hour": session.end_hour,
        "manipulation_risk": session.manipulation_risk,
        "description": session.description,
        "is_active": session.is_active,
        "time_remaining": str(session.time_remaining) if session.time_remaining else None
    }


# Debug: Print routes when module is loaded
print(f"[DEBUG] routes.py loaded, total routes: {len([r for r in router.routes if hasattr(r, 'path')])}")
_metal_routes = [r.path for r in router.routes if hasattr(r, 'path') and 'metal' in r.path]
print(f"[DEBUG] Metal routes: {_metal_routes}")
