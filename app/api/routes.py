"""API routes for the trading dashboard."""
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query, HTTPException
import asyncio
import pandas as pd

from ..data import MarketDataFetcher, NewsDataFetcher, SocialDataFetcher
from ..data.tradingview import TradingViewIntegration
from ..data.precious_metals import PreciousMetalsDataFetcher
from ..data.polymarket import PolymarketFetcher
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
polymarket_fetcher = PolymarketFetcher()


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

        # Scale DataFrame to spot prices BEFORE analysis
        # This ensures all detected levels and alert descriptions use spot prices
        scaled_df = df.copy()
        scaled_df['open'] = df['open'] * scale_factor
        scaled_df['high'] = df['high'] * scale_factor
        scaled_df['low'] = df['low'] * scale_factor
        scaled_df['close'] = df['close'] * scale_factor
        # Volume stays the same - it's already in shares/contracts

        # Run manipulation analysis on scaled data
        analysis = manipulation_detector.analyze(scaled_df, symbol)

        return {
            "symbol": symbol,
            "current_price": round(analysis.current_price, 2),
            "manipulation_score": analysis.overall_manipulation_score,
            "session_risk": analysis.session_risk,
            "active_alerts": [
                {
                    "type": alert.manipulation_type.value,
                    "severity": alert.severity.value,
                    "description": alert.description,
                    "price_at_detection": round(alert.price_at_detection, 2),
                    "key_level": round(alert.key_level_involved, 2) if alert.key_level_involved else None,
                    "expected_reversal": alert.expected_reversal,
                    "confidence": alert.confidence
                }
                for alert in analysis.active_alerts
            ],
            "key_levels": [
                {
                    "price": round(level.price, 2),
                    "type": level.level_type,
                    "strength": level.strength,
                    "touches": level.touches
                }
                for level in analysis.key_levels
            ],
            "order_blocks": [
                {
                    "high": round(ob.price_high, 2),
                    "low": round(ob.price_low, 2),
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


@router.get("/api/metals/intelligence")
async def get_market_intelligence() -> Dict[str, Any]:
    """Get market intelligence summary for gold and silver."""
    try:
        loop = asyncio.get_event_loop()

        # Get current prices and data
        metals_data = await loop.run_in_executor(None, metals_fetcher.get_all_metals)
        session = metals_fetcher.get_current_session()

        # Get gold and silver data
        gold_data = metals_data.get("XAU/USD")
        silver_data = metals_data.get("XAG/USD")

        gold_price = gold_data.current_price.price if gold_data else 0
        gold_change = gold_data.current_price.change_percent if gold_data else 0
        silver_price = silver_data.current_price.price if silver_data else 0
        silver_change = silver_data.current_price.change_percent if silver_data else 0

        # Generate dynamic summaries based on market conditions
        gold_direction = "higher" if gold_change > 0 else "lower"
        silver_direction = "higher" if silver_change > 0 else "lower"

        gold_momentum = "bullish" if gold_change > 0.5 else "bearish" if gold_change < -0.5 else "neutral"
        silver_momentum = "bullish" if silver_change > 0.5 else "bearish" if silver_change < -0.5 else "neutral"

        # Session-based commentary
        session_risk = session.manipulation_risk.lower()
        session_name = session.name

        return {
            "gold": {
                "price_action": f"Gold trading at ${gold_price:.2f}, {abs(gold_change):.2f}% {gold_direction} with {gold_momentum} momentum",
                "drivers": "USD strength/weakness, Treasury yields, safe-haven demand, central bank buying",
                "outlook": f"Current session ({session_name}) has {session_risk} manipulation risk. Watch for key level tests and volume spikes."
            },
            "silver": {
                "price_action": f"Silver at ${silver_price:.2f}, {abs(silver_change):.2f}% {silver_direction} following gold with {silver_momentum} bias",
                "drivers": "Industrial demand (solar, electronics), gold correlation, investment demand",
                "outlook": f"Silver typically shows 1.5-2x gold volatility. Current gold/silver ratio indicates {'silver undervalued' if (silver_price > 0 and gold_price/silver_price > 80) else 'normal range' if (silver_price > 0 and gold_price/silver_price > 70) else 'silver overvalued' if silver_price > 0 else 'ratio unavailable (no price data)'}."
            },
            "central_banks": {
                "fed": "Fed monitoring inflation data. Rate decisions impact gold inversely through USD and real yields.",
                "ecb": "ECB maintaining cautious stance. European demand for gold hedging remains steady.",
                "gold_reserves": "Central banks globally continue net gold buying trend (China, India, Turkey leading)."
            },
            "contracts": {
                "gold_expiry": "COMEX Gold (GC) - Active month contract, rollover typically 3-4 days before expiry",
                "silver_expiry": "COMEX Silver (SI) - Following gold futures calendar",
                "delivery_info": "Physical delivery 1st-3rd business day. LBMA London fix at 10:30 AM & 3:00 PM GMT."
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metals/scalp")
async def get_scalp_suggestions(
    symbol: str = Query("XAU/USD", description="Metal symbol (XAU/USD or XAG/USD)")
) -> Dict[str, Any]:
    """Get scalp trading suggestions for precious metals - designed for quick wins."""
    try:
        loop = asyncio.get_event_loop()

        # Get current metal data
        metal_data = await loop.run_in_executor(
            None,
            lambda: metals_fetcher.get_metal_data(symbol)
        )

        # Get OHLCV DataFrame
        df = metals_fetcher.get_ohlcv_dataframe(symbol, "5m")

        if df.empty:
            return {
                "symbol": symbol,
                "action": "WAIT",
                "reason": "Insufficient market data",
                "confidence": 0,
                "suggestions": []
            }

        # Get spot price and calculate scale factor
        spot_price = metal_data.current_price.price
        etf_price = df['close'].iloc[-1] if not df.empty else 1
        scale_factor = spot_price / etf_price if etf_price > 0 else 1

        # Scale DataFrame to spot prices
        scaled_df = df.copy()
        scaled_df['open'] = df['open'] * scale_factor
        scaled_df['high'] = df['high'] * scale_factor
        scaled_df['low'] = df['low'] * scale_factor
        scaled_df['close'] = df['close'] * scale_factor

        current_price = scaled_df['close'].iloc[-1]

        # Calculate indicators
        # RSI
        delta = scaled_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # EMAs
        ema_9 = scaled_df['close'].ewm(span=9).mean().iloc[-1]
        ema_21 = scaled_df['close'].ewm(span=21).mean().iloc[-1]
        ema_50 = scaled_df['close'].ewm(span=50).mean().iloc[-1]

        # VWAP approximation (using typical price * volume)
        typical_price = (scaled_df['high'] + scaled_df['low'] + scaled_df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        current_vwap = vwap.iloc[-1]

        # ATR for stop loss calculation
        high_low = scaled_df['high'] - scaled_df['low']
        high_close = abs(scaled_df['high'] - scaled_df['close'].shift())
        low_close = abs(scaled_df['low'] - scaled_df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]

        # Session info
        session = metals_fetcher.get_current_session()
        session_risk = session.manipulation_risk.lower()

        # Get manipulation analysis
        manipulation = manipulation_detector.analyze(scaled_df, symbol)

        # Determine scalp parameters based on symbol
        if "XAU" in symbol:
            min_sl = 2.0  # Minimum $2 stop loss for gold
            max_sl = 5.0  # Maximum $5 stop loss
            tick_size = 0.10
        else:  # XAG
            min_sl = 0.15  # Minimum $0.15 stop loss for silver
            max_sl = 0.40  # Maximum $0.40 stop loss
            tick_size = 0.01

        # Calculate dynamic stop loss based on ATR (0.5-1x ATR for scalping)
        scalp_sl = max(min_sl, min(max_sl, atr * 0.7))

        # Build confluence score and direction
        bullish_signals = 0
        bearish_signals = 0
        confluence_factors = []

        # RSI analysis
        if current_rsi < 30:
            bullish_signals += 2
            confluence_factors.append({"factor": "RSI Oversold", "direction": "BUY", "weight": 2})
        elif current_rsi < 40:
            bullish_signals += 1
            confluence_factors.append({"factor": "RSI Low", "direction": "BUY", "weight": 1})
        elif current_rsi > 70:
            bearish_signals += 2
            confluence_factors.append({"factor": "RSI Overbought", "direction": "SELL", "weight": 2})
        elif current_rsi > 60:
            bearish_signals += 1
            confluence_factors.append({"factor": "RSI High", "direction": "SELL", "weight": 1})

        # EMA analysis
        if current_price > ema_9 > ema_21:
            bullish_signals += 2
            confluence_factors.append({"factor": "Price > EMA9 > EMA21 (Bullish Trend)", "direction": "BUY", "weight": 2})
        elif current_price < ema_9 < ema_21:
            bearish_signals += 2
            confluence_factors.append({"factor": "Price < EMA9 < EMA21 (Bearish Trend)", "direction": "SELL", "weight": 2})

        # Mean reversion (price far from VWAP)
        vwap_distance_pct = ((current_price - current_vwap) / current_vwap) * 100 if current_vwap != 0 else 0
        if vwap_distance_pct < -0.3:
            bullish_signals += 1
            confluence_factors.append({"factor": f"Below VWAP ({vwap_distance_pct:.2f}%) - Reversion", "direction": "BUY", "weight": 1})
        elif vwap_distance_pct > 0.3:
            bearish_signals += 1
            confluence_factors.append({"factor": f"Above VWAP (+{vwap_distance_pct:.2f}%) - Reversion", "direction": "SELL", "weight": 1})

        # Session timing
        if session_risk == "low":
            confluence_factors.append({"factor": f"{session.name} - Low Manipulation Risk", "direction": "FAVORABLE", "weight": 1})
        elif session_risk == "high" or session_risk == "highest":
            confluence_factors.append({"factor": f"{session.name} - High Manipulation Risk", "direction": "CAUTION", "weight": -1})

        # Key level proximity for entries
        support_levels = [l for l in manipulation.key_levels if l.level_type == "SUPPORT"]
        resistance_levels = [l for l in manipulation.key_levels if l.level_type == "RESISTANCE"]

        nearest_support = min([l.price for l in support_levels], key=lambda x: abs(current_price - x), default=None) if support_levels else None
        nearest_resistance = min([l.price for l in resistance_levels], key=lambda x: abs(current_price - x), default=None) if resistance_levels else None

        if nearest_support and abs(current_price - nearest_support) < atr * 0.5:
            bullish_signals += 1
            confluence_factors.append({"factor": f"Near Support ${nearest_support:.2f}", "direction": "BUY", "weight": 1})

        if nearest_resistance and abs(current_price - nearest_resistance) < atr * 0.5:
            bearish_signals += 1
            confluence_factors.append({"factor": f"Near Resistance ${nearest_resistance:.2f}", "direction": "SELL", "weight": 1})

        # Manipulation alerts - reduce confidence during manipulation
        if manipulation.active_alerts:
            for alert in manipulation.active_alerts:
                if alert.expected_reversal == "UP":
                    bullish_signals += 1
                    confluence_factors.append({"factor": f"{alert.manipulation_type.value} - Expected UP", "direction": "BUY", "weight": 1})
                elif alert.expected_reversal == "DOWN":
                    bearish_signals += 1
                    confluence_factors.append({"factor": f"{alert.manipulation_type.value} - Expected DOWN", "direction": "SELL", "weight": 1})

        # Calculate final direction and confidence
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            action = "WAIT"
            direction = None
            confidence = 0
        elif bullish_signals > bearish_signals:
            direction = "BUY"
            confidence = min(95, (bullish_signals / max(total_signals, 1)) * 100 * (1 - manipulation.overall_manipulation_score * 0.3))
            if confidence >= 60 and session_risk not in ["high", "highest"]:
                action = "BUY_NOW"
            elif confidence >= 40:
                action = "PREPARE_BUY"
            else:
                action = "WAIT"
        else:
            direction = "SELL"
            confidence = min(95, (bearish_signals / max(total_signals, 1)) * 100 * (1 - manipulation.overall_manipulation_score * 0.3))
            if confidence >= 60 and session_risk not in ["high", "highest"]:
                action = "SELL_NOW"
            elif confidence >= 40:
                action = "PREPARE_SELL"
            else:
                action = "WAIT"

        # Calculate entry, SL, and TP
        if direction == "BUY":
            entry = round(current_price, 2)
            stop_loss = round(entry - scalp_sl, 2)
            tp1 = round(entry + scalp_sl * 1.0, 2)  # 1:1 R:R
            tp2 = round(entry + scalp_sl * 1.5, 2)  # 1.5:1 R:R
            tp3 = round(entry + scalp_sl * 2.0, 2)  # 2:1 R:R
        elif direction == "SELL":
            entry = round(current_price, 2)
            stop_loss = round(entry + scalp_sl, 2)
            tp1 = round(entry - scalp_sl * 1.0, 2)
            tp2 = round(entry - scalp_sl * 1.5, 2)
            tp3 = round(entry - scalp_sl * 2.0, 2)
        else:
            entry = None
            stop_loss = None
            tp1 = tp2 = tp3 = None

        # Build actionable suggestions
        suggestions = []

        # Primary suggestion based on action
        if action == "BUY_NOW":
            suggestions.append({
                "priority": 1,
                "type": "ENTRY",
                "text": f"BUY {symbol} @ ${entry:.2f}",
                "detail": f"Strong bullish confluence ({confidence:.0f}%). Set SL at ${stop_loss:.2f} (${scalp_sl:.2f} risk)"
            })
            suggestions.append({
                "priority": 2,
                "type": "TARGET",
                "text": f"TP1: ${tp1:.2f} (+${scalp_sl:.2f}) | TP2: ${tp2:.2f} (+${scalp_sl*1.5:.2f})",
                "detail": "Scale out 50% at TP1, trail stop to breakeven, take rest at TP2"
            })
        elif action == "SELL_NOW":
            suggestions.append({
                "priority": 1,
                "type": "ENTRY",
                "text": f"SELL {symbol} @ ${entry:.2f}",
                "detail": f"Strong bearish confluence ({confidence:.0f}%). Set SL at ${stop_loss:.2f} (${scalp_sl:.2f} risk)"
            })
            suggestions.append({
                "priority": 2,
                "type": "TARGET",
                "text": f"TP1: ${tp1:.2f} (-${scalp_sl:.2f}) | TP2: ${tp2:.2f} (-${scalp_sl*1.5:.2f})",
                "detail": "Scale out 50% at TP1, trail stop to breakeven, take rest at TP2"
            })
        elif action == "PREPARE_BUY":
            wait_price = nearest_support if nearest_support else round(current_price - atr * 0.3, 2)
            suggestions.append({
                "priority": 1,
                "type": "ALERT",
                "text": f"Set BUY limit @ ${wait_price:.2f}",
                "detail": f"Wait for pullback to support. Current confidence: {confidence:.0f}%"
            })
        elif action == "PREPARE_SELL":
            wait_price = nearest_resistance if nearest_resistance else round(current_price + atr * 0.3, 2)
            suggestions.append({
                "priority": 1,
                "type": "ALERT",
                "text": f"Set SELL limit @ ${wait_price:.2f}",
                "detail": f"Wait for rally to resistance. Current confidence: {confidence:.0f}%"
            })
        else:
            suggestions.append({
                "priority": 1,
                "type": "WAIT",
                "text": "No clear scalp setup - STAY FLAT",
                "detail": "Wait for stronger confluence. Preserve capital for A+ setups."
            })

        # Session-based suggestion
        if session_risk in ["high", "highest"]:
            suggestions.append({
                "priority": 3,
                "type": "WARNING",
                "text": f"HIGH MANIPULATION RISK - {session.name}",
                "detail": "Reduce position size by 50% or wait for next session"
            })
        elif session_risk == "low":
            suggestions.append({
                "priority": 3,
                "type": "INFO",
                "text": f"Favorable session: {session.name}",
                "detail": "Good liquidity, lower manipulation risk - normal position sizing OK"
            })

        # Key level suggestion
        if nearest_support and nearest_resistance:
            range_size = nearest_resistance - nearest_support
            price_in_range = (current_price - nearest_support) / range_size * 100
            suggestions.append({
                "priority": 4,
                "type": "LEVELS",
                "text": f"Range: ${nearest_support:.2f} - ${nearest_resistance:.2f}",
                "detail": f"Price at {price_in_range:.0f}% of range. {'Near support - BUY zone' if price_in_range < 30 else 'Near resistance - SELL zone' if price_in_range > 70 else 'Mid-range - wait for extremes'}"
            })

        # Scalping rules reminder
        suggestions.append({
            "priority": 5,
            "type": "RULES",
            "text": "Scalp Rules: Max 2% risk | Cut losers fast | Lock profits quick",
            "detail": "Move SL to breakeven after +0.5R. Never average down. Max 3 scalps/session."
        })

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "action": action,
            "direction": direction,
            "confidence": round(confidence, 1),
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "take_profit_3": tp3,
            "risk_amount": round(scalp_sl, 2),
            "atr": round(atr, 2),
            "rsi": round(current_rsi, 1),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "vwap": round(current_vwap, 2),
            "session": session.name,
            "session_risk": session_risk,
            "manipulation_score": round(manipulation.overall_manipulation_score * 100, 0),
            "confluence_factors": confluence_factors,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== POLYMARKET API ENDPOINTS ====================

@router.get("/api/polymarket/trending")
async def get_polymarket_trending(limit: int = Query(20, description="Number of events")) -> Dict[str, Any]:
    """Get trending Polymarket prediction markets."""
    try:
        events = await polymarket_fetcher.get_trending_events(limit)
        return {
            "events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "category": e.category,
                    "volume": round(e.volume, 2),
                    "liquidity": round(e.liquidity, 2),
                    "yes_price": round(e.yes_price * 100, 1),
                    "no_price": round(e.no_price * 100, 1),
                    "question": e.question,
                    "tags": e.tags,
                    "is_active": e.is_active
                }
                for e in events
            ],
            "count": len(events),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/polymarket/metals-intel")
async def get_polymarket_metals_intel() -> Dict[str, Any]:
    """Get Polymarket events relevant to precious metals trading."""
    try:
        events = await polymarket_fetcher.get_metals_relevant_events()
        geo_events = await polymarket_fetcher.get_geopolitical_events()

        # Build metals-relevant signals
        signals = []
        for e in events[:10]:
            price_signal = polymarket_fetcher.get_price_signal(e)
            signals.append({
                "id": e.id,
                "title": e.title,
                "category": e.category,
                "volume": round(e.volume, 2),
                "yes_price": round(e.yes_price * 100, 1),
                "no_price": round(e.no_price * 100, 1),
                "metals_relevance": round(e.metals_relevance * 100, 0),
                "relevance_reason": e.relevance_reason,
                "metals_bias": price_signal["metals_bias"],
                "bias_strength": price_signal["bias_strength"],
                "question": e.question
            })

        # Overall metals bias from prediction markets
        bullish_weight = sum(
            s["yes_price"] * s["metals_relevance"]
            for s in signals if s["metals_bias"] == "bullish"
        )
        bearish_weight = sum(
            s["yes_price"] * s["metals_relevance"]
            for s in signals if s["metals_bias"] == "bearish"
        )
        total_weight = bullish_weight + bearish_weight
        overall_bias = "neutral"
        if total_weight > 0:
            bull_pct = bullish_weight / total_weight
            if bull_pct > 0.6:
                overall_bias = "bullish"
            elif bull_pct < 0.4:
                overall_bias = "bearish"

        return {
            "metals_signals": signals,
            "geopolitical": [
                {
                    "id": e.id,
                    "title": e.title,
                    "category": e.category,
                    "volume": round(e.volume, 2),
                    "yes_price": round(e.yes_price * 100, 1),
                    "no_price": round(e.no_price * 100, 1),
                    "question": e.question
                }
                for e in geo_events[:8]
            ],
            "overall_bias": overall_bias,
            "signal_count": len(signals),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/polymarket/feed")
async def get_polymarket_feed() -> Dict[str, Any]:
    """Get full Polymarket market feed."""
    try:
        feed = await polymarket_fetcher.get_market_feed()
        return {
            "trending": [
                {
                    "id": e.id,
                    "title": e.title,
                    "category": e.category,
                    "volume": round(e.volume, 2),
                    "yes_price": round(e.yes_price * 100, 1),
                    "no_price": round(e.no_price * 100, 1),
                    "question": e.question
                }
                for e in feed.trending_events
            ],
            "metals_relevant": [
                {
                    "id": e.id,
                    "title": e.title,
                    "volume": round(e.volume, 2),
                    "yes_price": round(e.yes_price * 100, 1),
                    "metals_relevance": round(e.metals_relevance * 100, 0),
                    "relevance_reason": e.relevance_reason
                }
                for e in feed.metals_relevant
            ],
            "total_volume_24h": round(feed.total_volume_24h, 2),
            "timestamp": feed.timestamp.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── OSINT Intelligence Layer Endpoints ───────────────────────────────────────

def _get_osint_state():
    """Retrieve osint_data from quant state. Returns None if not available."""
    try:
        from app.main import _quant_state
        if _quant_state is None:
            return None
        return getattr(_quant_state, "osint_data", None)
    except Exception:
        return None


@router.get("/api/osint/status")
async def osint_status():
    """GRI score + fast OSINT score + AI macro summary + blended composite."""
    try:
        osint = _get_osint_state()
        if osint is None:
            return {"available": False, "message": "OSINT layer initializing..."}
        return {
            "available": True,
            "timestamp": osint.timestamp,
            "gri": {
                "score": osint.gri_score,
                "label": osint.gri_label,
                "geo": osint.gri_geo_component,
                "monetary": osint.gri_monetary_component,
                "safe_haven": osint.gri_safe_haven_component,
                "retail": osint.gri_retail_component,
            },
            "fast_score": osint.osint_fast_score,
            "fast_delta": osint.osint_fast_delta,
            "fast_label": osint.osint_fast_label,
            "ai_summary": osint.ai_summary,
            "ai_confidence": osint.ai_confidence,
            "ai_summary_cached_at": osint.ai_summary_cached_at,
            "blended_composite": osint.blended_composite,
            "blended_direction": osint.blended_direction,
            "fear_spike": osint.fear_spike,
            "hawkish_dominant": osint.hawkish_dominant,
            "long_multiplier_boost": osint.long_multiplier_boost,
            "trade": {
                "rule": osint.trade_bias_rule,
                "size_multiplier": osint.trade_size_multiplier,
                "direction_bias": osint.trade_direction_bias,
                "rationale": osint.trade_rationale,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/osint/narratives")
async def osint_narratives():
    """Ranked macro narratives with confidence scores and gold impact."""
    try:
        osint = _get_osint_state()
        if osint is None:
            return {"available": False, "narratives": []}
        return {
            "available": True,
            "timestamp": osint.timestamp,
            "narratives": osint.narratives,
            "dominant": osint.narratives[0] if osint.narratives else None,
            "hawkish_dominant": osint.hawkish_dominant,
            "fear_spike": osint.fear_spike,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/osint/social")
async def osint_social():
    """Reddit posts and Twitter/X tweets with availability flags."""
    try:
        osint = _get_osint_state()
        if osint is None:
            return {
                "available": False,
                "reddit": {"available": False, "posts": []},
                "twitter": {"available": False, "posts": []},
            }
        return {
            "available": True,
            "timestamp": osint.timestamp,
            "reddit": {
                "available": osint.reddit_available,
                "posts": osint.reddit_posts,
                "count": len(osint.reddit_posts),
            },
            "twitter": {
                "available": osint.twitter_available,
                "posts": osint.twitter_posts,
                "count": len(osint.twitter_posts),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/osint/gri")
async def osint_gri():
    """Gold Risk Index component breakdown + risk flags."""
    try:
        osint = _get_osint_state()
        if osint is None:
            return {"available": False, "gri_score": None}
        return {
            "available": True,
            "timestamp": osint.timestamp,
            "gri_score": osint.gri_score,
            "gri_label": osint.gri_label,
            "components": {
                "geo": {"score": osint.gri_geo_component, "weight": 0.35},
                "monetary": {"score": osint.gri_monetary_component, "weight": 0.25},
                "safe_haven": {"score": osint.gri_safe_haven_component, "weight": 0.25},
                "retail": {"score": osint.gri_retail_component, "weight": 0.15},
            },
            "risk_flags": {
                "fear_spike": osint.fear_spike,
                "hawkish_dominant": osint.hawkish_dominant,
                "long_multiplier_boost": osint.long_multiplier_boost,
            },
            "blended_composite": osint.blended_composite,
            "blended_direction": osint.blended_direction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────


# Debug: Print routes when module is loaded
print(f"[DEBUG] routes.py loaded, total routes: {len([r for r in router.routes if hasattr(r, 'path')])}")
_metal_routes = [r.path for r in router.routes if hasattr(r, 'path') and 'metal' in r.path]
print(f"[DEBUG] Metal routes: {_metal_routes}")
