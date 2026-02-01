"""TradingView widget integration for chart embeds."""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TradingViewWidget:
    """TradingView widget configuration."""
    symbol: str
    width: str = "100%"
    height: str = "400"
    theme: str = "dark"
    interval: str = "D"
    timezone: str = "Etc/UTC"
    style: str = "1"
    locale: str = "en"
    toolbar_bg: str = "#f1f3f6"
    enable_publishing: bool = False
    hide_side_toolbar: bool = False
    allow_symbol_change: bool = True
    studies: List[str] = None
    container_id: str = None

    def __post_init__(self):
        if self.studies is None:
            self.studies = []
        if self.container_id is None:
            self.container_id = f"tv_chart_{self.symbol.replace(':', '_')}"


class TradingViewIntegration:
    """Generates TradingView widget embed codes."""

    # Symbol mappings for different markets
    SYMBOL_MAPPINGS = {
        # Stocks
        "AAPL": "NASDAQ:AAPL",
        "GOOGL": "NASDAQ:GOOGL",
        "MSFT": "NASDAQ:MSFT",
        "AMZN": "NASDAQ:AMZN",
        "TSLA": "NASDAQ:TSLA",
        "META": "NASDAQ:META",
        "NVDA": "NASDAQ:NVDA",
        # Crypto
        "BTC": "BINANCE:BTCUSDT",
        "ETH": "BINANCE:ETHUSDT",
        "SOL": "BINANCE:SOLUSDT",
        "ADA": "BINANCE:ADAUSDT",
        "BITCOIN": "BINANCE:BTCUSDT",
        "ETHEREUM": "BINANCE:ETHUSDT",
        # Forex
        "EURUSD": "FX:EURUSD",
        "GBPUSD": "FX:GBPUSD",
        "USDJPY": "FX:USDJPY",
        "AUDUSD": "FX:AUDUSD",
        # Indices
        "SPX": "SP:SPX",
        "DJI": "DJ:DJI",
        "NDX": "NASDAQ:NDX",
    }

    # Available TradingView studies (indicators)
    AVAILABLE_STUDIES = {
        "sma": "MASimple@tv-basicstudies",
        "ema": "MAExp@tv-basicstudies",
        "rsi": "RSI@tv-basicstudies",
        "macd": "MACD@tv-basicstudies",
        "bollinger": "BB@tv-basicstudies",
        "volume": "Volume@tv-basicstudies",
        "fibonacci": "FibRetracement@tv-basicstudies",
        "ichimoku": "IchimokuCloud@tv-basicstudies",
        "stoch": "Stochastic@tv-basicstudies",
        "atr": "ATR@tv-basicstudies",
    }

    def __init__(self, default_theme: str = "dark"):
        self.default_theme = default_theme

    def get_tv_symbol(self, symbol: str) -> str:
        """Convert a simple symbol to TradingView format."""
        symbol_upper = symbol.upper()
        return self.SYMBOL_MAPPINGS.get(symbol_upper, symbol_upper)

    def generate_chart_widget(
        self,
        symbol: str,
        width: str = "100%",
        height: str = "500",
        interval: str = "D",
        studies: List[str] = None,
        theme: str = None
    ) -> str:
        """
        Generate TradingView Advanced Chart widget HTML.

        Args:
            symbol: Asset symbol
            width: Widget width
            height: Widget height
            interval: Time interval (1, 5, 15, 60, D, W, M)
            studies: List of indicator names to add
            theme: 'light' or 'dark'

        Returns:
            HTML string for the widget
        """
        tv_symbol = self.get_tv_symbol(symbol)
        theme = theme or self.default_theme
        container_id = f"tv_chart_{symbol.replace(':', '_').replace('/', '_')}"

        # Convert study names to TradingView format
        study_list = []
        if studies:
            for study in studies:
                if study.lower() in self.AVAILABLE_STUDIES:
                    study_list.append(self.AVAILABLE_STUDIES[study.lower()])

        studies_json = str(study_list).replace("'", '"')

        return f'''
        <div class="tradingview-widget-container" id="{container_id}">
            <div id="{container_id}_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{
                "width": "{width}",
                "height": "{height}",
                "symbol": "{tv_symbol}",
                "interval": "{interval}",
                "timezone": "Etc/UTC",
                "theme": "{theme}",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "studies": {studies_json},
                "container_id": "{container_id}_chart"
            }});
            </script>
        </div>
        '''

    def generate_ticker_tape(
        self,
        symbols: List[str] = None,
        theme: str = None
    ) -> str:
        """
        Generate TradingView Ticker Tape widget HTML.

        Args:
            symbols: List of asset symbols
            theme: 'light' or 'dark'

        Returns:
            HTML string for the ticker tape widget
        """
        symbols = symbols or ["AAPL", "GOOGL", "BTC", "ETH", "EURUSD"]
        theme = theme or self.default_theme

        symbol_list = []
        for symbol in symbols:
            tv_symbol = self.get_tv_symbol(symbol)
            symbol_list.append(f'{{"proName": "{tv_symbol}", "title": "{symbol}"}}')

        symbols_json = "[" + ",".join(symbol_list) + "]"

        return f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
            {{
                "symbols": {symbols_json},
                "showSymbolLogo": true,
                "colorTheme": "{theme}",
                "isTransparent": false,
                "displayMode": "adaptive",
                "locale": "en"
            }}
            </script>
        </div>
        '''

    def generate_market_overview(
        self,
        tabs: List[Dict] = None,
        theme: str = None
    ) -> str:
        """
        Generate TradingView Market Overview widget HTML.

        Args:
            tabs: List of tab configurations
            theme: 'light' or 'dark'

        Returns:
            HTML string for the market overview widget
        """
        theme = theme or self.default_theme

        if tabs is None:
            tabs = [
                {
                    "title": "Stocks",
                    "symbols": [
                        {"s": "NASDAQ:AAPL"},
                        {"s": "NASDAQ:GOOGL"},
                        {"s": "NASDAQ:MSFT"},
                        {"s": "NASDAQ:AMZN"},
                        {"s": "NASDAQ:TSLA"}
                    ]
                },
                {
                    "title": "Crypto",
                    "symbols": [
                        {"s": "BINANCE:BTCUSDT"},
                        {"s": "BINANCE:ETHUSDT"},
                        {"s": "BINANCE:SOLUSDT"},
                        {"s": "BINANCE:ADAUSDT"}
                    ]
                },
                {
                    "title": "Forex",
                    "symbols": [
                        {"s": "FX:EURUSD"},
                        {"s": "FX:GBPUSD"},
                        {"s": "FX:USDJPY"}
                    ]
                }
            ]

        tabs_json = str(tabs).replace("'", '"')

        return f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
            {{
                "colorTheme": "{theme}",
                "dateRange": "12M",
                "showChart": true,
                "locale": "en",
                "width": "100%",
                "height": "500",
                "largeChartUrl": "",
                "isTransparent": false,
                "showSymbolLogo": true,
                "showFloatingTooltip": false,
                "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
                "plotLineColorFalling": "rgba(41, 98, 255, 1)",
                "gridLineColor": "rgba(42, 46, 57, 0)",
                "scaleFontColor": "rgba(134, 137, 147, 1)",
                "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
                "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
                "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
                "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
                "symbolActiveColor": "rgba(41, 98, 255, 0.12)",
                "tabs": {tabs_json}
            }}
            </script>
        </div>
        '''

    def generate_technical_analysis(
        self,
        symbol: str,
        width: str = "100%",
        height: str = "400",
        theme: str = None
    ) -> str:
        """
        Generate TradingView Technical Analysis widget HTML.

        Args:
            symbol: Asset symbol
            width: Widget width
            height: Widget height
            theme: 'light' or 'dark'

        Returns:
            HTML string for the technical analysis widget
        """
        tv_symbol = self.get_tv_symbol(symbol)
        theme = theme or self.default_theme

        return f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
            {{
                "interval": "1D",
                "width": "{width}",
                "height": "{height}",
                "symbol": "{tv_symbol}",
                "showIntervalTabs": true,
                "locale": "en",
                "colorTheme": "{theme}"
            }}
            </script>
        </div>
        '''

    def generate_mini_chart(
        self,
        symbol: str,
        width: str = "350",
        height: str = "220",
        theme: str = None
    ) -> str:
        """
        Generate TradingView Mini Chart widget HTML.

        Args:
            symbol: Asset symbol
            width: Widget width
            height: Widget height
            theme: 'light' or 'dark'

        Returns:
            HTML string for the mini chart widget
        """
        tv_symbol = self.get_tv_symbol(symbol)
        theme = theme or self.default_theme

        return f'''
        <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
            {{
                "symbol": "{tv_symbol}",
                "width": "{width}",
                "height": "{height}",
                "locale": "en",
                "dateRange": "1M",
                "colorTheme": "{theme}",
                "isTransparent": false,
                "autosize": false,
                "largeChartUrl": ""
            }}
            </script>
        </div>
        '''
