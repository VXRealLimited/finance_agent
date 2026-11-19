import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Tuple
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Import the modular backtesting & data components
from data_manager import MarketDataManager
from indicators import IndicatorEngine
from backtester import Backtester

# In-memory store for the active crypto trading strategy configuration.
STRATEGY_CONFIG: Dict = {
    "pair": None,
    "timeframe": "4h",
    "buy_signals": [],
    "sell_signals": [],
}

# --- Helper Functions for Indicator Standardization & Parsing ---

BASE_PRICE_COLS = {
    "PRICE": "close",
    "CLOSE": "close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "VOLUME": "volume",
}


def _parse_indicator_token(token: str) -> Tuple[str, dict | None]:
    """
    Maps a standardized agent token (e.g., 'SMA_250', 'RSI_14', 'MACD_LINE', 'PRICE')
    to its DataFrame column name and IndicatorEngine configuration dict.
    """
    token = token.strip().upper()

    # Check if it's a standard OHLCV price/volume column
    if token in BASE_PRICE_COLS:
        return BASE_PRICE_COLS[token], None

    # Check if it's a numeric threshold (e.g., '30', '70.5')
    try:
        float(token)
        return token, None
    except ValueError:
        pass

    # Match pattern: TYPE_PERIOD (e.g., SMA_250, EMA_20, RSI_14)
    match = re.match(r"^(SMA|EMA|RSI)_(\d+)$", token)
    if match:
        ind_type, period = match.group(1), int(match.group(2))
        return token, {"type": ind_type, "params": {"period": period}, "name": token}

    # Handle default shorthand without period (e.g., 'RSI' -> 'RSI_14', 'SMA' -> 'SMA_20')
    if token == "RSI":
        return "RSI_14", {"type": "RSI", "params": {"period": 14}, "name": "RSI_14"}
    if token == "SMA":
        return "SMA_20", {"type": "SMA", "params": {"period": 20}, "name": "SMA_20"}
    if token == "EMA":
        return "EMA_20", {"type": "EMA", "params": {"period": 20}, "name": "EMA_20"}

    # Handle MACD components
    if token in ("MACD", "MACD_LINE", "MACD_SIGNAL"):
        col = "MACD_signal" if token == "MACD_SIGNAL" else "MACD_line"
        return col, {
            "type": "MACD",
            "params": {"fast": 12, "slow": 26, "signal": 9},
            "name": "MACD",
        }

    # Handle Bollinger Bands components
    if token in ("BB_UPPER", "BB_LOWER"):
        col = "BB_upper" if token == "BB_UPPER" else "BB_lower"
        return col, {"type": "BBANDS", "params": {"period": 20, "std": 2}, "name": "BB"}

    raise ValueError(
        f"Unsupported indicator '{token}'. Supported formats: "
        f"SMA_<period>, EMA_<period>, RSI_<period>, MACD_LINE, MACD_SIGNAL, BB_UPPER, BB_LOWER, PRICE, VOLUME."
    )


def _extract_indicator_configs(rules: List[dict]) -> List[dict]:
    """Collects unique IndicatorEngine configs from all buy and sell rules."""
    configs_by_name = {}
    for rule in rules:
        for field in (rule["indicator"], rule["value"]):
            _, cfg = _parse_indicator_token(field)
            if cfg:
                configs_by_name[cfg["name"]] = cfg
    return list(configs_by_name.values())


def _parse_period_to_days(period_str: str) -> int:
    """Converts strings like '180d', '1y', '30d' into integer days."""
    period_str = period_str.strip().lower()
    if period_str.endswith("d"):
        return int(period_str[:-1])
    if period_str.endswith("y"):
        return int(period_str[:-1]) * 365
    if period_str.endswith("w") or period_str.endswith("wk"):
        return int(re.sub(r"\D", "", period_str)) * 7
    return int(period_str)


# --- Pydantic Input Schemas for Strict Tool Calling ---


class SetCryptoPairInput(BaseModel):
    pair: str = Field(
        ...,
        description="The cryptocurrency trading pair only (e.g., 'BTC/USDT', 'ETH/USDT', 'SOLUSDT'). Stocks and forex are not supported.",
    )
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d", "1wk"] = Field(
        default="4h",
        description="Candlestick timeframe for evaluating crypto market signals.",
    )


class SignalRuleInput(BaseModel):
    indicator: str = Field(
        ...,
        description=(
            "Standardized technical indicator or price metric. Supported formats: "
            "'PRICE', 'VOLUME', 'SMA_<period>' (e.g., 'SMA_250'), 'EMA_<period>' (e.g., 'EMA_20'), "
            "'RSI_<period>' (e.g., 'RSI_14'), 'MACD_LINE', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER'."
        ),
    )
    operator: Literal["<", "<=", ">", ">=", "==", "crosses_above", "crosses_below"] = (
        Field(
            ...,
            description="Comparison operator between the indicator and target value.",
        )
    )
    value: str = Field(
        ...,
        description="Threshold number (e.g., '30', '70') or standardized comparison indicator (e.g., 'SMA_250', 'EMA_200', 'BB_LOWER').",
    )
    description: str = Field(
        ...,
        description="Human-readable summary of this rule (e.g., 'Price crosses above 250D SMA').",
    )


class RunBacktestInput(BaseModel):
    period: str = Field(
        default="180d",
        description="Historical lookback duration to test (e.g., '90d', '180d', '1y').",
    )
    initial_capital: float = Field(
        default=10000.0,
        description="Starting capital in USDT for the simulation.",
    )


# --- Agent Tools ---


@tool(args_schema=SetCryptoPairInput)
def set_crypto_pair(pair: str, timeframe: str = "4h") -> str:
    """Set or update the target cryptocurrency trading pair and timeframe for the strategy. Only crypto pairs are supported."""
    # Normalize pair (e.g. 'BTC/USDT' or 'BTC-USDT' -> 'BTCUSDT')
    normalized_pair = pair.upper().replace("/", "").replace("-", "").strip()
    # Normalize timeframe for Binance compatibility
    normalized_tf = "1w" if timeframe == "1wk" else timeframe

    STRATEGY_CONFIG["pair"] = normalized_pair
    STRATEGY_CONFIG["timeframe"] = normalized_tf
    return f"Target crypto pair set to {STRATEGY_CONFIG['pair']} on the {normalized_tf} timeframe."


@tool(args_schema=SignalRuleInput)
def add_buy_signal(indicator: str, operator: str, value: str, description: str) -> str:
    """Add a long entry/buy condition to the crypto trading strategy."""
    try:
        # Validate tokens against our standardized indicator registry immediately
        ind_col, _ = _parse_indicator_token(indicator)
        val_col, _ = _parse_indicator_token(value)
    except ValueError as e:
        return f"Invalid indicator in buy signal: {str(e)}"

    rule = {
        "indicator": indicator.upper().strip(),
        "indicator_col": ind_col,
        "operator": operator,
        "value": value.upper().strip(),
        "value_col": val_col,
        "description": description,
    }
    STRATEGY_CONFIG["buy_signals"].append(rule)
    return f"Added BUY signal: {description} ({rule['indicator']} {operator} {rule['value']})."


@tool(args_schema=SignalRuleInput)
def add_sell_signal(indicator: str, operator: str, value: str, description: str) -> str:
    """Add an exit/sell condition to the crypto trading strategy."""
    try:
        # Validate tokens against our standardized indicator registry immediately
        ind_col, _ = _parse_indicator_token(indicator)
        val_col, _ = _parse_indicator_token(value)
    except ValueError as e:
        return f"Invalid indicator in sell signal: {str(e)}"

    rule = {
        "indicator": indicator.upper().strip(),
        "indicator_col": ind_col,
        "operator": operator,
        "value": value.upper().strip(),
        "value_col": val_col,
        "description": description,
    }
    STRATEGY_CONFIG["sell_signals"].append(rule)
    return f"Added SELL signal: {description} ({rule['indicator']} {operator} {rule['value']})."


@tool
def get_current_strategy() -> str:
    """Retrieve the currently configured crypto pair, buy signals, and sell signals."""
    pair = STRATEGY_CONFIG["pair"] or "Not set"
    timeframe = STRATEGY_CONFIG["timeframe"]
    buys = STRATEGY_CONFIG["buy_signals"]
    sells = STRATEGY_CONFIG["sell_signals"]

    buy_text = (
        "\n  - ".join(
            [
                f"{r['description']} [{r['indicator']} {r['operator']} {r['value']}]"
                for r in buys
            ]
        )
        if buys
        else "None configured"
    )
    sell_text = (
        "\n  - ".join(
            [
                f"{r['description']} [{r['indicator']} {r['operator']} {r['value']}]"
                for r in sells
            ]
        )
        if sells
        else "None configured"
    )

    return (
        f"Current Crypto Strategy Configuration:\n"
        f"• Crypto Pair: {pair} ({timeframe})\n"
        f"• Buy Signals:\n  - {buy_text}\n"
        f"• Sell Signals:\n  - {sell_text}"
    )


@tool
def clear_signals(signal_type: Literal["buy", "sell", "all"] = "all") -> str:
    """Clear existing buy signals, sell signals, or both to reset the crypto strategy."""
    if signal_type in ("buy", "all"):
        STRATEGY_CONFIG["buy_signals"].clear()
    if signal_type in ("sell", "all"):
        STRATEGY_CONFIG["sell_signals"].clear()
    return f"Cleared {signal_type} signal(s) from the crypto strategy."


@tool(args_schema=RunBacktestInput)
def run_backtest(period: str = "180d", initial_capital: float = 10000.0) -> str:
    """Run a historical crypto backtest using MongoDB cached OHLCV data (lazy-fetching from Binance if needed) and the configured buy/sell signals."""
    if not STRATEGY_CONFIG["pair"]:
        return "Cannot run backtest: No crypto trading pair set. Ask the user which crypto pair (e.g., BTC/USDT) to test."
    if not STRATEGY_CONFIG["buy_signals"] or not STRATEGY_CONFIG["sell_signals"]:
        return "Cannot run backtest: Both a buy signal and a sell signal must be configured first."

    symbol = STRATEGY_CONFIG["pair"]
    interval = STRATEGY_CONFIG["timeframe"]
    buy_rules = STRATEGY_CONFIG["buy_signals"]
    sell_rules = STRATEGY_CONFIG["sell_signals"]

    try:
        # 1. Determine time window
        days = _parse_period_to_days(period)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        # 2. Extract standardized indicator configs & compute required warm-up bars (e.g. 250 bars for SMA_250)
        all_rules = buy_rules + sell_rules
        indicator_configs = _extract_indicator_configs(all_rules)
        warmup_bars = IndicatorEngine.get_required_warmup(indicator_configs)

        # 3. Lazy-load from local MongoDB (automatically fetches missing candles from Binance)
        data_mgr = MarketDataManager(mongo_uri="mongodb://localhost:27017/")
        raw_df = data_mgr.get_ohlcv(
            symbol=symbol,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
            warmup_bars=warmup_bars,
        )

        if raw_df.empty:
            return (
                f"Backtest failed: No OHLCV data returned for {symbol} on {interval}."
            )

        # 4. Compute standardized indicators across warm-up + test window
        enriched_df = IndicatorEngine.apply_indicators(raw_df, indicator_configs)

        # 5. Execute isolated backtest engine with dynamic rules
        engine = Backtester(initial_capital=initial_capital, fee_rate=0.001)
        results = engine.run(
            df=enriched_df,
            backtest_start_dt=pd.Timestamp(start_dt),
            buy_rules=buy_rules,
            sell_rules=sell_rules,
        )

        # 6. Format summary for the agent
        recent_trades = results["trades"][-5:]
        trades_summary = (
            "\n".join(
                [
                    f"  - {t['type']} @ ${t['price']:,.2f} ({t['datetime'][:16]})"
                    for t in recent_trades
                ]
            )
            or "  - No trades executed during this period."
        )

        return (
            f"Backtest Results for {symbol} ({interval} | Last {period}):\n"
            f"• Initial Capital: ${results['initial_capital']:,.2f}\n"
            f"• Final Equity: ${results['final_equity']:,.2f}\n"
            f"• Total Return: {results['total_return_pct']}%\n"
            f"• Max Drawdown: {results['max_drawdown_pct']}%\n"
            f"• Total Trades: {results['total_trades']}\n"
            f"• Warmup Bars Used: {warmup_bars} ({interval} candles)\n"
            f"• Latest Trades:\n{trades_summary}"
        )

    except Exception as e:
        return f"Backtest execution error: {str(e)}"


# Export list of crypto strategy tools to bind to Ollama / LangGraph ToolNode
AGENT_TOOLS = [
    set_crypto_pair,
    add_buy_signal,
    add_sell_signal,
    get_current_strategy,
    clear_signals,
    run_backtest,
]
