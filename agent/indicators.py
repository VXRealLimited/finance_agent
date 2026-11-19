import pandas as pd


class IndicatorEngine:
    # Registry defining supported indicators and how to compute their warmup bars
    SUPPORTED_INDICATORS = {
        "SMA": lambda p: p.get("period", 20),
        "EMA": lambda p: p.get("period", 20) * 2,  # Extra buffer for EMA convergence
        "RSI": lambda p: p.get("period", 14) + 1,
        "MACD": lambda p: p.get("slow", 26) + p.get("signal", 9),
        "BBANDS": lambda p: p.get("period", 20),
    }

    @classmethod
    def get_required_warmup(cls, indicator_configs: list[dict]) -> int:
        """Calculates the maximum historical bars needed before the backtest start date."""
        if not indicator_configs:
            return 0
        max_bars = 0
        for cfg in indicator_configs:
            ind_type = cfg["type"].upper()
            if ind_type not in cls.SUPPORTED_INDICATORS:
                raise ValueError(f"Unsupported indicator: {ind_type}")
            warmup = cls.SUPPORTED_INDICATORS[ind_type](cfg.get("params", {}))
            max_bars = max(max_bars, warmup)
        return max_bars

    @classmethod
    def apply_indicators(
        cls, df: pd.DataFrame, indicator_configs: list[dict]
    ) -> pd.DataFrame:
        """Applies standardized indicator columns to an OHLCV DataFrame."""
        df = df.copy()
        for cfg in indicator_configs:
            ind_type = cfg["type"].upper()
            params = cfg.get("params", {})
            col_name = cfg.get("name")

            if ind_type == "SMA":
                period = params.get("period", 20)
                name = col_name or f"SMA_{period}"
                df[name] = df["close"].rolling(window=period).mean()

            elif ind_type == "EMA":
                period = params.get("period", 20)
                name = col_name or f"EMA_{period}"
                df[name] = df["close"].ewm(span=period, adjust=False).mean()

            elif ind_type == "RSI":
                period = params.get("period", 14)
                name = col_name or f"RSI_{period}"
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[name] = 100 - (100 / (1 + rs))

            elif ind_type == "MACD":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
                ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
                df[f"{col_name or 'MACD'}_line"] = ema_fast - ema_slow
                df[f"{col_name or 'MACD'}_signal"] = (
                    df[f"{col_name or 'MACD'}_line"]
                    .ewm(span=signal, adjust=False)
                    .mean()
                )

            elif ind_type == "BBANDS":
                period = params.get("period", 20)
                std_dev = params.get("std", 2)
                sma = df["close"].rolling(window=period).mean()
                std = df["close"].rolling(window=period).std()
                df[f"{col_name or 'BB'}_upper"] = sma + (std * std_dev)
                df[f"{col_name or 'BB'}_lower"] = sma - (std * std_dev)

        return df
