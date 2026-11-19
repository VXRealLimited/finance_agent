import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

    def _resolve_operand(self, df: pd.DataFrame, token: str) -> pd.Series:
        """
        Resolves a standardized column name (e.g., 'SMA_250', 'RSI_14', 'close')
        or a numeric threshold string (e.g., '30', '70.5') into a pandas Series.
        """
        if token in df.columns:
            return df[token]
        try:
            numeric_val = float(token)
            return pd.Series(numeric_val, index=df.index)
        except ValueError:
            raise ValueError(
                f"Indicator column '{token}' not found in DataFrame and is not a valid number."
            )

    def _evaluate_rule_set(
        self, df: pd.DataFrame, rules: list[dict], combine_mode: str = "AND"
    ) -> pd.Series:
        """
        Evaluates a list of standardized rule dictionaries into a single boolean Series.
        Each rule dict expects: {'indicator_col': str, 'operator': str, 'value_col': str}
        """
        if not rules:
            return pd.Series(False, index=df.index)

        if combine_mode.upper() == "AND":
            combined = pd.Series(True, index=df.index)
        elif combine_mode.upper() == "OR":
            combined = pd.Series(False, index=df.index)
        else:
            raise ValueError(
                f"Unsupported combine_mode: {combine_mode}. Use 'AND' or 'OR'."
            )

        for rule in rules:
            left = self._resolve_operand(df, rule["indicator_col"])
            right = self._resolve_operand(df, rule["value_col"])
            op = rule["operator"]

            if op == "<":
                cond = left < right
            elif op == "<=":
                cond = left <= right
            elif op == ">":
                cond = left > right
            elif op == ">=":
                cond = left >= right
            elif op == "==":
                cond = np.isclose(left, right)
            elif op == "crosses_above":
                cond = (left > right) & (left.shift(1) <= right.shift(1))
            elif op == "crosses_below":
                cond = (left < right) & (left.shift(1) >= right.shift(1))
            else:
                raise ValueError(f"Unsupported operator: '{op}'")

            # Prevent NaN warmup rows from triggering false positives
            cond = pd.Series(cond, index=df.index).fillna(False)

            if combine_mode.upper() == "AND":
                combined = combined & cond
            else:
                combined = combined | cond

        return combined

    def run(
        self,
        df: pd.DataFrame,
        backtest_start_dt: pd.Timestamp,
        buy_rules: list[dict],
        sell_rules: list[dict],
        buy_mode: str = "AND",
        sell_mode: str = "AND",
    ) -> dict:
        """
        Runs a generic long-only backtest using any dynamic list of buy and sell rules.
        """
        if not buy_rules or not sell_rules:
            raise ValueError(
                "Both buy_rules and sell_rules must contain at least one rule."
            )

        df = df.copy()

        # Vectorized signal evaluation across full DataFrame (including warmup for accurate crossovers)
        df["_buy_signal"] = self._evaluate_rule_set(
            df, buy_rules, combine_mode=buy_mode
        )
        df["_sell_signal"] = self._evaluate_rule_set(
            df, sell_rules, combine_mode=sell_mode
        )

        # Trim warmup period so simulation runs strictly inside the requested test window
        sim_df = df[df.index >= backtest_start_dt].copy()

        if sim_df.empty:
            raise ValueError("No data available in the specified backtest date range.")

        capital = self.initial_capital
        position = 0.0
        entry_cost = 0.0
        trades = []
        round_trip_returns = []
        equity_curve = []

        for timestamp, row in sim_df.iterrows():
            price = row["close"]

            # Entry Execution
            if position == 0 and row["_buy_signal"]:
                entry_cost = capital
                position = (capital * (1 - self.fee_rate)) / price
                capital = 0.0
                trades.append(
                    {
                        "type": "BUY",
                        "datetime": str(timestamp),
                        "price": price,
                        "size": position,
                    }
                )

            # Exit Execution
            elif position > 0 and row["_sell_signal"]:
                capital = (position * price) * (1 - self.fee_rate)
                pnl_pct = ((capital - entry_cost) / entry_cost) * 100
                round_trip_returns.append(pnl_pct)
                trades.append(
                    {
                        "type": "SELL",
                        "datetime": str(timestamp),
                        "price": price,
                        "capital": capital,
                        "pnl_pct": round(pnl_pct, 2),
                    }
                )
                position = 0.0
                entry_cost = 0.0

            current_equity = capital + (position * price)
            equity_curve.append({"datetime": timestamp, "equity": current_equity})

        # Performance Summary Metrics
        final_equity = equity_curve[-1]["equity"]
        total_return_pct = (
            (final_equity - self.initial_capital) / self.initial_capital
        ) * 100

        eq_series = pd.Series([e["equity"] for e in equity_curve])
        rolling_max = eq_series.cummax()
        max_drawdown_pct = (((eq_series - rolling_max) / rolling_max).min()) * 100

        winning_trades = [r for r in round_trip_returns if r > 0]
        win_rate_pct = (
            (len(winning_trades) / len(round_trip_returns) * 100)
            if round_trip_returns
            else 0.0
        )

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "completed_round_trips": len(round_trip_returns),
            "total_trades": len(trades),
            "trades": trades,
            "equity_curve": [
                {"datetime": e["datetime"].isoformat(), "equity": round(e["equity"], 2)}
                for e in equity_curve
            ],
        }
