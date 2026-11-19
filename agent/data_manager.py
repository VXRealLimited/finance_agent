import requests
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class MarketDataManager:
    def __init__(
        self, mongo_uri="mongodb://localhost:27017/", db_name="crypto_backtest"
    ):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["ohlcv"]
        # Ensure fast lookups and prevent duplicate candles across intervals
        self.collection.create_index(
            [("symbol", 1), ("interval", 1), ("open_time", 1)], unique=True
        )

    def _fetch_from_binance(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[dict]:
        """Fetches klines from Binance public API in chunks of 1000."""
        url = "https://api.binance.com/api/v3/klines"
        all_records = []
        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            for k in data:
                all_records.append(
                    {
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": int(k[6]),
                    }
                )

            # Advance start time to the next candle after the last fetched one
            current_start = int(data[-1][0]) + INTERVAL_MS[interval]

        return all_records

    def _save_to_mongo(self, records: list[dict]):
        if not records:
            return
        operations = [
            UpdateOne(
                {
                    "symbol": r["symbol"],
                    "interval": r["interval"],
                    "open_time": r["open_time"],
                },
                {"$set": r},
                upsert=True,
            )
            for r in records
        ]
        self.collection.bulk_write(operations)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        warmup_bars: int = 0,
    ) -> pd.DataFrame:
        """
        Lazy-loads OHLCV data including warmup bars prior to start_dt.
        Fetches missing ranges from Binance and caches them in MongoDB.
        """
        if interval not in INTERVAL_MS:
            raise ValueError(f"Unsupported interval: {interval}")

        step_ms = INTERVAL_MS[interval]
        start_ms = int(start_dt.timestamp() * 1000) - (warmup_bars * step_ms)
        end_ms = int(end_dt.timestamp() * 1000)

        # Query existing data in MongoDB
        query = {
            "symbol": symbol.upper(),
            "interval": interval,
            "open_time": {"$gte": start_ms, "$lte": end_ms},
        }
        existing = list(self.collection.find(query, {"_id": 0}).sort("open_time", 1))

        expected_count = ((end_ms - start_ms) // step_ms) + 1

        # Lazy check: If data is missing or incomplete, sync from Binance
        if len(existing) < expected_count:
            if not existing:
                fetched = self._fetch_from_binance(symbol, interval, start_ms, end_ms)
                self._save_to_mongo(fetched)
            else:
                # Check leading gap (e.g., if warmup_bars pushed start_ms earlier than cached data)
                first_cached = existing[0]["open_time"]
                if first_cached > start_ms + step_ms:
                    fetched_pre = self._fetch_from_binance(
                        symbol, interval, start_ms, first_cached - step_ms
                    )
                    self._save_to_mongo(fetched_pre)

                # Check trailing gap
                last_cached = existing[-1]["open_time"]
                if last_cached < end_ms - step_ms:
                    fetched_post = self._fetch_from_binance(
                        symbol, interval, last_cached + step_ms, end_ms
                    )
                    self._save_to_mongo(fetched_post)

            # Re-query complete dataset
            existing = list(
                self.collection.find(query, {"_id": 0}).sort("open_time", 1)
            )

        df = pd.DataFrame(existing)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)
        return df
