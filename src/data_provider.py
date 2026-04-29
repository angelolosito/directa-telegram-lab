from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .indicators import add_indicators


class DataProviderError(RuntimeError):
    pass


def fetch_daily_data(symbol: str, lookback_days: int, timezone: str) -> pd.DataFrame:
    tz = ZoneInfo(timezone)
    end = datetime.now(tz=tz).date() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise DataProviderError(f"Nessun dato ricevuto per {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected.difference(df.columns)
    if missing:
        raise DataProviderError(f"Dati incompleti per {symbol}. Mancano: {sorted(missing)}")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    return add_indicators(df)
