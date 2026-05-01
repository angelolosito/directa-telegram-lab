from __future__ import annotations

from datetime import datetime, timedelta
from multiprocessing import get_all_start_methods, get_context
from queue import Empty
from zoneinfo import ZoneInfo

import pandas as pd

from .indicators import add_indicators


class DataProviderError(RuntimeError):
    pass


def _download_worker(
    queue,
    symbol: str,
    start: str,
    end: str,
    request_timeout: int,
) -> None:
    try:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=request_timeout,
        )
        queue.put(("ok", df))
    except Exception as e:  # noqa: BLE001
        queue.put(("error", repr(e)))


def _download_with_deadline(
    symbol: str,
    start: str,
    end: str,
    request_timeout: int,
    deadline_seconds: int,
) -> pd.DataFrame:
    start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
    ctx = get_context(start_method)
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_download_worker,
        args=(queue, symbol, start, end, request_timeout),
    )
    process.start()
    process.join(deadline_seconds)

    if process.is_alive():
        process.terminate()
        process.join(2)
        queue.close()
        raise TimeoutError(f"download oltre {deadline_seconds} secondi")

    try:
        status, payload = queue.get(timeout=1)
    except Empty as e:
        raise RuntimeError(f"download terminato senza dati, exit code {process.exitcode}") from e
    finally:
        queue.close()

    if status == "error":
        raise RuntimeError(payload)
    return payload


def fetch_daily_data(
    symbol: str,
    lookback_days: int,
    timezone: str,
    request_timeout: int = 8,
    retries: int = 1,
    process_timeout: int = 20,
) -> pd.DataFrame:
    tz = ZoneInfo(timezone)
    end = datetime.now(tz=tz).date() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    attempts = max(1, retries + 1)
    deadline_seconds = max(process_timeout, request_timeout)
    last_error: Exception | None = None
    df = pd.DataFrame()

    for attempt in range(1, attempts + 1):
        try:
            df = _download_with_deadline(
                symbol=symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                request_timeout=request_timeout,
                deadline_seconds=deadline_seconds,
            )
            if not df.empty:
                break
            last_error = DataProviderError(f"Nessun dato ricevuto per {symbol}")
        except Exception as e:  # noqa: BLE001
            last_error = e

        if attempt == attempts:
            raise DataProviderError(
                f"{symbol}: download dati non riuscito dopo {attempts} tentativi: {last_error}"
            ) from last_error

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
