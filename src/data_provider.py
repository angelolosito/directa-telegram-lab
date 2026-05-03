from __future__ import annotations

import time
from datetime import datetime, timedelta
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from queue import Empty
from zoneinfo import ZoneInfo

import pandas as pd

from .indicators import add_indicators


class DataProviderError(RuntimeError):
    pass


def _safe_cache_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=", "_").replace(":", "_")


def _cache_path(cache_dir: str | Path | None, symbol: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{_safe_cache_name(symbol)}.csv"


def _read_cached_daily(cache_dir: str | Path | None, symbol: str) -> pd.DataFrame | None:
    path = _cache_path(cache_dir, symbol)
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    return df if not df.empty else None


def _write_cached_daily(cache_dir: str | Path | None, symbol: str, df: pd.DataFrame) -> None:
    path = _cache_path(cache_dir, symbol)
    if path is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


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
    retries: int = 2,
    process_timeout: int = 20,
    retry_backoff_seconds: float = 3.0,
    cache_dir: str | Path | None = None,
    use_cache_on_failure: bool = True,
    fallback_symbols: list[str] | None = None,
    prefer_cache: bool = False,
    cache_only: bool = False,
) -> pd.DataFrame:
    tz = ZoneInfo(timezone)
    end = datetime.now(tz=tz).date() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    attempts = max(1, retries + 1)
    deadline_seconds = max(process_timeout, request_timeout)
    source_symbols = [symbol]
    for fallback in fallback_symbols or []:
        if fallback and fallback not in source_symbols:
            source_symbols.append(fallback)
    source_errors: list[str] = []
    df = pd.DataFrame()
    used_symbol = symbol

    if prefer_cache or cache_only:
        for source_symbol in source_symbols:
            cached = _read_cached_daily(cache_dir, source_symbol)
            if cached is not None:
                df = cached
                used_symbol = source_symbol
                break
        if df.empty and cache_only:
            fallback_text = (
                f" Fallback cercati in cache: {', '.join(source_symbols[1:])}."
                if len(source_symbols) > 1
                else ""
            )
            raise DataProviderError(
                f"{symbol}: dato non presente in cache; download live disattivato.{fallback_text}"
            )

    if df.empty:
        for source_symbol in source_symbols:
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    df = _download_with_deadline(
                        symbol=source_symbol,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        request_timeout=request_timeout,
                        deadline_seconds=deadline_seconds,
                    )
                    if not df.empty:
                        used_symbol = source_symbol
                        break
                    last_error = DataProviderError(f"Nessun dato ricevuto per {source_symbol}")
                except Exception as e:  # noqa: BLE001
                    last_error = e

                if attempt == attempts:
                    cached = _read_cached_daily(cache_dir, source_symbol) if use_cache_on_failure else None
                    if cached is not None:
                        df = cached
                        used_symbol = source_symbol
                        break
                    source_errors.append(f"{source_symbol}: {last_error}")
                    break
                time.sleep(retry_backoff_seconds * attempt)
            if not df.empty:
                break

    if df.empty:
        fallback_text = f" Fallback provati: {', '.join(source_symbols[1:])}." if len(source_symbols) > 1 else ""
        raise DataProviderError(
            f"{symbol}: download dati non riuscito dopo {attempts} tentativi.{fallback_text} "
            f"Dettagli: {'; '.join(source_errors) or 'nessun dato ricevuto'}"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected.difference(df.columns)
    if missing:
        raise DataProviderError(f"Dati incompleti per {symbol}. Mancano: {sorted(missing)}")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    _write_cached_daily(cache_dir, used_symbol, df)
    if used_symbol != symbol:
        _write_cached_daily(cache_dir, symbol, df)
    result = add_indicators(df)
    result.attrs["source_symbol"] = used_symbol
    return result
