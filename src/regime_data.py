from __future__ import annotations

from pathlib import Path

from .data_provider import DataProviderError, fetch_daily_data
from .market_regime import configured_benchmarks


def _regime_clean_rows(df) -> int:
    if df is None:
        return 0
    required = {"Close", "SMA50", "SMA200"}
    if not required.issubset(df.columns):
        return 0
    return len(df.dropna(subset=["Close", "SMA50", "SMA200"]))


def fetch_market_regime_data(
    cfg: dict,
    known_market_data: dict,
    timezone: str,
    lookback_days: int,
    request_timeout: int,
    download_retries: int,
    process_timeout: int,
    retry_backoff_seconds: float = 3.0,
    cache_dir: Path | None = None,
    use_cache_on_failure: bool = True,
) -> tuple[dict, list[str]]:
    regime_data = {}
    errors: list[str] = []
    benchmarks = configured_benchmarks(cfg)
    if not benchmarks:
        return regime_data, errors

    regime_cfg = cfg.get("market_regime", {})
    regime_lookback_days = int(regime_cfg.get("lookback_days", max(lookback_days, 320)))
    min_rows = int(regime_cfg.get("min_rows_required", 220))
    min_usable_rows = int(regime_cfg.get("min_usable_rows_required", 30))

    for benchmark in benchmarks:
        symbol = benchmark["symbol"]
        known_df = known_market_data.get(symbol)
        known_clean_rows = _regime_clean_rows(known_df)
        if known_df is not None and known_clean_rows >= min_rows:
            regime_data[symbol] = known_df
            continue

        try:
            df = fetch_daily_data(
                symbol,
                lookback_days=regime_lookback_days,
                timezone=timezone,
                request_timeout=request_timeout,
                retries=download_retries,
                process_timeout=process_timeout,
                retry_backoff_seconds=retry_backoff_seconds,
                cache_dir=cache_dir,
                use_cache_on_failure=use_cache_on_failure,
            )
            clean_rows = _regime_clean_rows(df)
            if clean_rows >= min_rows:
                regime_data[symbol] = df
            elif clean_rows >= min_usable_rows:
                errors.append(
                    f"{symbol}: storico regime ridotto ({clean_rows} righe valide), "
                    f"ma sufficiente per classificare il mercato."
                )
                regime_data[symbol] = df
            elif known_df is not None and known_clean_rows >= min_usable_rows:
                errors.append(
                    f"{symbol}: uso storico gia disponibile per il regime "
                    f"({known_clean_rows} righe valide) dopo refetch insufficiente."
                )
                regime_data[symbol] = known_df
            else:
                errors.append(f"{symbol}: storico insufficiente per filtro regime mercato.")
        except DataProviderError as e:
            if known_df is not None and known_clean_rows >= min_usable_rows:
                errors.append(
                    f"{symbol}: download regime fallito, uso storico gia disponibile "
                    f"({known_clean_rows} righe valide). Dettaglio: {e}"
                )
                regime_data[symbol] = known_df
            else:
                errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            if known_df is not None and known_clean_rows >= min_usable_rows:
                errors.append(
                    f"{symbol}: errore nel refetch regime, uso storico gia disponibile "
                    f"({known_clean_rows} righe valide). Dettaglio: {e}"
                )
                regime_data[symbol] = known_df
            else:
                errors.append(f"{symbol}: errore imprevisto nel filtro regime mercato: {e}")
    return regime_data, errors
