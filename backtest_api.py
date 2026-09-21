"""HTTP integration and historical data adapter for backtesting."""
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backtest import BACKTEST_MARKETS, BacktestError, BacktestRequest, prepare_data, run_backtest, ComparisonRequest, comparison_base, run_comparison

router = APIRouter()


class DataProviderError(RuntimeError):
    pass


@router.get("/api/ticker-search")
def ticker_search(query: str, market: str = "US"):
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a company name or ticker.")
    suffixes = {"US": "", "HK": ".HK", "CN_SH": ".SS", "CN_SZ": ".SZ"}
    if market not in suffixes:
        raise HTTPException(status_code=422, detail="Unsupported market.")
    try:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v1/finance/search?q={quote(query)}",
            params={"quotesCount": 20, "newsCount": 0},
            timeout=10,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Ticker search is temporarily unavailable. Try a ticker symbol instead.") from exc

    suffix = suffixes[market]
    matches = []
    for item in quotes:
        symbol = str(item.get("symbol", "")).upper()
        if not symbol or (suffix and not symbol.endswith(suffix)):
            continue
        if market == "US" and "." in symbol:
            continue
        matches.append({
            "symbol": symbol,
            "name": item.get("shortname") or item.get("longname") or symbol,
        })
    if not matches:
        raise HTTPException(status_code=404, detail=f"No {market} ticker was found for '{query}'.")
    return {"matches": matches[:5]}


def download_data(req):
    def loader(symbol, start, end):
        try:
            ticker = yf.Ticker(symbol)
            frame = ticker.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True, raise_errors=True)
        except Exception as exc:
            raise DataProviderError(f"Could not retrieve historical data for {symbol}. Retry later.") from exc
        close = frame.get("Close", pd.Series(dtype=float, index=pd.DatetimeIndex([]))).dropna()
        if close.empty:
            raise DataProviderError(f"No historical price data was found for {symbol} in the requested window.")
        if symbol in req.tickers:
            metadata = ticker.get_history_metadata()
            expected_currency = BACKTEST_MARKETS[req.market]["currency"]
            if metadata.get("currency") != expected_currency:
                raise BacktestError(f"{symbol}: requires verified {expected_currency} price data")
        return close
    return prepare_data(req, loader)


@router.post("/api/backtest")
def backtest_endpoint(req: BacktestRequest):
    try:
        data = download_data(req)
    except BacktestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Historical data provider unavailable. Retry later or use a saved snapshot with the CLI.") from exc
    try:
        return run_backtest(req, *data)
    except BacktestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/backtest", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "backtest.html")


@router.post("/api/backtest/compare")
def comparison_endpoint(req: ComparisonRequest):
    # Download once over the longest window so each run sees the same prices.
    data_req = comparison_base(req, max(req.lookbacks), req.frequencies[0], 1.0)
    try:
        data = download_data(data_req)
        return run_comparison(req, *data)
    except BacktestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
