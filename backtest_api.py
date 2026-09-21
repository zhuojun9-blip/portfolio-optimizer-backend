"""HTTP integration and historical data adapter for backtesting."""
from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backtest import BacktestError, BacktestRequest, prepare_data, run_backtest

router = APIRouter()


def download_data(req):
    def loader(symbol, start, end):
        ticker = yf.Ticker(symbol)
        frame = ticker.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True, raise_errors=True)
        if symbol in req.tickers:
            metadata = ticker.get_history_metadata()
            if metadata.get("currency") != "USD":
                raise BacktestError(f"{symbol}: this version requires verified USD price data")
        return frame.get("Close", pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    return prepare_data(req, loader)


@router.post("/api/backtest")
def backtest_endpoint(req: BacktestRequest):
    try:
        data = download_data(req)
    except BacktestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Historical data provider unavailable. Retry later or use a saved snapshot with the CLI.") from exc
    try:
        return run_backtest(req, *data)
    except BacktestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/backtest", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "backtest.html")
