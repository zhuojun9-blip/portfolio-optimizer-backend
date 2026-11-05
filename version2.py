# backend/app.py
import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import List, Literal, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import statsmodels.api as sm

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# ---- Optional: SciPy for constrained optimization ----
try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


MARKET_BENCHMARKS = {
    "US": "^GSPC",
    "HK": "^HSI",
    "CN_SH": "000001.SS",
    "CN_SZ": "399001.SZ",
    "JP": "^N225",
    "CA": "^GSPTSE",
    "UK": "^FTSE",
    "IN": "^NSEI",
}

MARKET_COUNTRIES = {
    "HK": "hong kong",
    "CN_SH": "china",
    "CN_SZ": "china",
    "JP": "japan",
    "CA": "canada",
    "UK": "united kingdom",
    "IN": "india",
}


def bounded_close(symbol: str, start: date, end: date) -> pd.Series:
    """Download and enforce [start, end) using the exchange-local calendar date."""
    history = yf.Ticker(symbol).history(
        start=start.isoformat(), end=end.isoformat(), auto_adjust=True
    )
    close = history.get("Close", pd.Series(dtype=float)).copy()
    if close.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    close = close.sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    return close.loc[(close.index >= pd.Timestamp(start)) & (close.index < pd.Timestamp(end))]


def latest_local_government_yield(market: str) -> float:
    """Return the latest 10-year government bond yield as a decimal."""
    api_key = os.getenv("TRADING_ECONOMICS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Local risk-free-rate data is unavailable. Configure TRADING_ECONOMICS_API_KEY on the server.",
        )

    country = MARKET_COUNTRIES[market]
    try:
        response = requests.get(
            f"https://api.tradingeconomics.com/markets/bonds/country/{country}",
            params={"c": api_key},
            timeout=10,
        )
        response.raise_for_status()
        quotes = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not retrieve the local government yield for {country}: {exc}",
        )

    for quote in quotes if isinstance(quotes, list) else []:
        name = str(quote.get("Name", "")).lower()
        value = quote.get("Last")
        if "10 year" in name and "government" in name and isinstance(value, (int, float)) and math.isfinite(value):
            return float(value) / 100.0

    raise HTTPException(
        status_code=503,
        detail=f"No current 10-year government yield is available for {country}.",
    )


# ----------------- helpers from your script -----------------
def build_returns_aligned(stock_close: pd.Series, market_close: pd.Series) -> pd.DataFrame:
    s1 = stock_close.dropna().copy()
    s2 = market_close.dropna().copy()
    if getattr(s1.index, "tz", None) is not None:
        s1.index = s1.index.tz_localize(None)
    if getattr(s2.index, "tz", None) is not None:
        s2.index = s2.index.tz_localize(None)
    s1.index = s1.index.normalize()
    s2.index = s2.index.normalize()
    s1.name, s2.name = "stock", "market"
    both = pd.concat([s1, s2], axis=1, join="inner").dropna()
    return both.pct_change().dropna()

def annualize_cov(cov_daily: np.ndarray, periods: int = 252) -> np.ndarray:
    return cov_daily * periods

def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma_ann: np.ndarray, rf: float) -> dict:
    port_ret = float(w @ mu)
    port_var = float(w @ Sigma_ann @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else None
    return {"expected_return": port_ret, "volatility": port_vol, "sharpe": sharpe}

def _require_scipy() -> None:
    if not _HAVE_SCIPY:
        raise HTTPException(
            status_code=503,
            detail="Portfolio optimization is unavailable because SciPy is not installed.",
        )

def _solver_weights(result, strategy: str, mu=None, allow_short=False, target=None) -> np.ndarray:
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=f"{strategy} optimization failed: {result.message}",
        )
    w = np.asarray(result.x, dtype=float)
    if mu is not None and (
        w.shape != mu.shape or not np.isfinite(w).all()
        or abs(w.sum() - 1) > 1e-7
        or w.min() < (-1 if allow_short else 0) - 1e-8 or w.max() > 1 + 1e-8
        or (target is not None and abs(w @ mu - target) > 1e-7)
    ):
        raise HTTPException(status_code=422, detail=f"{strategy}: invalid optimizer weights.")
    return w

def _validated_inputs(mu, cov):
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    if (mu.ndim != 1 or not mu.size or cov.shape != (mu.size, mu.size)
            or not np.isfinite(mu).all() or not np.isfinite(cov).all()):
        raise HTTPException(status_code=422, detail="Invalid optimization inputs.")
    cov = (cov + cov.T) / 2
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise HTTPException(status_code=422, detail="Covariance must be positive semidefinite.")
    return mu, cov

def feasible_target_return_range(mu: np.ndarray, allow_short: bool) -> tuple[float, float]:
    # Linear objectives attain extrema by filling the highest/lowest returns
    # from the lower bound. This includes the net-investment short constraint.
    mu = np.asarray(mu, float)
    if mu.ndim != 1 or not mu.size or not np.isfinite(mu).all():
        raise HTTPException(status_code=422, detail="Invalid expected returns.")
    def extreme(order):
        w = np.full(len(mu), -1.0 if allow_short else 0.0)
        remaining = 1 - w.sum()
        for i in order:
            add = min(1 - w[i], remaining)
            w[i] += add
            remaining -= add
        return float(w @ mu)
    return extreme(np.argsort(mu)), extreme(np.argsort(-mu))


def optimize_max_sharpe(mu: np.ndarray, Sigma_ann: np.ndarray, rf: float, allow_short: bool) -> np.ndarray:
    _require_scipy()
    mu, Sigma_ann = _validated_inputs(mu, Sigma_ann)
    if not np.isfinite(rf):
        raise HTTPException(status_code=422, detail="Invalid risk-free rate.")
    from backtest import solve_weights, BacktestError
    if not allow_short:
        try:
            return solve_weights("max_sharpe", mu, Sigma_ann, rf)[0]
        except BacktestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    maximum = feasible_target_return_range(mu, True)[1]
    if maximum - rf <= 1e-12:
        raise HTTPException(status_code=422, detail=(
            "No feasible portfolio has positive model-estimated excess return in this window "
            "(numerical tolerance 1e-12). Maximum Sharpe is unavailable; no fallback was applied."))
    # Convex transformed problem, preserving -1 <= w_i <= 1 and sum(w)=1.
    e = (mu - rf) / max(float(np.max(np.abs(mu - rf))), 1e-8)
    n = len(mu)
    start = np.full(n, -1.)
    remaining = n + 1.
    for i in np.argsort(-mu):
        add = min(2., remaining)
        start[i] += add
        remaining -= add
    res = minimize(lambda y: float(y @ Sigma_ann @ y), start / (start @ e),
        jac=lambda y: 2 * Sigma_ann @ y, method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda y: y @ e - 1},
            {"type": "ineq", "fun": lambda y: y.sum() - y},
            {"type": "ineq", "fun": lambda y: y.sum() + y},
        ], options={"maxiter": 1000, "ftol": 1e-12})
    y = _solver_weights(res, "Maximum-Sharpe")
    if (not np.isfinite(y).all() or y.sum() <= 0 or abs(y @ e - 1) > 1e-6):
        raise HTTPException(status_code=422, detail="Maximum-Sharpe: invalid transformed solution.")
    res.x = y / y.sum()
    w = _solver_weights(res, "Maximum-Sharpe", mu, True)
    if w @ Sigma_ann @ w <= 1e-16:
        raise HTTPException(status_code=422, detail="Maximum-Sharpe requires positive variance.")
    return w


def optimize_min_variance(mu: np.ndarray, Sigma_ann: np.ndarray, allow_short: bool) -> np.ndarray:
    mu, Sigma_ann = _validated_inputs(mu, Sigma_ann)
    n = len(mu)
    _require_scipy()

    def var_obj(w): return w @ Sigma_ann @ w

    w0 = np.ones(n) / n
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})
    return _solver_weights(res, "Minimum-variance", mu, allow_short)

def optimize_target_return(mu: np.ndarray, Sigma_ann: np.ndarray, target_ret: float, allow_short: bool) -> np.ndarray:
    mu, Sigma_ann = _validated_inputs(mu, Sigma_ann)
    n = len(mu)
    _require_scipy()
    if not np.isfinite(target_ret):
        raise HTTPException(status_code=422, detail="Target return must be finite.")
    minimum_return, maximum_return = feasible_target_return_range(mu, allow_short)
    tolerance = 1e-8
    if target_ret < minimum_return - tolerance or target_ret > maximum_return + tolerance:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target return {target_ret:.2%} is infeasible. "
                f"Choose a value between {minimum_return:.2%} and {maximum_return:.2%}."
            ),
        )

    def var_obj(w): return w @ Sigma_ann @ w

    w0 = np.ones(n) / n
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, mu=mu, t=target_ret: w @ mu - t},
    ]
    if np.ptp(mu) <= 1e-12:
        cons = cons[:1]
    res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    return _solver_weights(res, "Target-return", mu, allow_short, target_ret)


# ----------------- request/response models -----------------
class OptimizeRequest(BaseModel):
    tickers: List[str]
    mode: Literal["max_sharpe", "min_variance", "target_return"] = "max_sharpe"
    targetReturn: Optional[float] = None            # decimal, e.g. 0.12 for 12%
    allowShort: bool = False
    market: Literal["US", "HK", "CN_SH", "CN_SZ", "JP", "CA", "UK", "IN"] = "US"
    historyYears: int = Field(default=5, gt=0)
    # Legacy calendar-day window ending before an exclusive asOfDate cutoff.
    historyDays: Optional[int] = Field(default=None, gt=0, strict=True)
    asOfDate: Optional[date] = Field(
        default=None,
        description="Exclusive end date (YYYY-MM-DD); defaults to today in UTC.",
    )
    startDate: Optional[date] = Field(
        default=None,
        description="Inclusive start date (YYYY-MM-DD). Requires endDate.",
    )
    endDate: Optional[date] = Field(
        default=None,
        description="Inclusive end date (YYYY-MM-DD).",
    )

    @field_validator("asOfDate", "endDate")
    @classmethod
    def validate_end_date(cls, value):
        if value is not None and value > utc_today():
            raise ValueError("End date cannot be in the future")
        return value

class OptimizeResponse(BaseModel):
    market: Dict[str, float]
    perStock: Dict[str, Dict[str, float]]
    usedTickers: List[str]
    weights: Dict[str, float]
    stats: Dict[str, Optional[float]]
    baseline: Dict[str, Optional[float]]


# ----------------- app + CORS -----------------
app = FastAPI(title="Portfolio Optimizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- core endpoint -----------------
@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    if req.mode == "target_return" and (req.targetReturn is None or not math.isfinite(req.targetReturn)):
        raise HTTPException(status_code=422, detail="Target-return mode requires a finite targetReturn.")
    years = req.historyYears
    period = f"{years}y"

    if req.startDate is not None and req.endDate is None:
        raise HTTPException(status_code=422, detail="Date range requires an end date.")
    if req.startDate is None and req.endDate is not None and req.historyDays is None:
        raise HTTPException(status_code=422, detail="End date requires a start date or lookback days.")
    if req.startDate is not None and req.historyDays is not None:
        raise HTTPException(status_code=422, detail="Use either a date range or lookback days, not both.")
    if req.endDate is not None and req.asOfDate is not None:
        raise HTTPException(status_code=422, detail="Use either endDate or asOfDate, not both.")

    bounded = any((req.historyDays is not None, req.asOfDate is not None, req.startDate is not None))
    if req.startDate is not None:
        if req.startDate > req.endDate:
            raise HTTPException(status_code=422, detail="Start date must be on or before end date.")
        start = req.startDate
        end = req.endDate + timedelta(days=1)
    else:
        end = (req.endDate + timedelta(days=1)) if req.endDate is not None else (req.asOfDate or utc_today())
        start = None
    if bounded:
        try:
            if start is None:
                start = (
                    end - timedelta(days=req.historyDays)
                    if req.historyDays is not None
                    else (pd.Timestamp(end) - pd.DateOffset(years=years)).date()
                )
            rate_start = end - timedelta(days=31)
        except (OverflowError, ValueError):
            raise HTTPException(status_code=422, detail="Requested lookback is outside the supported date range.")

    if req.market != "US" and bounded and end < utc_today():
        raise HTTPException(status_code=422, detail=(
            "Historical non-US optimization requires a historical local risk-free rate. "
            "Only current local yields are available; today's yield cannot be used for this window."
        ))

    def get_close(symbol, label):
        try:
            if bounded:
                close = bounded_close(symbol, start, end)
            else:
                history = yf.Ticker(symbol).history(period=period)
                close = history.get("Close", pd.Series(dtype=float))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not retrieve data for {label}: {exc}")
        if close.dropna().empty:
            raise HTTPException(
                status_code=422,
                detail=f"No price data was found for ticker '{symbol}'. Check the ticker and selected market.",
            )
        return close

    # Market (Rm) & Risk-free (Rf)
    benchmark_close = get_close(MARKET_BENCHMARKS[req.market], "the selected market benchmark")
    stock_closes = {ticker: get_close(ticker, f"ticker '{ticker}'") for ticker in req.tickers}
    if bounded:
        market_returns = benchmark_close.pct_change(fill_method=None).dropna()
        if len(market_returns) < 2 or not np.isfinite(market_returns).all() or market_returns.var() <= 0:
            raise HTTPException(status_code=422, detail="Window needs at least two finite, varying market returns.")
    if bounded:
        # Arithmetic annual expected return, not realized CAGR.
        Rm = float(market_returns.mean() * 252)
    else:
        benchmark_year = benchmark_close.resample("YE").last()
        annual_returns = benchmark_year.pct_change().dropna()
        Rm = float(annual_returns.mean())

    if not np.isfinite(Rm):
        raise HTTPException(status_code=422, detail="Insufficient market history to estimate annual return.")

    if req.market == "US":
        if bounded:
            tnx = bounded_close("^TNX", rate_start, end).dropna()
            tnx = tnx.loc[np.isfinite(tnx)]
            if tnx.empty:
                raise HTTPException(
                    status_code=422,
                    detail="No valid US Treasury yield in the 31 calendar days before the cutoff.",
                )
        else:
            tnx = yf.Ticker("^TNX").history(period="1mo")["Close"]
        Rf = float(tnx.iloc[-1] / 100.0)
    else:
        Rf = latest_local_government_yield(req.market)

    # Per-stock: CAPM expected returns + daily returns for covariance
    per_stock: Dict[str, Dict[str, float]] = {}
    daily_returns_matrix = []
    capm_mu: Dict[str, float] = {}

    market_full_close = benchmark_close

    for ticker in req.tickers:
        hist_close = stock_closes[ticker]
        # aligned returns
        returns = build_returns_aligned(hist_close, market_full_close)

        if bounded and (
            len(returns) < 2 or not np.isfinite(returns.to_numpy()).all()
            or returns["market"].var() <= 0
        ):
            raise HTTPException(status_code=422, detail=f"Insufficient valid aligned returns for {ticker}.")

        # CAPM beta & expected return
        if not returns.empty:
            X = sm.add_constant(returns["market"])
            model = sm.OLS(returns["stock"], X).fit()
            beta = float(model.params.get("market", float("nan")))
            exp_ret = float(Rf + beta * (Rm - Rf))
            capm_mu[ticker] = exp_ret
            stock_daily = returns["stock"].copy()
            stock_daily.name = ticker
            daily_returns_matrix.append(stock_daily)
        else:
            # fallback: still collect stock-only daily returns for covariance
            single = hist_close.dropna().pct_change().dropna()
            single.name = ticker
            daily_returns_matrix.append(single)
            beta = float("nan")
            exp_ret = float("nan")

        # simple CAGR (optional)
        prices = hist_close.dropna()
        cagr = float("nan")
        if len(prices) > 1:
            years_span = max(1, (prices.index[-1].year - prices.index[0].year))
            if bounded:
                years_span = (prices.index[-1] - prices.index[0]).total_seconds() / (365.25 * 86400)
            if prices.iloc[0] > 0 and years_span > 0:
                cagr = float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years_span) - 1)

        per_stock[ticker] = {"beta": beta, "expectedCAPM": exp_ret, "cagrApprox": cagr}

    if len(daily_returns_matrix) == 0:
        # Nothing usable
        return OptimizeResponse(
            market={"Rm": Rm, "Rf": Rf},
            perStock=per_stock,
            usedTickers=[],
            weights={},
            stats={},
            baseline={}
        )

    stock_returns_df = pd.concat(daily_returns_matrix, axis=1, join="inner").dropna()
    if bounded and len(stock_returns_df) < 2:
        raise HTTPException(status_code=422, detail="Window needs at least two common daily returns across stocks.")
    used_tickers = list(stock_returns_df.columns)

    # expected returns vector (annual)
    mu_series = pd.Series(index=used_tickers, dtype=float)
    for col in used_tickers:
        if col in capm_mu and np.isfinite(capm_mu[col]):
            mu_series[col] = capm_mu[col]
        else:
            mu_series[col] = stock_returns_df[col].mean() * 252.0

    Sigma_ann = annualize_cov(stock_returns_df.cov().values, periods=252)
    mu = mu_series.values

    allow_short = req.allowShort
    if req.mode == "max_sharpe":
        w = optimize_max_sharpe(mu, Sigma_ann, Rf, allow_short)
    elif req.mode == "min_variance":
        w = optimize_min_variance(mu, Sigma_ann, allow_short)
    else:
        w = optimize_target_return(mu, Sigma_ann, req.targetReturn, allow_short)

    stats = portfolio_stats(w, mu, Sigma_ann, Rf)
    weq = np.ones_like(w) / len(w)
    stats_eq = portfolio_stats(weq, mu, Sigma_ann, Rf)

    return OptimizeResponse(
        market={"Rm": Rm, "Rf": Rf},
        perStock=per_stock,
        usedTickers=used_tickers,
        weights={t: float(x) for t, x in zip(used_tickers, w.tolist())},
        stats={k: None if v is None else float(v) for k, v in stats.items()},
        baseline={k: None if v is None else float(v) for k, v in stats_eq.items()},
    )

# Independent research endpoints preserve the existing optimizer API contract.
from backtest_api import router as backtest_router
app.include_router(backtest_router)
