"""Deterministic, long-only out-of-sample portfolio research.

Data fetching is outside run_backtest so a saved snapshot can be replayed.
All execution is at adjusted close; weights chosen before that date earn only
subsequent close-to-close returns. No transaction costs, leverage or taxes.
"""
from datetime import date, datetime, timedelta, timezone
from typing import Literal
import hashlib

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.optimize import minimize


BACKTEST_MARKETS = {
    "US": {"benchmark": "^GSPC", "currency": "USD"},
    "HK": {"benchmark": "^HSI", "currency": "HKD"},
    "CN_SH": {"benchmark": "000001.SS", "currency": "CNY"},
    "CN_SZ": {"benchmark": "399001.SZ", "currency": "CNY"},
}


class BacktestError(ValueError):
    pass


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str] = Field(min_length=2, max_length=20)
    market: Literal["US", "HK", "CN_SH", "CN_SZ"] = "US"
    startDate: date  # Earliest execution date, inclusive.
    endDate: date  # Final valuation date, inclusive.
    mode: Literal["single", "rolling"] = "single"
    frequency: Literal["weekly", "monthly", "quarterly"] = "quarterly"
    lookbackDays: int = Field(default=1826, ge=30, le=7305, strict=True)
    estimationStart: date | None = None  # Explicit training window for single mode.
    estimationEnd: date | None = None  # Inclusive; must precede startDate.
    initialCapital: float = Field(default=100000, gt=0, le=1e12, allow_inf_nan=False)
    targetReturn: float | None = Field(default=None, ge=-1, le=10, allow_inf_nan=False)
    # None means target the equal-weight expected return in each training window.
    userWeights: dict[str, float] | None = None

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, values):
        import re
        values = [v.strip().upper() for v in values]
        if len(set(values)) != len(values):
            raise ValueError("Duplicate tickers are not allowed")
        return values

    @model_validator(mode="after")
    def validate_dates(self):
        import re
        patterns = {
            "US": (r"[A-Z][A-Z0-9-]{0,14}", "US symbols such as AAPL or BRK-B"),
            "HK": (r"\d{4}\.HK", "Hong Kong symbols such as 0700.HK"),
            "CN_SH": (r"\d{6}\.SS", "Shanghai symbols such as 600519.SS"),
            "CN_SZ": (r"\d{6}\.SZ", "Shenzhen symbols such as 000001.SZ"),
        }
        pattern, description = patterns[self.market]
        if any(not re.fullmatch(pattern, ticker) for ticker in self.tickers):
            raise ValueError(f"Use {description}")
        if not date(1970, 1, 1) <= self.startDate < self.endDate:
            raise ValueError("Require 1970-01-01 <= startDate < endDate")
        if self.endDate >= datetime.now(timezone.utc).date():
            raise ValueError("endDate must be before today (completed daily observations only)")
        if (self.endDate - self.startDate).days > 7305:
            raise ValueError("Evaluation period is limited to 20 years per request")
        if (self.estimationStart is None) != (self.estimationEnd is None):
            raise ValueError("Provide both estimationStart and estimationEnd, or neither")
        if self.estimationStart is not None:
            if self.mode != "single":
                raise ValueError("Explicit estimation dates apply only to single-period mode")
            if not date(1950, 1, 1) <= self.estimationStart < self.estimationEnd < self.startDate:
                raise ValueError("Estimation dates must be ordered and end before startDate")
            if (self.estimationEnd - self.estimationStart).days > 7305:
                raise ValueError("Estimation window is limited to 20 years")
        if self.userWeights is not None:
            normalized = {ticker.strip().upper(): weight for ticker, weight in self.userWeights.items()}
            if set(normalized) != set(self.tickers):
                raise ValueError("User weights must include each selected ticker exactly once")
            if any(not np.isfinite(weight) or weight < 0 for weight in normalized.values()):
                raise ValueError("User weights must be finite and non-negative")
            if not np.isclose(sum(normalized.values()), 1.0, atol=1e-6):
                raise ValueError("User weights must sum to 1.0")
            self.userWeights = normalized
        return self


def clean_series(series):
    series = series.copy()
    idx = pd.DatetimeIndex(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    series.index = idx.normalize()
    if series.index.has_duplicates:
        raise BacktestError("Duplicate dates in market data")
    return series.sort_index()


def prepare_data(req, loader):
    """loader(symbol, inclusive_start, exclusive_end) -> adjusted Close Series."""
    begin = req.estimationStart or req.startDate - timedelta(days=req.lookbackDays)
    end = req.endDate + timedelta(days=1)
    benchmark = BACKTEST_MARKETS[req.market]["benchmark"]
    market = clean_series(loader(benchmark, begin, end))
    market = market.loc[(market.index >= pd.Timestamp(begin)) & (market.index < pd.Timestamp(end))]
    if market.empty:
        raise BacktestError("No market data in the requested window")
    stocks = {}
    for ticker in req.tickers:
        s = clean_series(loader(ticker, begin, end))
        stocks[ticker] = s.reindex(market.index)
    prices = pd.DataFrame(stocks)
    values = np.column_stack([prices.to_numpy(), market.to_numpy()])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise BacktestError("Missing/nonpositive prices on market sessions. No assets or dates were silently removed; choose a window with complete histories.")
    if req.market == "US":
        yields = clean_series(loader("^TNX", begin - timedelta(days=31), end)) / 100
        yields = yields.loc[np.isfinite(yields)]
    else:
        # Returns are calculated in local currency; use a zero local cash baseline
        # until a reliable historical sovereign-yield feed is configured per market.
        yields = pd.Series(0.0, index=market.index)
    return prices, market, yields


def rate_before(yields, cutoff):
    cutoff = pd.Timestamp(cutoff)
    past = yields.loc[(yields.index < cutoff) & (yields.index >= cutoff - pd.Timedelta(days=31))]
    if past.empty or not np.isfinite(past.iloc[-1]) or past.iloc[-1] <= -1:
        raise BacktestError(f"No valid Treasury yield within 31 days before {cutoff.date()}")
    return float(past.iloc[-1])


def estimate_inputs(prices, market, yields, start, cutoff):
    """Pure trailing-window CAPM estimator; never inspect cutoff/future rows."""
    hist = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(cutoff))]
    hist_market = market.reindex(hist.index)
    returns = hist.pct_change(fill_method=None).iloc[1:]
    mr = hist_market.pct_change(fill_method=None).iloc[1:]
    minimum = max(20, prices.shape[1] + 1)
    if len(returns) < minimum:
        raise BacktestError(f"Need at least {minimum} training returns before {pd.Timestamp(cutoff).date()}; got {len(returns)}")
    if not np.isfinite(returns.to_numpy()).all() or not np.isfinite(mr).all() or mr.var() <= 1e-16:
        raise BacktestError("Invalid or constant training returns")
    # Equivalent to the existing OLS slope with an intercept.
    beta = np.array([returns[t].cov(mr) / mr.var() for t in prices.columns])
    rf = rate_before(yields, cutoff)
    rm = float(mr.mean() * 252)
    mu = rf + beta * (rm - rf)
    cov = returns.cov().to_numpy() * 252
    return mu, cov, rf, {
        "trainingStart": hist.index[0].date().isoformat(),
        "trainingEnd": hist.index[-1].date().isoformat(),
        "observations": len(returns), "marketExpectedReturn": rm, "riskFreeRate": rf,
        "beta": dict(zip(prices.columns, beta.tolist())),
    }


def solve_weights(strategy, mu, cov, rf, target=None):
    """Same objectives as the optimizer, with explicit failures and no fallback."""
    n = len(mu)
    eq = np.ones(n) / n
    if strategy == "equal_weight":
        return eq, "Equal weight"
    if strategy == "target_return":
        target = float(mu.mean()) if target is None else target
        if target < mu.min() - 1e-8 or target > mu.max() + 1e-8:
            raise BacktestError(f"Target return {target:.4%} outside feasible range [{mu.min():.4%}, {mu.max():.4%}]")
    def objective(w):
        variance = float(w @ cov @ w)
        if strategy == "max_sharpe":
            return -(w @ mu - rf) / np.sqrt(max(variance, 1e-18))
        return variance
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if strategy == "target_return" and np.ptp(mu) > 1e-10:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target})
    result = minimize(objective, eq, method="SLSQP", bounds=[(0, 1)] * n,
                      constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    w = result.x
    valid = (result.success and np.isfinite(w).all() and abs(w.sum() - 1) < 1e-6
             and w.min() >= -1e-7 and w.max() <= 1 + 1e-7)
    if strategy == "target_return":
        valid = valid and abs(w @ mu - target) < 1e-6
    if not valid or w @ cov @ w <= 1e-16:
        raise BacktestError(f"{strategy}: invalid optimizer result ({result.message})")
    return w, str(result.message)


def execution_dates(index, mode, frequency):
    if mode == "single":
        return {index[0]}
    periods = index.to_period({"weekly": "W-SUN", "monthly": "M", "quarterly": "Q"}[frequency])
    return set(index[~periods.duplicated()])


def realized_metrics(nav, rf_daily):
    rets = nav.pct_change().iloc[1:]
    excess = rets.to_numpy() - np.asarray(rf_daily)
    std = float(np.std(excess, ddof=1))
    elapsed = (nav.index[-1] - nav.index[0]).days / 365.25
    return {
        "totalReturn": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / elapsed) - 1),
        "volatility": float(rets.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(np.mean(excess) / std * np.sqrt(252)) if std > 1e-12 else None,
        "maxDrawdown": float((nav / nav.cummax() - 1).min()),
        "finalValue": float(nav.iloc[-1]), "returnObservations": len(rets),
    }


def run_backtest(req, prices, market, yields):
    """Input prices must cover every market session; absence is an error."""
    prices = prices.copy()
    if list(prices.columns) != req.tickers or not prices.index.equals(market.index):
        raise BacktestError("Price and market indices/ticker order must match")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise BacktestError("Dates must be unique and sorted")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise BacktestError("Invalid stock prices")
    yields = clean_series(yields)
    test = prices.loc[str(req.startDate):str(req.endDate)]
    if len(test) < 3:
        raise BacktestError("Need at least three valuation dates (two realized daily returns)")
    schedule = execution_dates(test.index, req.mode, req.frequency)
    names = ["max_sharpe", "min_variance", "target_return", "equal_weight"]
    if req.userWeights is not None:
        names.append("user_holdings")
    states = {s: {"status": "ok", "holdings": None, "nav": [], "weights": [], "rebalances": [], "turnover": 0.} for s in names}
    previous_price = None
    rf_daily = []
    for pos, (day, price) in enumerate(test.iterrows()):
        p = price.to_numpy()
        if pos:
            # Cash proxy known at the previous valuation close; no forward-fill from future.
            rate = rate_before(yields, test.index[pos - 1] + pd.Timedelta(days=1))
            rf_daily.append((1 + rate) ** (1 / 252) - 1)
        if day in schedule:
            if req.estimationStart:
                start = req.estimationStart
                cutoff = req.estimationEnd + timedelta(days=1)
            else:
                start = day.date() - timedelta(days=req.lookbackDays)
                cutoff = day.date()
            mu, cov, rf, info = estimate_inputs(prices, market, yields, start, cutoff)
        for strategy, state in states.items():
            if state["status"] != "ok":
                continue
            if pos:
                state["holdings"] *= p / previous_price
            value = float(state["holdings"].sum()) if pos else req.initialCapital
            if day in schedule:
                before = state["holdings"] / value if pos else np.zeros(len(p))
                if strategy == "user_holdings" and pos:
                    pass
                else:
                    try:
                        if strategy == "user_holdings":
                            w = np.array([req.userWeights[ticker] for ticker in req.tickers], dtype=float)
                            message = "User-defined buy-and-hold allocation"
                        else:
                            w, message = solve_weights(strategy, mu, cov, rf, req.targetReturn)
                    except BacktestError as exc:
                        state.update(status="failed", error=str(exc), failureDate=day.date().isoformat())
                        # Never report a partial run as comparable full-period performance.
                        continue
                    turnover = float(np.abs(w - before).sum() / 2) if pos else 0.
                    state["turnover"] += turnover
                    state["holdings"] = value * w
                    state["rebalances"].append({**info, "executionDate": day.date().isoformat(),
                        "weights": dict(zip(req.tickers, w.tolist())),
                        "preTradeWeights": dict(zip(req.tickers, before.tolist())),
                        "turnover": turnover, "solverMessage": message,
                        "expectedReturn": float(w @ mu), "expectedVolatility": float(np.sqrt(w @ cov @ w)),
                        "targetReturn": (float(mu.mean()) if req.targetReturn is None else req.targetReturn) if strategy == "target_return" else None})
            state["nav"].append(value)
            state["weights"].append((state["holdings"] / value).tolist())
        previous_price = p.copy()
    output = {}
    dates = test.index.strftime("%Y-%m-%d").tolist()
    for strategy, state in states.items():
        if state["status"] != "ok":
            output[strategy] = {"status": "failed", "error": state["error"], "failureDate": state["failureDate"], "metrics": None,
                                "rebalances": state["rebalances"]}
            continue
        nav = pd.Series(state["nav"], index=test.index)
        metrics = realized_metrics(nav, rf_daily)
        metrics.update(turnover=state["turnover"], transactionCosts=0., rebalanceCount=len(state["rebalances"]))
        output[strategy] = {"status": "ok", "metrics": metrics, "nav": state["nav"],
            "drawdown": (nav / nav.cummax() - 1).tolist(), "weights": state["weights"], "rebalances": state["rebalances"]}
    # Hash only the data available within the requested run, not injected future rows.
    benchmark = BACKTEST_MARKETS[req.market]["benchmark"]
    snapshot = pd.concat([prices.loc[:str(req.endDate)], market.loc[:str(req.endDate)].rename(benchmark),
                          yields.loc[:str(req.endDate)].rename("rf")], axis=1)
    return {"config": req.model_dump(mode="json"), "dates": dates, "strategies": output,
        "dataSha256": hashlib.sha256(snapshot.to_csv().encode()).hexdigest(),
        "assumptions": ["Zero transaction costs; long-only, fully invested, fractional holdings.",
            "Adjusted-close total-return approximation with dividend reinvestment; no separate dividend credits.",
            ("Sharpe uses historical US 10-year Treasury yields." if req.market == "US"
             else "Sharpe uses a 0% local cash baseline; historical local sovereign yields are not yet configured."),
            "Execute at first market close in each calendar period; use only prices before execution day.",
            "S&P 500 price index (^GSPC) market proxy excludes dividends; annual market estimate = mean daily return × 252.",
            "Sharpe uses lagged ^TNX annual yields converted by (1 + yield)^(1/252) - 1; this is a cash-rate proxy, not a realized Treasury return.",
            "Fixed user-selected US/USD universe; survivorship bias and historical data revisions are not eliminated.",
            "Failed strategies have no full-period performance metrics. Turnover excludes initial investment."]}
