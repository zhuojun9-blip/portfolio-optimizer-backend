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
    maxWeight: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)

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
        if self.maxWeight * len(self.tickers) < 1 - 1e-10:
            raise ValueError("Maximum weight is infeasible: number of stocks × cap must be at least 100%")
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
        "expectedReturns": dict(zip(prices.columns, mu.tolist())),
        "annualCovariance": cov.tolist(),
        "covarianceTickers": list(prices.columns),
    }


def solve_weights(strategy, mu, cov, rf, target=None, max_weight=1.0):
    """Validate feasible solutions and compare starts; never silently change objectives.

    Positive excess-return Sharpe uses a convex variance minimization in scaled
    weights. Nonpositive unconstrained Sharpe is maximized at a simplex vertex.
    With a cap and nonpositive excess returns, use deterministic multistart;
    this branch is explicitly not a certificate of global optimality.
    """
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    n = len(mu)
    if strategy not in {"max_sharpe", "min_variance", "target_return", "equal_weight"}:
        raise BacktestError("Unknown strategy")
    if (cov.shape != (n, n) or not np.isfinite(mu).all() or
            not np.isfinite(cov).all() or not np.isfinite(rf) or
            not np.isfinite(max_weight) or not 0 < max_weight <= 1 or n * max_weight < 1 - 1e-10):
        raise BacktestError("Invalid optimization inputs or infeasible weight cap")
    cov = (cov + cov.T) / 2
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise BacktestError("Covariance must be positive semidefinite")
    eq = np.ones(n) / n
    if strategy == "equal_weight":
        return eq, "Equal weight"

    def corner(order):
        w = np.zeros(n)
        remaining = 1.0
        for i in order:
            w[i] = min(max_weight, remaining)
            remaining -= w[i]
        return w

    low, high = corner(np.argsort(mu)), corner(np.argsort(-mu))
    starts = [eq, low, high]
    starts += [corner(np.roll(np.arange(n), i)) for i in range(n)]
    if strategy == "target_return":
        target = float(mu.mean()) if target is None else float(target)
        lo, hi = float(low @ mu), float(high @ mu)
        if not np.isfinite(target) or target < lo - 1e-8 or target > hi + 1e-8:
            raise BacktestError(f"Target return outside feasible capped range [{lo:.4%}, {hi:.4%}]")
        starts = [eq if hi - lo < 1e-12 else low + (high - low) * ((target - lo) / (hi - lo))]

    def valid(w):
        return (np.isfinite(w).all() and abs(w.sum() - 1) < 1e-7 and
                w.min() >= -1e-8 and w.max() <= max_weight + 1e-8 and
                w @ cov @ w > 1e-16 and
                (strategy != "target_return" or abs(w @ mu - target) < 1e-7))

    def objective(w):
        variance = float(w @ cov @ w)
        return -(w @ mu - rf) / np.sqrt(max(variance, 1e-18)) if strategy == "max_sharpe" else variance

    excess = mu - rf
    if strategy == "max_sharpe" and excess.max() <= 0 and max_weight == 1:
        choices = [w for w in np.eye(n) if valid(w)]
        if not choices:
            raise BacktestError("No positive-variance feasible allocation")
        return min(choices, key=objective), "Nonpositive excess returns: exact single-asset comparison"

    options = {"maxiter": 1000, "ftol": 1e-12}
    if strategy == "max_sharpe" and high @ excess > 1e-12:
        # y = w / (excess @ w); enforce excess @ y = 1.
        scale = max(float(np.max(np.abs(excess))), 1e-8)
        e = excess / scale
        solution = minimize(lambda y: float(y @ cov @ y), high / (high @ e),
            jac=lambda y: 2 * cov @ y, method="SLSQP", bounds=[(0, None)] * n,
            constraints=[{"type": "eq", "fun": lambda y: y @ e - 1, "jac": lambda y: e},
                         {"type": "ineq", "fun": lambda y: max_weight * y.sum() - y,
                          "jac": lambda y: max_weight * np.ones((n, n)) - np.eye(n)}], options=options)
        y = solution.x
        w = y / y.sum() if y.sum() > 0 else np.full(n, np.nan)
        if not solution.success or not valid(w) or abs(y @ e - 1) > 1e-6:
            raise BacktestError(f"max_sharpe: invalid optimizer result ({solution.message})")
        if any(valid(v) and objective(v) < objective(w) - 1e-7 for v in starts):
            raise BacktestError("max_sharpe: solution is inferior to a feasible comparison allocation")
        return w, "Positive-excess Sharpe: convex formulation; comparison checks passed"

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if strategy == "target_return" and np.ptp(mu) > 1e-10:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target})
    candidates = []
    for start in starts if strategy == "max_sharpe" else starts[:1]:
        solution = minimize(objective, start, method="SLSQP", bounds=[(0, max_weight)] * n,
                            constraints=constraints, options=options)
        if solution.success and valid(solution.x):
            candidates.append(solution.x)
    if not candidates:
        raise BacktestError(f"{strategy}: invalid optimizer result ({solution.message})")
    candidates += [w for w in starts if valid(w)]
    w = min(candidates, key=objective)
    return w, ("Nonpositive feasible excess returns: capped multistart, global optimum not certified"
               if strategy == "max_sharpe" else "Variance minimization; feasibility and comparison checks passed")


def execution_dates(index, mode, frequency):
    if mode == "single":
        return {index[0]}
    periods = index.to_period({"weekly": "W-SUN", "monthly": "M", "quarterly": "Q"}[frequency])
    return set(index[~periods.duplicated()])


def realized_metrics(nav, rf_daily):
    rets = nav.pct_change().iloc[1:]
    excess = rets.to_numpy() - np.asarray(rf_daily)
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    elapsed = (nav.index[-1] - nav.index[0]).days / 365.25
    return {
        "totalReturn": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / elapsed) - 1),
        "volatility": float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else None,
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
                            w, message = solve_weights(strategy, mu, cov, rf, req.targetReturn, req.maxWeight)
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
                        "estimatedSharpe": float((w @ mu - rf) / np.sqrt(w @ cov @ w)) if w @ cov @ w > 1e-16 else None,
                        "nonpositiveExcessReturns": bool(np.max(mu - rf) <= 0),
                        "weightCap": req.maxWeight if strategy != "user_holdings" else None,
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
        metrics["maxStockWeight"] = float(np.max(state["weights"]))
        metrics["maxTargetWeight"] = max(max(b["weights"].values()) for b in state["rebalances"])
        metrics.update(turnover=state["turnover"], transactionCosts=0., rebalanceCount=len(state["rebalances"]))
        output[strategy] = {"status": "ok", "metrics": metrics, "nav": state["nav"],
            "annualMetrics": annual_metrics(nav, rf_daily),
            "drawdown": (nav / nav.cummax() - 1).tolist(), "weights": state["weights"], "rebalances": state["rebalances"]}
    # Hash only the data available within the requested run, not injected future rows.
    benchmark = BACKTEST_MARKETS[req.market]["benchmark"]
    snapshot = pd.concat([prices.loc[:str(req.endDate)], market.loc[:str(req.endDate)].rename(benchmark),
                          yields.loc[:str(req.endDate)].rename("rf")], axis=1)
    return {"config": req.model_dump(mode="json"), "dates": dates, "strategies": output,
        "marketInfo": {**BACKTEST_MARKETS[req.market], "cashBaseline": "lagged ^TNX yield proxy" if req.market == "US" else "0% local cash assumption"},
        "riskFreeDaily": rf_daily,
        "riskFreeDates": dates[1:],
        "dataSha256": hashlib.sha256(snapshot.to_csv().encode()).hexdigest(),
        "assumptions": ["Zero transaction costs; long-only, fully invested, fractional holdings.",
            "Adjusted-close total-return approximation with dividend reinvestment; no separate dividend credits.",
            ("Sharpe uses historical US 10-year Treasury yields." if req.market == "US"
             else "Sharpe uses a 0% local cash baseline; historical local sovereign yields are not yet configured."),
            "Execute at first market close in each calendar period; use only prices before execution day.",
            f"CAPM benchmark: {benchmark}; price-index returns exclude dividends; annual estimate = mean daily return × 252.",
            ("Lagged ^TNX yields converted by (1 + yield)^(1/252) - 1; a cash-rate proxy, not a realized Treasury return." if req.market == "US" else "Local sovereign yields are unavailable: 0% is an explicit assumption, not an observed risk-free rate."),
            f"Fixed {req.market} universe valued in {BACKTEST_MARKETS[req.market]['currency']}; survivorship bias and data revisions remain.",
            "Weight caps apply at executions to optimized portfolios; weights may drift above the cap. User holdings are uncapped buy-and-hold.",
            "Failed strategies have no full-period performance metrics. Turnover excludes initial investment."]}


def annual_metrics(nav, rf_daily):
    """Yearly returns include the prior year's last close; flag partial years."""
    out = []
    for year in sorted(set(nav.index[1:].year)):
        positions = np.flatnonzero(nav.index[1:].year == year) + 1
        a, b = int(positions[0]), int(positions[-1])
        part = nav.iloc[a - 1:b + 1]
        m = realized_metrics(part, np.asarray(rf_daily)[a - 1:b])
        out.append({"year": int(year), "startDate": str(part.index[0].date()),
                    "endDate": str(part.index[-1].date()),
                    "partialYear": bool(part.index[0].year == year or (b == len(nav) - 1 and (nav.index[-1].month < 12 or nav.index[-1].day < 28))),
                    **m})
    return out


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: BacktestRequest
    lookbacks: list[int] = Field(default_factory=lambda: [90, 365, 1826], min_length=1, max_length=3)
    frequencies: list[Literal["monthly", "quarterly"]] = Field(default_factory=lambda: ["monthly", "quarterly"], min_length=1, max_length=2)
    cappedWeight: float = Field(default=0.6, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_grid(self):
        if len(set(self.lookbacks)) != len(self.lookbacks) or any(x < 30 or x > 7305 for x in self.lookbacks):
            raise ValueError("Use distinct lookbacks between 30 and 7305 days")
        if len(set(self.frequencies)) != len(self.frequencies):
            raise ValueError("Use distinct frequencies")
        if self.cappedWeight * len(self.base.tickers) < 1 - 1e-10:
            raise ValueError("Comparison weight cap is infeasible for this universe")
        return self


def comparison_base(req, lookback, frequency, cap):
    values = req.base.model_dump()
    values.update(mode="rolling", estimationStart=None, estimationEnd=None,
                  lookbackDays=lookback, frequency=frequency, maxWeight=cap)
    return BacktestRequest(**values)


def run_comparison(req, prices, market, yields):
    """One shared snapshot and evaluation dates for all original/capped runs."""
    rows, runs = [], []
    for lookback in req.lookbacks:
        for frequency in req.frequencies:
            for variant, cap in [("original", 1.0), ("capped", req.cappedWeight)]:
                cfg = comparison_base(req, lookback, frequency, cap)
                entry = {"lookbackDays": lookback, "frequency": frequency, "variant": variant, "maxWeight": cap}
                try:
                    result = run_backtest(cfg, prices, market, yields)
                except BacktestError as exc:
                    runs.append({**entry, "status": "failed", "error": str(exc)})
                    continue
                runs.append({**entry, "status": "ok", "result": result})
                baseline = result["strategies"]["equal_weight"]
                for key, strategy in result["strategies"].items():
                    row = {**entry, "strategy": key, "status": strategy["status"]}
                    if strategy["status"] == "ok":
                        m, bm = strategy["metrics"], baseline["metrics"]
                        delta = None if m["sharpe"] is None or bm["sharpe"] is None else m["sharpe"] - bm["sharpe"]
                        row.update(metrics=m, sharpeDifference=delta, annualMetrics=strategy["annualMetrics"])
                    else:
                        row.update(error=strategy["error"], failureDate=strategy["failureDate"])
                    rows.append(row)
    return {"request": req.model_dump(mode="json"), "rows": rows, "runs": runs,
            "note": "Exploratory comparisons, not an untouched test. Equal weight is rebalanced at the same frequency. Costs are zero."}
