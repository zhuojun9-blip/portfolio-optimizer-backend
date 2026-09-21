# Out-of-sample portfolio backtests

Run `uvicorn version2:app --reload`, then open `/backtest` on that backend.
This adds a self-contained dashboard and `POST /api/backtest`; it does not
change the existing optimizer UI or any Netlify project association.
The dashboard uses same-origin requests, requires no frontend build or external
charting CDN, and can be linked from the existing frontend after backend deploy.

## Two experiments

- `mode: "single"`: estimate once, buy at the first market close on or after
  `startDate`, and hold through `endDate` (inclusive). Use `lookbackDays` or
  an explicit inclusive `estimationStart`/`estimationEnd` pair. The pair must
  end before `startDate`; it overrides lookbackDays. No interim rebalancing.
- `mode: "rolling"`: repeat on the first market session of each calendar week
  (Monday–Sunday), month, or quarter. A partial first period starts on the first
  eligible session. Each execution has a trailing `lookbackDays` calendar-day
  estimation window. The default frequency is quarterly; weekly and monthly
  are available. Do not pick/report only the best frequency after viewing results.

Example request is in `examples/backtest.json`. To test a single explicit
training window, set mode to single, estimationStart to 2016-01-01 and
estimationEnd to 2020-12-31. The holding period can start on 2021-01-01.
The API returns actual execution/valuation dates, which can differ from the
requested dates because of weekends and holidays.

## Accounting and comparison

All four strategies (maximum Sharpe, minimum variance, target return and equal
weight) have identical stocks, evaluation dates, execution rules, and frequency.
At an execution close, old holdings first earn the day's close-to-close return;
then their accumulated value is allocated to new weights. New weights earn only
subsequent returns. The initial allocation earns no return before its execution
close. Capital is never reset between windows. Holdings drift naturally.

The first release is long-only, fully invested in fractional holdings, USD, with
zero costs, taxes and slippage. Yahoo currency metadata must confirm USD. Symbols
with foreign-market suffixes are rejected; no FX conversion is implemented.
Adjusted prices approximate total returns with dividend reinvestment. Do not
also credit dividends separately. The theoretical execution approximation uses
adjusted units, not literal historical share counts.

We use an OLS-equivalent beta (stock/market covariance divided by market
variance), CAPM expected returns, and daily sample covariance × 252. For all
backtest windows, annual market expected return is mean daily market return ×
252. This is intentionally consistent across window lengths, unlike the legacy
optimizer's year-end estimator. The market proxy ^GSPC is a price index, while
adjusted stock returns include dividends; this modeling mismatch is disclosed.

The target strategy defaults to the current equal-weight *model expected*
return. Supply targetReturn as an annual decimal (0.10 = 10%) for a fixed target.
Infeasible targets, solver failure, or invalid weights mark that strategy failed;
no equal-weight fallback, incomplete performance table row, or partial equity
curve is presented as a successful full-period strategy. Successful earlier
rebalance records remain available for debugging. SLSQP numerical convergence
is not a certificate of global optimality for the Sharpe objective.

The loader requests each stock/index once per experiment, rejects missing or
nonpositive stock prices on market sessions, and does not forward-fill stock
prices or silently change the universe. Every training window needs at least
max(20, number of stocks + 1) daily returns. Actual training dates/counts appear
in the audit. Passing this minimum does not establish statistical reliability.

## Metrics

- Total return = final value / initial value − 1.
- CAGR = (final / initial)^(365.25 / actual elapsed calendar days) − 1.
- Volatility = sample standard deviation of daily portfolio returns × sqrt(252).
- Sharpe = mean(daily excess returns) / sample standard deviation(daily excess
  returns) × sqrt(252). A zero denominator produces null, displayed as a dash.
- Daily risk-free proxy = (1 + historical annual ^TNX yield / 100)^(1/252) − 1.
  Use the latest finite yield dated at or before the previous valuation close,
  with a 31-calendar-day stale-data limit. Never fill from the future. CAPM uses
  the most recent yield strictly before its estimation cutoff. A 10-year yield
  is only a proxy, not an actual short-term cash investment return.
- Drawdown = value / running peak − 1; maximum drawdown is the most negative value.
- Turnover = sum across rebalances of 0.5 × sum(abs(target − drifted weight));
  initial investment is excluded. This is cumulative, not annualized. Costs = 0.

Metrics derived from short periods can be unstable. Fixed user-selected stocks
can create survivorship/selection bias; Yahoo data are not a point-in-time
vintage archive. Do not interpret past results as a forecast. Model estimates
in the rebalance audit are distinct from realized metrics.

## Reproducible offline runs

```bash
python run_backtest.py examples/backtest.json --output results/quarterly
python run_backtest.py results/quarterly/config.json --data results/quarterly/data --output results/replay
```

The first command saves adjusted prices, the index, historical rates, config,
results.json, summary.csv, equity.csv and daily weights.csv. The second makes no
network calls. Results include a dataset SHA-256 for provenance. The dashboard
exports full results JSON and equity CSV; use the CLI to retain input data too.
Do not commit private datasets or generated performance claims without review.

## Verification

```bash
pip install -r requirements-dev.txt
python -m unittest discover -v
```

Tests use synthetic prices (not claimed investment performance). They check
hand-calculated holdings/drift, no return before execution, old weights earning
execution-day returns, rolling schedules, future-data isolation, historical rate
availability, infeasible targets, solver failure, metrics and API errors.

This change adds no deployment configuration. The intended frontend remains
majestic-kataifi-bac07d; do not deploy this Python backend as a static Netlify site.

Dashboard DOM behavior can also be checked without a market-data connection:

```bash
npm install --prefix ui-tests
npm test --prefix ui-tests
```

These test mode switching, payloads, chart/table rendering, weight selection,
failed strategy display, and stale-result removal after errors. They are DOM
unit tests, not a claim of visual verification in every browser.
