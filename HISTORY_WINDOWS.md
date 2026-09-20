# Optimization history windows

POST /api/optimize accepts an optional exclusive historical cutoff, asOfDate,
in YYYY-MM-DD format.

Example:

```json
{
  "tickers": ["AAPL", "JPM"],
  "mode": "max_sharpe",
  "historyDays": 90,
  "asOfDate": "2021-01-01",
  "allowShort": false
}
```

This uses prices dated October 3 through December 31, 2020. The start is
inclusive and the end is exclusive; weekends and holidays contribute no prices.
historyDays counts calendar days, not trading sessions. It takes precedence
over historyYears. With asOfDate and historyYears instead, the start is the
cutoff minus that many calendar years (February 29 clips to February 28 when
needed). The default historyYears is 5.

Day-based requests without asOfDate end before today in UTC. Legacy requests
with neither historyDays nor asOfDate retain the existing year-based provider
period and latest-rate behavior for compatibility.

For bounded requests, stocks and market use identical boundaries. Returned
rows are sorted and filtered locally as well as in the download request.
Exchange-local dates are preserved when stripping timezone information.
Adjusted close prices are requested explicitly.

Rf is the latest finite ^TNX close strictly before the cutoff, divided by 100.
It is searched within the preceding 31 calendar days independently of the
stock window. If none exists, the request returns HTTP 422; it never falls back
to today's rate. Future cutoffs, invalid dates, and insufficient observations
are rejected.

Day-based windows estimate annual market return as 252 times the arithmetic
mean daily market return. Year-based windows retain the legacy year-end return
estimator, including its partial-calendar-year limitation. Outputs remain
annualized. Very short windows may pass mathematical checks but do not provide
reliable estimates.

This feature generates a historical allocation, not a backtest. ^TNX remains
a long-term yield proxy, not a realized short-term cash return. Yahoo historical
data can be revised: date filtering does not provide a point-in-time data
archive. Existing solver failure fallbacks have not yet been revised.

Run the deterministic tests from the repository directory:

```bash
python -m unittest discover -v
```

The tests use simulated data, including extreme observations on and after the
cutoff that must not affect any optimization mode. They do not verify Yahoo's
live service.
