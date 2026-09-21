import unittest
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

import version2 as api


class AsOfTests(unittest.TestCase):
    def run_request(self, request=None, poison=False, no_rate=False):
        calls = []
        request = request or {"historyDays": 90, "asOfDate": "2021-01-01"}
        cutoff = pd.Timestamp(request.get("asOfDate", request.get("endDate", "2024-03-01")))
        if "endDate" in request:
            cutoff += pd.Timedelta(days=1)

        def ticker(symbol):
            class FakeTicker:
                def history(self, **kwargs):
                    calls.append((symbol, kwargs))
                    # Deliberately return unsliced data: the backend must enforce its boundary.
                    if symbol == "^TNX":
                        frame = pd.DataFrame(
                            {"Close": [1.0, 2.0, np.nan]},
                            index=[cutoff - pd.Timedelta(days=7),
                                   cutoff - pd.Timedelta(days=3),
                                   cutoff - pd.Timedelta(days=1)],
                        )
                        if no_rate:
                            frame = frame.iloc[0:0]
                    else:
                        idx = pd.bdate_range("2015-01-01", cutoff - pd.Timedelta(days=1))
                        k = np.arange(len(idx))
                        r = 0.0002 + 0.003 * np.sin(k) + (0.001 * np.cos(k) if symbol == "B" else 0)
                        frame = pd.DataFrame({"Close": 100 * np.cumprod(1 + r)}, index=idx)
                    if poison:
                        future = pd.DataFrame({"Close": [99999., 0.001]},
                                             index=[cutoff, cutoff + pd.Timedelta(days=4)])
                        frame = pd.concat([frame, future])
                    frame.index = pd.DatetimeIndex(frame.index).tz_localize("America/New_York")
                    return frame.iloc[::-1]
            return FakeTicker()

        with patch.object(api, "utc_today", return_value=date(2024, 3, 1)):
            req = api.OptimizeRequest(tickers=["A", "B"], **request)
            with patch.object(api.yf, "Ticker", side_effect=ticker):
                out = api.optimize(req)
        return out, calls

    def test_exact_window_and_historical_rate(self):
        out, calls = self.run_request()
        for symbol, kw in calls:
            self.assertEqual(kw["end"], "2021-01-01")
            self.assertEqual(kw["start"], "2020-12-01" if symbol == "^TNX" else "2020-10-03")
            self.assertTrue(kw["auto_adjust"])
        self.assertAlmostEqual(out.market["Rf"], 0.02)

    def test_cutoff_and_future_prices_do_not_affect_any_mode(self):
        for mode in ("max_sharpe", "min_variance", "target_return"):
            req = {"historyDays": 90, "asOfDate": "2021-01-01", "mode": mode}
            if mode == "target_return":
                probe, _ = self.run_request()
                req["targetReturn"] = np.mean([v["expectedCAPM"] for v in probe.perStock.values()])
            clean, _ = self.run_request(req)
            poisoned, _ = self.run_request(req, poison=True)
            self.assertEqual(clean.model_dump(), poisoned.model_dump())
            self.assertAlmostEqual(sum(clean.weights.values()), 1.)

    def test_missing_rate_does_not_fall_back_to_future(self):
        with self.assertRaises(HTTPException) as ctx:
            self.run_request(no_rate=True, poison=True)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Treasury", ctx.exception.detail)

    def test_weekend_cutoff(self):
        out, calls = self.run_request({"historyDays": 90, "asOfDate": "2021-01-03"})
        self.assertAlmostEqual(out.market["Rf"], 0.02)
        self.assertTrue(all(kw["end"] == "2021-01-03" for _, kw in calls))

    def test_year_window_leap_day(self):
        out, calls = self.run_request({"historyYears": 1, "asOfDate": "2020-02-29"})
        self.assertTrue(np.isfinite(out.market["Rm"]))
        for symbol, kw in calls:
            if symbol != "^TNX":
                self.assertEqual(kw["start"], "2019-02-28")
                self.assertEqual(kw["end"], "2020-02-29")

    def test_omitted_date_defaults_to_today(self):
        _, calls = self.run_request({"historyDays": 90})
        self.assertTrue(all(kw["end"] == "2024-03-01" for _, kw in calls))

    def test_invalid_and_future_dates(self):
        with patch.object(api, "utc_today", return_value=date(2024, 3, 1)):
            for value in ("not-a-date", "2021-02-30", "2024-03-02"):
                with self.assertRaises(ValidationError):
                    api.OptimizeRequest(tickers=["A"], asOfDate=value)

    def test_days_override_years(self):
        _, calls = self.run_request({"historyDays": 90, "historyYears": 10, "asOfDate": "2021-01-01"})
        self.assertEqual(calls[0][1]["start"], "2020-10-03")

    def test_inclusive_end_date_windows(self):
        _, calls = self.run_request({"startDate": "2020-10-03", "endDate": "2020-12-31"})
        for symbol, kwargs in calls:
            self.assertEqual(kwargs["end"], "2021-01-01")
            self.assertEqual(kwargs["start"], "2020-12-01" if symbol == "^TNX" else "2020-10-03")

        _, calls = self.run_request({"historyDays": 90, "endDate": "2020-12-31"})
        self.assertEqual(calls[0][1]["start"], "2020-10-03")
        self.assertTrue(all(kwargs["end"] == "2021-01-01" for _, kwargs in calls))


if __name__ == "__main__":
    unittest.main()
