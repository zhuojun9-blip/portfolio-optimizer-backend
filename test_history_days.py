import unittest
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd
from pydantic import ValidationError
from fastapi import HTTPException
import version2 as api


class HistoryDaysTests(unittest.TestCase):
    def test_validation(self):
        for value in (0, -2, 1.5, True):
            with self.assertRaises(ValidationError):
                api.OptimizeRequest(tickers=["A"], historyDays=value)
        self.assertEqual(api.OptimizeRequest(tickers=["A"]).historyYears, 5)

    def run_request(self, days=None, rows=60):
        calls = []
        def ticker(symbol):
            class FakeTicker:
                def history(self, **kwargs):
                    calls.append((symbol, kwargs))
                    if symbol == "^TNX":
                        return pd.DataFrame({"Close": [4.0]}, index=pd.to_datetime(["2025-01-01"]))
                    if "start" in kwargs:
                        idx = pd.bdate_range(kwargs["start"], kwargs["end"], inclusive="left")[:rows]
                    else:
                        idx = pd.bdate_range("2020-01-01", "2025-01-01")
                    k = np.arange(len(idx))
                    daily = 0.0003 + 0.002 * np.sin(k)
                    if symbol == "B":
                        daily = 0.0002 + 0.001 * np.cos(k)
                    return pd.DataFrame({"Close": 100 * np.cumprod(1 + daily)}, index=idx)
            return FakeTicker()
        with patch.object(api.yf, "Ticker", side_effect=ticker):
            out = api.optimize(api.OptimizeRequest(tickers=["A", "B"], historyDays=days))
        return out, calls

    def test_custom_day_windows(self):
        for days in (30, 90, 400):
            out, calls = self.run_request(days)
            windows = [kw for symbol, kw in calls if symbol != "^TNX"]
            self.assertTrue(all(kw == windows[0] for kw in windows))
            self.assertEqual((date.fromisoformat(windows[0]["end"]) - date.fromisoformat(windows[0]["start"])).days, days)
            self.assertAlmostEqual(sum(out.weights.values()), 1)
            self.assertTrue(all(np.isfinite(v) for v in out.stats.values()))
            idx = pd.bdate_range(windows[0]["start"], windows[0]["end"], inclusive="left")[:60]
            expected_rm = (0.0003 + 0.002 * np.sin(np.arange(len(idx))))[1:].mean() * 252
            self.assertAlmostEqual(out.market["Rm"], expected_rm)

    def test_insufficient_data(self):
        with self.assertRaises(HTTPException) as ctx:
            self.run_request(1, rows=1)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_legacy_year_request(self):
        out, calls = self.run_request()
        self.assertTrue(all(kw == {"period": "5y"} for symbol, kw in calls if symbol != "^TNX"))
        self.assertTrue(np.isfinite(out.market["Rm"]))


if __name__ == "__main__":
    unittest.main()
