import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

import version2 as api


class MarketBenchmarkTests(unittest.TestCase):
    def fake_ticker(self, calls):
        def ticker(symbol):
            class FakeTicker:
                def history(self, **kwargs):
                    calls.append((symbol, kwargs))
                    index = pd.bdate_range(kwargs["start"], kwargs["end"], inclusive="left")
                    step = np.arange(len(index))
                    returns = 0.0004 + 0.002 * np.sin(step)
                    return pd.DataFrame({"Close": 100 * np.cumprod(1 + returns)}, index=index)

            return FakeTicker()

        return ticker

    def test_hong_kong_uses_hang_seng_and_automatic_local_rate(self):
        calls = []
        request = api.OptimizeRequest(
            tickers=["0700.HK", "9988.HK"],
            market="HK",
            historyDays=90,
            asOfDate="2021-01-01",
        )
        with patch.object(api.yf, "Ticker", side_effect=self.fake_ticker(calls)), patch.object(
            api, "latest_local_government_yield", return_value=0.03
        ) as rate_lookup:
            result = api.optimize(request)

        symbols = [symbol for symbol, _ in calls]
        self.assertIn("^HSI", symbols)
        self.assertNotIn("^GSPC", symbols)
        self.assertNotIn("^TNX", symbols)
        rate_lookup.assert_called_once_with("HK")
        self.assertEqual(result.market["Rf"], 0.03)

    def test_non_us_market_reports_missing_rate_provider(self):
        calls = []
        request = api.OptimizeRequest(
            tickers=["0700.HK"],
            market="HK",
            historyDays=90,
            asOfDate="2021-01-01",
        )
        with patch.object(api.yf, "Ticker", side_effect=self.fake_ticker(calls)):
            with self.assertRaises(HTTPException) as context:
                api.optimize(request)

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("TRADING_ECONOMICS_API_KEY", context.exception.detail)

    def test_unknown_ticker_has_a_clear_error(self):
        def ticker(symbol):
            class FakeTicker:
                def history(self, **kwargs):
                    if symbol == "BAD.SS":
                        return pd.DataFrame()
                    index = pd.bdate_range("2020-01-01", "2021-01-01")
                    return pd.DataFrame({"Close": np.arange(len(index)) + 100}, index=index)

            return FakeTicker()

        request = api.OptimizeRequest(tickers=["BAD.SS"], market="US")
        with patch.object(api.yf, "Ticker", side_effect=ticker):
            with self.assertRaises(HTTPException) as context:
                api.optimize(request)

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("No price data was found for ticker 'BAD.SS'", context.exception.detail)


if __name__ == "__main__":
    unittest.main()