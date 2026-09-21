import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError
import backtest as bt
import backtest_api
from version2 import app


def sample_data():
    idx = pd.bdate_range('2019-01-01', '2021-06-30')
    k = np.arange(len(idx))
    market = pd.Series(100 * np.cumprod(1 + .0003 + .004 * np.sin(k)), index=idx)
    prices = pd.DataFrame({'A': market * (1 + .001 * np.sin(k * 2)),
                           'B': 100 * np.cumprod(1 + .0002 + .003 * np.cos(k))}, index=idx)
    return prices, market, pd.Series(.01, index=idx)


def request(**kw):
    return bt.BacktestRequest(**dict({'tickers': ['A', 'B'], 'startDate': '2021-01-01',
        'endDate': '2021-06-30', 'lookbackDays': 365}, **kw))


class BacktestTests(unittest.TestCase):
    def test_single_fixed_holdings_and_drift(self):
        prices, market, rf = sample_data()
        out = bt.run_backtest(request(), prices, market, rf)
        eq = out['strategies']['equal_weight']
        holding = prices.loc[out['dates']]
        expected = 50000 * (holding['A'] / holding['A'].iloc[0] + holding['B'] / holding['B'].iloc[0])
        np.testing.assert_allclose(eq['nav'], expected)
        self.assertEqual(len(eq['rebalances']), 1)
        self.assertNotEqual(eq['weights'][-1], [.5, .5])
        self.assertEqual(eq['metrics']['transactionCosts'], 0.)
        self.assertEqual(eq['metrics']['turnover'], 0.)

    def test_rebalance_day_earns_old_weights_then_trades(self):
        idx = pd.to_datetime(['2021-01-29', '2021-02-01', '2021-02-02'])
        prices = pd.DataFrame({'A': [100., 120., 120.], 'B': [100., 100., 110.]}, index=idx)
        market = prices['A']
        rf = pd.Series(.01, index=pd.date_range('2020-12-01', '2021-02-02'))
        fake_inputs = (np.array([.1, .2]), np.eye(2) * .1, .01,
                       {'trainingEnd': '2021-01-28', 'observations': 50})
        with patch.object(bt, 'estimate_inputs', return_value=fake_inputs):
            out = bt.run_backtest(request(startDate='2021-01-29', endDate='2021-02-02',
                mode='rolling', frequency='monthly', initialCapital=100), prices, market, rf)
        eq = out['strategies']['equal_weight']
        # Friday 50/50 -> Monday 60/50 -> rebalance 55/55 -> Tuesday 55/60.5.
        np.testing.assert_allclose(eq['nav'], [100, 110, 115.5])
        self.assertAlmostEqual(eq['rebalances'][1]['preTradeWeights']['A'], 60/110)
        self.assertAlmostEqual(eq['rebalances'][1]['turnover'], 60/110 - .5)

    def test_future_prices_do_not_change_initial_weights(self):
        prices, market, rf = sample_data()
        clean = bt.run_backtest(request(), prices, market, rf)
        changed = prices.copy()
        changed.loc['2021-01-01':, 'A'] *= 10
        future = bt.run_backtest(request(), changed, market, rf)
        for s in clean['strategies']:
            self.assertEqual(clean['strategies'][s]['rebalances'][0], future['strategies'][s]['rebalances'][0])

    def test_estimator_excludes_future_market_and_yield(self):
        p, m, r = sample_data()
        original = bt.estimate_inputs(p, m, r, '2020-01-01', '2021-01-01')
        p.loc['2021-01-01':] *= 100
        m.loc['2021-01-01':] *= 100
        r.loc['2021-01-01':] = 99
        altered = bt.estimate_inputs(p, m, r, '2020-01-01', '2021-01-01')
        for a, b in zip(original[:3], altered[:3]):
            np.testing.assert_allclose(a, b)

    def test_rolling_training_end_and_capital_continuity(self):
        data = sample_data()
        for frequency, count in [('monthly', 6), ('quarterly', 2), ('weekly', 27)]:
            out = bt.run_backtest(request(mode='rolling', frequency=frequency), *data)
            eq = out['strategies']['equal_weight']
            self.assertEqual(len(eq['rebalances']), count)
            for r in eq['rebalances']:
                self.assertLess(r['trainingEnd'], r['executionDate'])
            self.assertNotEqual(eq['nav'][-1], 100000)

    def test_fixed_target_infeasible_has_no_misleading_metrics(self):
        out = bt.run_backtest(request(targetReturn=10), *sample_data())
        failed = out['strategies']['target_return']
        self.assertEqual(failed['status'], 'failed')
        self.assertIsNone(failed['metrics'])
        self.assertNotIn('nav', failed)
        self.assertEqual(out['strategies']['equal_weight']['status'], 'ok')

    def test_solver_failure_is_not_equal_weight_fallback(self):
        with patch.object(bt, 'minimize', return_value=SimpleNamespace(success=False, x=np.array([.5,.5]), message='failed')):
            with self.assertRaises(bt.BacktestError):
                bt.solve_weights('max_sharpe', np.array([.1,.2]), np.eye(2), .01)

    def test_default_target_matches_equal_expected_return(self):
        mu, cov = np.array([.08, .15]), np.array([[.04,.01],[.01,.09]])
        w, _ = bt.solve_weights('target_return', mu, cov, .02)
        self.assertAlmostEqual(w @ mu, mu.mean(), places=6)

    def test_metrics_known_drawdown_and_sharpe(self):
        nav = pd.Series([100., 120., 90., 110.], index=pd.to_datetime(['2021-01-01','2021-01-04','2021-01-05','2021-01-06']))
        m = bt.realized_metrics(nav, [0.,0.,0.])
        returns = np.array([.2, -.25, 110/90-1])
        self.assertAlmostEqual(m['totalReturn'], .1)
        self.assertAlmostEqual(m['maxDrawdown'], -.25)
        self.assertAlmostEqual(m['sharpe'], returns.mean()/returns.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(m['cagr'], 1.1**(365.25/5)-1)

    def test_rate_not_future_filled(self):
        rates = pd.Series([.04], index=pd.to_datetime(['2021-01-04']))
        with self.assertRaises(bt.BacktestError):
            bt.rate_before(rates, '2021-01-04')
        self.assertEqual(bt.rate_before(rates, '2021-01-05'), .04)
        with self.assertRaises(bt.BacktestError):
            bt.rate_before(rates, '2021-03-01')

    def test_explicit_estimation_gap(self):
        out = bt.run_backtest(request(estimationStart='2019-01-01', estimationEnd='2020-10-01'), *sample_data())
        self.assertEqual(out['strategies']['equal_weight']['rebalances'][0]['trainingEnd'], '2020-10-01')

    def test_missing_data_rejected(self):
        prices, market, rf = sample_data()
        prices.iloc[20,0] = np.nan
        with self.assertRaises(bt.BacktestError):
            bt.run_backtest(request(), prices, market, rf)

    def test_validation(self):
        for kw in [{'tickers':['A','A']}, {'tickers':['0700.HK','A']}, {'startDate':'2021-06-30'},
                   {'mode':'rolling','estimationStart':'2020-01-01','estimationEnd':'2020-12-31'},
                   {'estimationStart':'2021-01-01','estimationEnd':'2021-02-01'}, {'initialCapital':float('nan')}]:
            with self.assertRaises(ValidationError):
                request(**kw)

    def test_api_contract_and_dashboard(self):
        with TestClient(app) as client:
            with patch.object(backtest_api, 'download_data', return_value=sample_data()):
                response = client.post('/api/backtest', json=request().model_dump(mode='json'))
            self.assertEqual(response.status_code, 200)
            json.dumps(response.json(), allow_nan=False)
            self.assertIn('equal_weight', response.json()['strategies'])
            page = client.get('/backtest')
            self.assertEqual(page.status_code, 200)
            self.assertIn('Run backtest', page.text)
            self.assertEqual(client.post('/api/backtest', json={}).status_code, 422)
            with patch.object(backtest_api, 'download_data', side_effect=RuntimeError('provider problem')):
                self.assertEqual(client.post('/api/backtest', json=request().model_dump(mode='json')).status_code, 502)

    def test_loader_rejects_non_usd_metadata(self):
        prices, market, rf = sample_data()
        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol
            def history(self, **kwargs):
                return {"^GSPC": market, "A": prices["A"]}[self.symbol].to_frame("Close")
            def get_history_metadata(self):
                return {"currency": "HKD"}
        with patch.object(backtest_api.yf, "Ticker", FakeTicker):
            with self.assertRaisesRegex(bt.BacktestError, "requires verified USD"):
                backtest_api.download_data(request())

    def test_loader_reports_the_symbol_when_provider_fails(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, **kwargs):
                raise RuntimeError("provider outage")

        with patch.object(backtest_api.yf, "Ticker", FakeTicker):
            with self.assertRaisesRegex(backtest_api.DataProviderError, r"\^GSPC"):
                backtest_api.download_data(request())

    def test_loader_downloads_once_per_symbol(self):
        prices, market, rf = sample_data()
        calls = []
        def loader(symbol, start, end):
            calls.append(symbol)
            return {'A':prices['A'],'B':prices['B'],'^GSPC':market,'^TNX':rf*100}[symbol]
        data = bt.prepare_data(request(), loader)
        self.assertCountEqual(calls, ['A','B','^GSPC','^TNX'])
        self.assertTrue(data[0].index.equals(data[1].index))

    def test_hong_kong_uses_hang_seng_and_local_cash_baseline(self):
        req = request(tickers=["0700.HK", "9988.HK"], market="HK")
        prices, market, _ = sample_data()
        calls = []

        def loader(symbol, start, end):
            calls.append(symbol)
            return {
                "0700.HK": prices["A"],
                "9988.HK": prices["B"],
                "^HSI": market,
            }[symbol]

        _, loaded_market, rates = bt.prepare_data(req, loader)
        self.assertCountEqual(calls, ["0700.HK", "9988.HK", "^HSI"])
        self.assertGreater(len(loaded_market), 0)
        self.assertTrue(loaded_market.index.isin(market.index).all())
        self.assertTrue((rates == 0).all())

    def test_mainland_market_ticker_rules(self):
        req = request(tickers=["600519.SS", "601318.SS"], market="CN_SH")
        self.assertEqual(req.market, "CN_SH")
        with self.assertRaises(ValidationError):
            request(tickers=["AAPL", "601318.SS"], market="CN_SH")

    def test_user_holdings_are_compared_as_buy_and_hold(self):
        prices, market, rates = sample_data()
        req = request(userWeights={"A": 0.75, "B": 0.25})
        output = bt.run_backtest(req, prices, market, rates)
        holdings = output["strategies"]["user_holdings"]
        self.assertEqual(holdings["status"], "ok")
        self.assertEqual(len(holdings["rebalances"]), 1)
        self.assertEqual(holdings["rebalances"][0]["weights"], {"A": 0.75, "B": 0.25})
        self.assertNotEqual(holdings["weights"][-1], [0.75, 0.25])

    def test_user_holdings_must_match_selected_tickers(self):
        with self.assertRaises(ValidationError):
            request(userWeights={"A": 1.0})
        with self.assertRaises(ValidationError):
            request(userWeights={"A": 0.6, "B": 0.3})


if __name__ == '__main__':
    unittest.main()
