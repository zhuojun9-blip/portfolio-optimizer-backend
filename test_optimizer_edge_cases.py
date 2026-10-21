"""Deterministic policy, feasibility, and numerical regression tests."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
import backtest as bt
import version2 as api
from test_backtest import sample_data, request


class OptimizerEdgeTests(unittest.TestCase):
    def test_cap_blocks_positive_excess_stock(self):
        for cap in [.5, .6]:
            with self.assertRaisesRegex(bt.BacktestError, "No feasible portfolio"):
                bt.solve_weights("max_sharpe", [.05, -.10], np.eye(2), .02, max_weight=cap)

    def test_zero_excess_rejected(self):
        with self.assertRaisesRegex(bt.BacktestError, "No feasible portfolio"):
            bt.solve_weights("max_sharpe", [.02, .02], np.eye(2), .02)

    def test_main_api_negative_excess_rejected(self):
        for short in [False, True]:
            with self.assertRaises(HTTPException) as exc:
                api.optimize_max_sharpe(np.array([-.02, -.02]), np.eye(2), .01, short)
            self.assertEqual(exc.exception.status_code, 422)
            self.assertIn("No feasible portfolio", exc.exception.detail)

    def test_short_positions_can_produce_positive_excess(self):
        # Three negative-excess assets, but 1,1,-1 can beat cash.
        mu = np.array([.03, .02, -.20])
        w = api.optimize_max_sharpe(mu, np.eye(3), .04, True)
        self.assertGreater(w @ mu, .04)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertLessEqual(np.max(np.abs(w)), 1 + 1e-8)
        # Independent feasible grid checks the transformed short-sale constraints.
        vals = np.linspace(-1, 1, 101)
        scores = [(v @ mu - .04) / np.linalg.norm(v)
                  for a in vals for b in vals
                  if abs(1-a-b) <= 1
                  for v in [np.array([a,b,1-a-b])]]
        self.assertGreaterEqual((w @ mu - .04) / np.linalg.norm(w), max(scores)-1e-6)

    def test_successful_but_invalid_solver_is_rejected(self):
        for weights in [[.2,.2], [1.2,-.2], [np.nan,np.nan]]:
            result = SimpleNamespace(success=True, x=np.array(weights), message="success")
            with patch.object(api, "minimize", return_value=result):
                with self.assertRaises(HTTPException):
                    api.optimize_min_variance(np.array([.1,.2]), np.eye(2), False)

    def test_identical_targets_do_not_add_redundant_constraint(self):
        w = api.optimize_target_return(np.array([.1,.1]), np.diag([1.,4.]), .1, False)
        np.testing.assert_allclose(w, [.8,.2], atol=1e-5)

    def test_target_mode_requires_target_before_download(self):
        with patch.object(api.yf, "Ticker") as download:
            response = TestClient(api.app).post("/api/optimize",
                json={"tickers":["AAPL","JPM"],"mode":"target_return"})
        self.assertEqual(response.status_code, 422)
        download.assert_not_called()

    def test_historical_foreign_rate_cannot_use_today(self):
        with patch.object(api, "latest_local_government_yield") as rate:
            with self.assertRaises(HTTPException) as exc:
                api.optimize(api.OptimizeRequest(tickers=["0700.HK", "9988.HK"],
                    market="HK", historyDays=90, asOfDate="2021-01-01"))
        self.assertEqual(exc.exception.status_code, 422)
        self.assertIn("historical local risk-free rate", exc.exception.detail)
        rate.assert_not_called()

    def test_zero_variance_valid_except_sharpe(self):
        for strategy in ["min_variance", "target_return"]:
            w,_ = bt.solve_weights(strategy, [.1,.1], np.zeros((2,2)), .02, .1)
            self.assertAlmostEqual(w.sum(), 1)
        self.assertIsNone(api.portfolio_stats(np.array([.5,.5]),
            np.array([.1,.1]), np.zeros((2,2)), .02)["sharpe"])
        with self.assertRaises(bt.BacktestError):
            bt.solve_weights("max_sharpe", [.1,.1], np.zeros((2,2)), .02)

    def test_bad_covariance_rejected(self):
        with self.assertRaises(HTTPException):
            api.optimize_min_variance(np.array([.1,.2]), np.diag([1.,-1.]), False)

    def test_backtest_keeps_failure_visible_and_other_strategies(self):
        data = sample_data()
        original = bt.estimate_inputs
        def negative(*args):
            mu,cov,rf,info = original(*args)
            return np.full_like(mu, rf-.02),cov,rf,info
        with patch.object(bt, "estimate_inputs", side_effect=negative):
            out = bt.run_backtest(request(), *data)
        failed = out["strategies"]["max_sharpe"]
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["metrics"])
        self.assertEqual(failed["failureDate"], out["dates"][0])
        for strategy in ["min_variance","target_return","equal_weight"]:
            self.assertEqual(out["strategies"][strategy]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
