"""Numerical regression and API contracts for research comparisons."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pydantic import ValidationError
from fastapi.testclient import TestClient
import backtest as bt
from test_backtest import sample_data, request
from version2 import app


class ResearchTests(unittest.TestCase):
    def test_one_return_year_is_json_serializable(self):
        import json
        nav = pd.Series([100., 101.], index=pd.to_datetime(['2024-12-31', '2025-01-02']))
        result = bt.annual_metrics(nav, [0.])
        self.assertIsNone(result[0]['volatility'])
        self.assertIsNone(result[0]['sharpe'])
        json.dumps(result, allow_nan=False)

    def test_negative_sharpe_stationary_point_is_not_accepted(self):
        w, _ = bt.solve_weights('max_sharpe', np.array([-.06, -.06]), np.eye(2)*.04, .04)
        self.assertAlmostEqual((w@np.array([-.06,-.06])-.04)/np.sqrt(w@(.04*np.eye(2))@w), -.5)
        self.assertAlmostEqual(w.max(), 1)

    def test_positive_sharpe_and_caps_match_dense_independent_grid(self):
        mu=np.array([.08,.16]);cov=np.array([[.03,.01],[.01,.09]])
        for cap in [1.,.6,.5]:
            w,_=bt.solve_weights('max_sharpe',mu,cov,.02,max_weight=cap)
            x=np.linspace(1-cap,cap,10001);ws=np.column_stack([x,1-x])
            scores=(ws@mu-.02)/np.sqrt(np.einsum('ij,jk,ik->i',ws,cov,ws))
            self.assertGreaterEqual((w@mu-.02)/np.sqrt(w@cov@w),scores.max()-1e-7)
            self.assertLessEqual(w.max(),cap+1e-8)

    def test_negative_capped_compares_vertices(self):
        mu=np.array([-.06,-.06]);cov=np.eye(2)*.04
        w,_=bt.solve_weights('max_sharpe',mu,cov,.04,max_weight=.6)
        self.assertAlmostEqual(w.max(),.6)

    def test_cap_and_target_feasibility(self):
        with self.assertRaises(ValidationError): request(maxWeight=.4)
        with self.assertRaises(bt.BacktestError):
            bt.solve_weights('target_return',np.array([.05,.15]),np.eye(2),.01,.14,.6)
        w,_=bt.solve_weights('target_return',np.array([.05,.15]),np.eye(2),.01,.1,.6)
        np.testing.assert_allclose(w,[.5,.5])

    def test_export_reproduces_sharpe_and_actual_cap_is_distinct(self):
        out=bt.run_backtest(request(maxWeight=.6,mode='rolling',frequency='monthly'),*sample_data())
        self.assertEqual(out['riskFreeDates'],out['dates'][1:])
        for s in out['strategies'].values():
            self.assertEqual(s['status'],'ok')
            rets=np.diff(s['nav'])/np.array(s['nav'][:-1]);excess=rets-out['riskFreeDaily']
            self.assertAlmostEqual(excess.mean()/excess.std(ddof=1)*np.sqrt(252),s['metrics']['sharpe'])
            self.assertLessEqual(s['metrics']['maxTargetWeight'],.6+1e-8)
            self.assertTrue(s['annualMetrics'][0]['partialYear'])

    def test_market_loader_and_assumptions(self):
        for market,symbols in [('HK',['0700.HK','9988.HK']),('CN_SH',['600519.SS','601318.SS']),('CN_SZ',['000001.SZ','000858.SZ'])]:
            prices,m,y=sample_data();prices.columns=symbols
            req=request(market=market,tickers=symbols)
            calls=[]
            def loader(symbol,start,end):
                calls.append(symbol)
                return m if symbol==bt.BACKTEST_MARKETS[market]['benchmark'] else prices[symbol]
            p,mm,yy=bt.prepare_data(req,loader)
            self.assertEqual(calls[0],bt.BACKTEST_MARKETS[market]['benchmark'])
            self.assertNotIn('^TNX',calls)
            out=bt.run_backtest(req,p,mm,yy)
            self.assertEqual(out['marketInfo']['currency'],bt.BACKTEST_MARKETS[market]['currency'])
            self.assertEqual(set(out['riskFreeDaily']),{0.})
            self.assertNotIn('^GSPC',' '.join(out['assumptions']))
            self.assertNotIn('^TNX',' '.join(out['assumptions']))

    def test_batch_uses_one_download_and_matches_standalone(self):
        req=bt.ComparisonRequest(base=request(),lookbacks=[90,365],frequencies=['quarterly'])
        data=sample_data()
        with patch('backtest_api.download_data',return_value=data) as load:
            response=TestClient(app).post('/api/backtest/compare',json=req.model_dump(mode='json'))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(load.call_count,1)
        self.assertEqual(load.call_args.args[0].lookbackDays,365)
        out=response.json();self.assertEqual(len(out['runs']),4)
        for run in out['runs']:
            self.assertEqual(run['status'],'ok')
            direct=bt.run_backtest(bt.comparison_base(req,run['lookbackDays'],run['frequency'],run['maxWeight']),*data)
            self.assertEqual(run['result'],direct)
        for row in out['rows']:
            if row['strategy']=='equal_weight':self.assertAlmostEqual(row['sharpeDifference'],0.)

    def test_year_boundary_includes_first_return(self):
        nav=pd.Series([100.,110.,121.],index=pd.to_datetime(['2020-12-31','2021-01-04','2021-01-05']))
        y=bt.annual_metrics(nav,[0.,0.])[0]
        self.assertAlmostEqual(y['totalReturn'],.21)
        self.assertEqual(y['returnObservations'],2)

    def test_comparison_retains_infeasible_strategy(self):
        req=bt.ComparisonRequest(base=request(targetReturn=9),lookbacks=[365],frequencies=['quarterly'])
        out=bt.run_comparison(req,*sample_data())
        self.assertTrue(all(r['status']=='failed' for r in out['rows'] if r['strategy']=='target_return'))
