// Optimizer.js
import React, { useState } from "react";
import axios from "axios";
const API =
  process.env.REACT_APP_API_URL || "http://localhost:8000/api/optimize";
const TODAY = new Date().toISOString().slice(0, 10);
const MARKETS = [
  { value: "US", label: "United States", suffix: "" },
  { value: "HK", label: "Hong Kong", suffix: ".HK" },
  { value: "CN_SH", label: "China - Shanghai", suffix: ".SS" },
  { value: "CN_SZ", label: "China - Shenzhen", suffix: ".SZ" },
  { value: "JP", label: "Japan", suffix: ".T" },
  { value: "CA", label: "Canada", suffix: ".TO" },
  { value: "UK", label: "United Kingdom", suffix: ".L" },
  { value: "IN", label: "India", suffix: ".NS" },
];
function Field({ label, children }) {
  return <label className="field"><span>{label}</span>{children}</label>;
}
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = value => Number.isFinite(value) ? value.toFixed(3) : "—";

function getOptimizationError(error) {
  const detail = error.response?.data?.detail;
  if (Array.isArray(detail)) {
    return `Invalid request: ${detail[0]?.msg || "Check the input values."}`;
  }
  if (typeof detail === "string") {
    return detail;
  }
  if (error.code === "ECONNABORTED") {
    return "The optimizer timed out. Please try again.";
  }
  if (error.request) {
    return "No response was received from the optimizer API. It may be waking up, unavailable, or blocked by the browser. Wait 30 seconds, then try again.";
  }
  return "Optimization failed. Please try again.";
}

function yearsAgoDate(years) {
  const date = new Date();
  date.setUTCFullYear(date.getUTCFullYear() - years);
  return date.toISOString().slice(0, 10);
}

function tickerForMarket(ticker, marketValue) {
  const symbol = ticker.trim().toUpperCase();
  const market = MARKETS.find(({ value }) => value === marketValue);
  if (!symbol || !market || !market.suffix || symbol.includes(".") || symbol.startsWith("^")) {
    return symbol;
  }
  if (market.value === "HK" && /^\d+$/.test(symbol)) {
    return `${symbol.padStart(4, "0")}${market.suffix}`;
  }
  return `${symbol}${market.suffix}`;
}

function tickerFormatError(ticker, marketValue) {
  const symbol = ticker.trim().toUpperCase();
  if (!symbol) {
    return "Enter a ticker symbol.";
  }
  if (marketValue === "HK" && !/^\d{1,5}(\.HK)?$/.test(symbol)) {
    return "Hong Kong tickers must be numeric, for example 700 or 9988.";
  }
  if ((marketValue === "CN_SH" || marketValue === "CN_SZ") && !/^\d{6}(\.(SS|SZ))?$/.test(symbol)) {
    return "China tickers must be six digits, for example 600519 or 000001.";
  }
  if (marketValue === "JP" && !/^\d{4}(\.T)?$/.test(symbol)) {
    return "Japan tickers must be four digits, for example 7203.";
  }
  return "";
}

export default function Optimizer() {
  // ---- form state
  const [tickers, setTickers] = useState(["AAPL", "MSFT", "NVDA"]);
  const [newTicker, setNewTicker] = useState("");
  const [market, setMarket] = useState("US");
  const [mode, setMode] = useState("max_sharpe");
  const [targetReturn, setTargetReturn] = useState("0.12");
  const [windowMode, setWindowMode] = useState("date_range");
  const [startDate, setStartDate] = useState(() => yearsAgoDate(5));
  const [endDate, setEndDate] = useState(TODAY);
  const [historyDays, setHistoryDays] = useState("90");

  // ---- result state
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);

  const addTicker = () => {
    const formatError = tickerFormatError(newTicker, market);
    if (formatError) {
      setErr(formatError);
      return;
    }
    const t = tickerForMarket(newTicker, market);
    if (t && !tickers.includes(t)) setTickers([...tickers, t]);
    setNewTicker("");
  };
  const removeTicker = (t) => setTickers((xs) => xs.filter((x) => x !== t));
  const changeMarket = (value) => {
    setMarket(value);
    setTickers([]);
    setRes(null);
    setErr("");
  };

  const runOptimize = async () => {
    setErr("");
    const lookbackDays = historyDays ? Number(historyDays) : null;
    const target = Number(targetReturn);

    if (tickers.length === 0) {
      setErr("Add at least one ticker.");
      return;
    }
    if (!endDate) {
      setErr("Select an end date.");
      return;
    }
    if (endDate > TODAY) {
      setErr("End date cannot be in the future.");
      return;
    }
    if (windowMode === "date_range" && !startDate) {
      setErr("Select a start date.");
      return;
    }
    if (windowMode === "date_range" && startDate > endDate) {
      setErr("Start date must be on or before end date.");
      return;
    }
    if (windowMode === "lookback" && (!Number.isInteger(lookbackDays) || lookbackDays < 1)) {
      setErr("Lookback days must be a whole number greater than 0.");
      return;
    }
    if (mode === "target_return" && !Number.isFinite(target)) {
      setErr("Enter a valid target return.");
      return;
    }

    setLoading(true);
    setRes(null);
    try {
      const payload = {
        tickers,
        mode,
        market,
        targetReturn: mode === "target_return" ? target : null,
        ...(windowMode === "date_range"
          ? { startDate, endDate }
          : { historyDays: lookbackDays, endDate }),
      };
      const r = await axios.post(API, payload, { timeout: 120000 });
      setRes(r.data);
    } catch (e) {
      console.error(e);
      setErr(getOptimizationError(e));
    } finally {
      setLoading(false);
    }
  };

  const allocations = res ? Object.entries(res.weights || {}) : [];
  const strategyName = {max_sharpe: "Max Sharpe", min_variance: "Min variance", target_return: "Target return"}[mode];
  return (
    <div className="workspace">
      <header className="page-heading">
        <div><p className="eyebrow">CAPM · Mean–variance</p><h1>Build your allocation</h1>
        <p>Estimate portfolio weights, then compare the model with equal weight.</p></div>
        <span className="context-note">Model estimates · Annualized</span>
      </header>
      <main className="optimizer-layout">
        <section className="panel settings-panel" aria-labelledby="settings-title">
          <h2 id="settings-title">Portfolio settings</h2>
          <div className="form-stack">
            <Field label="Portfolio market"><select value={market} onChange={e => changeMarket(e.target.value)}>
              {MARKETS.map(({value,label}) => <option key={value} value={value}>{label}</option>)}
            </select></Field>
            <div><label className="field" htmlFor="ticker-entry"><span>Add ticker</span></label>
              <div className="ticker-entry"><input id="ticker-entry" value={newTicker} onChange={e => setNewTicker(e.target.value)} onKeyDown={e => {if(e.key === "Enter") {e.preventDefault();addTicker();}}} placeholder={market === "HK" ? "e.g., 700" : "e.g., AMZN"}/><button className="secondary" onClick={addTicker}>Add</button></div>
            </div>
            <div><p className="field-label">Selected tickers <span className="muted">({tickers.length})</span></p>
              <div className="ticker-list">{tickers.map(t => <span className="ticker-tag" key={t}>{t}<button onClick={() => removeTicker(t)} aria-label={`Remove ${t}`}>×</button></span>)}</div>
              {!tickers.length && <p className="help">Add a ticker from your selected market.</p>}
            </div>
            <div className="form-divider"/>
            <Field label="Optimization objective"><select value={mode} onChange={e => setMode(e.target.value)}><option value="max_sharpe">Max Sharpe</option><option value="min_variance">Min variance</option><option value="target_return">Target return</option></select></Field>
            <Field label="Target return (decimal)"><input type="number" step="0.01" disabled={mode !== "target_return"} value={targetReturn} onChange={e => setTargetReturn(e.target.value)}/><small className="help">0.12 means 12% annually. Used for Target return.</small></Field>
            <div className="form-divider"/>
            <div><p className="field-label">Estimation window</p><div className="segmented" aria-label="Historical window">
              <button aria-pressed={windowMode === "date_range"} onClick={() => setWindowMode("date_range")}>Date range</button>
              <button aria-pressed={windowMode === "lookback"} onClick={() => setWindowMode("lookback")}>Lookback</button>
            </div></div>
            <div className="field-pair">
              {windowMode === "date_range" ? <Field label="Begin date"><input type="date" value={startDate} max={endDate || TODAY} onChange={e => setStartDate(e.target.value)}/></Field> : <Field label="Lookback days"><input type="number" min={1} value={historyDays} onChange={e => setHistoryDays(e.target.value)}/></Field>}
              <Field label="End date"><input type="date" value={endDate} min={windowMode === "date_range" ? startDate : undefined} max={TODAY} onChange={e => setEndDate(e.target.value)}/></Field>
            </div>
            <button className="primary run-button" onClick={runOptimize} disabled={loading}>{loading ? "Optimizing…" : "Run optimizer"}</button>
            {err && <div className="error-message" role="alert">{err}</div>}
          </div>
        </section>
        <div className="result-stack" aria-busy={loading} aria-live="polite">
          <section className="panel allocation-panel" aria-labelledby="allocation-title">
            <div className="panel-heading"><h2 id="allocation-title">Allocation weights</h2><span className="muted">{res ? `${allocations.length} assets` : "Awaiting calculation"}</span></div>
            {res && allocations.length ? <>
              <div className="allocation-bars" aria-label="Portfolio weights">{allocations.map(([ticker,weight]) => <div className="allocation-row" key={ticker}><span className="ticker-name">{ticker}</span><div className="bar-track"><div className="bar-fill" style={{width:`${Math.max(0, Math.min(100, weight*100))}%`}}/></div><span className="number">{percent(weight)}</span></div>)}</div>
              <div className="table-scroll"><table><caption className="sr-only">Stock allocation and model estimates</caption><thead><tr><th scope="col">Ticker</th><th scope="col">Weight</th><th scope="col">Beta</th><th scope="col">CAPM return</th><th scope="col">CAGR (approx.)</th></tr></thead><tbody>{res.usedTickers.map(t => {const info=res.perStock?.[t] || {};return <tr key={t}><th scope="row">{t}</th><td>{percent(res.weights[t] ?? 0)}</td><td>{decimal(info.beta)}</td><td>{percent(info.expectedCAPM)}</td><td>{percent(info.cagrApprox)}</td></tr>})}</tbody></table></div>
            </> : <div className="allocation-empty"><div className="empty-mark" aria-hidden="true">β</div><h3>{loading ? "Estimating your allocation" : "Your allocation starts here"}</h3><p>{loading ? "Loading prices and solving the allocation. The API may take a moment to wake up." : "Choose your stocks, objective and estimation window. Run the optimizer to see weights and compare model estimates."}</p><span className="empty-context">{tickers.length} selected assets · {strategyName}</span></div>}
          </section>
          <section className="panel" aria-labelledby="stats-title"><div className="panel-heading"><h2 id="stats-title">Portfolio estimates</h2><span className="muted">Annualized</span></div>
            {res ? <div className="table-scroll"><table className="estimate-table"><thead><tr><th scope="col">Metric</th><th scope="col">Optimized</th><th scope="col">Equal weight</th></tr></thead><tbody>
              <tr><th scope="row">Expected return</th><td>{percent(res.stats.expected_return)}</td><td>{percent(res.baseline.expected_return)}</td></tr>
              <tr><th scope="row">Volatility</th><td>{percent(res.stats.volatility)}</td><td>{percent(res.baseline.volatility)}</td></tr>
              <tr><th scope="row">Sharpe ratio</th><td>{decimal(Number(res.stats.sharpe))}</td><td>{decimal(Number(res.baseline.sharpe))}</td></tr>
            </tbody></table></div> : <p className="muted">Expected return, volatility and Sharpe will appear beside the equal-weight baseline.</p>}
            <p className="help method-note">Model estimates are not realized performance. Use Backtest to evaluate subsequent historical returns.</p>
          </section>
          <section className="panel market-panel"><h2>Market snapshot</h2>{res ? <div className="market-values"><div><span>Benchmark return</span><strong className="number">{percent(res.market.Rm)}</strong></div><div><span>Risk-free rate</span><strong className="number">{percent(res.market.Rf)}</strong></div></div> : <p className="muted">Benchmark return and risk-free rate appear after calculation.</p>}</section>
        </div>
      </main>
    </div>
  );
}
