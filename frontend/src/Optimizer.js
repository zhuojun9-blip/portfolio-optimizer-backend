// Optimizer.js
import React, { useState } from "react";
import axios from "axios";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from "recharts";

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
const COLORS = [
  "#2563eb",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#14b8a6",
  "#f43f5e",
  "#22c55e",
  "#eab308",
  "#0ea5e9",
];

function Section({ title, children }) {
  return (
    <div style={styles.card}>
      <div style={styles.cardTitle}>{title}</div>
      {children}
    </div>
  );
}

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

  const pieData = res
    ? Object.entries(res.weights || {}).map(([name, value]) => ({
        name,
        value,
      }))
    : [];

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={{ margin: 0 }}>Portfolio Optimizer</h1>
        <div style={{ color: "#6b7280" }}>
          CAPM + Mean-Variance (Max Sharpe / Min Variance / Target Return)
        </div>
      </header>

      <main style={styles.main}>
        {/* Left: Inputs */}
        <Section title="Inputs">
          <div style={{ display: "grid", gap: 12 }}>
            {/* Ticker entry */}
            <div>
              <div style={styles.label}>Portfolio market</div>
              <select
                value={market}
                onChange={(e) => changeMarket(e.target.value)}
                style={styles.input}
              >
                {MARKETS.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <div style={styles.label}>Add ticker</div>
              <div style={styles.tickerEntry}>
                <input
                  value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value)}
                  placeholder={market === "HK" ? "e.g., 700" : "e.g., AMZN"}
                  style={styles.input}
                />
                <button onClick={addTicker} style={styles.primaryBtn}>
                  Add
                </button>
              </div>
            </div>

            {/* Current tickers */}
            <div>
              <div style={styles.label}>Selected tickers</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {tickers.map((t) => (
                  <span key={t} style={styles.chip}>
                    {t}
                    <button
                      onClick={() => removeTicker(t)}
                      style={styles.chipX}
                      title="Remove"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>

            {/* Mode + target */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div>
                <div style={styles.label}>Mode</div>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  style={styles.input}
                >
                  <option value="max_sharpe">Max Sharpe</option>
                  <option value="min_variance">Min Variance</option>
                  <option value="target_return">Target Return</option>
                </select>
              </div>
              <div>
                <div style={styles.label}>Target Return (decimal)</div>
                <input
                  type="number"
                  step="0.01"
                  disabled={mode !== "target_return"}
                  value={targetReturn}
                  onChange={(e) => setTargetReturn(e.target.value)}
                  style={{ ...styles.input, opacity: mode === "target_return" ? 1 : 0.6 }}
                />
              </div>
            </div>

            {/* Historical window */}
            <div>
              <div style={styles.label}>Historical window</div>
              <div style={styles.segmentedControl}>
                <button
                  type="button"
                  onClick={() => setWindowMode("date_range")}
                  style={{ ...styles.segmentButton, ...(windowMode === "date_range" ? styles.segmentButtonActive : {}) }}
                >
                  Date range
                </button>
                <button
                  type="button"
                  onClick={() => setWindowMode("lookback")}
                  style={{ ...styles.segmentButton, ...(windowMode === "lookback" ? styles.segmentButtonActive : {}) }}
                >
                  Lookback
                </button>
              </div>
            </div>

            {windowMode === "date_range" ? (
              <div style={styles.formGrid}>
                <div>
                  <div style={styles.label}>Begin date</div>
                  <input
                    type="date"
                    value={startDate}
                    max={endDate || TODAY}
                    onChange={(e) => setStartDate(e.target.value)}
                    style={styles.input}
                  />
                </div>
                <div>
                  <div style={styles.label}>End date</div>
                  <input
                    type="date"
                    value={endDate}
                    min={startDate}
                    max={TODAY}
                    onChange={(e) => setEndDate(e.target.value)}
                    style={styles.input}
                  />
                </div>
              </div>
            ) : (
              <div style={styles.formGrid}>
                <div>
                  <div style={styles.label}>Lookback (calendar days)</div>
                  <input
                    type="number"
                    min={1}
                    value={historyDays}
                    onChange={(e) => setHistoryDays(e.target.value)}
                    style={styles.input}
                  />
                </div>
                <div>
                  <div style={styles.label}>End date</div>
                  <input
                    type="date"
                    value={endDate}
                    max={TODAY}
                    onChange={(e) => setEndDate(e.target.value)}
                    style={styles.input}
                  />
                </div>
              </div>
            )}

            <button onClick={runOptimize} style={styles.primaryBtn} disabled={loading}>
              {loading ? "Optimizing..." : "Run Optimizer"}
            </button>

            {err && <div style={styles.error}>{err}</div>}
          </div>
        </Section>

        {/* Right: Results */}
        <div style={{ display: "grid", gap: 16 }}>
          <Section title="Weights">
            {res && pieData.length > 0 ? (
              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 16 }}>
                {/* Table */}
                <div style={{ overflowX: "auto" }}>
                  <table style={styles.table}>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                        <th>β</th>
                        <th>CAPM Exp.</th>
                        <th>CAGR (approx)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {res.usedTickers.map((t) => {
                        const w = res.weights[t] ?? 0;
                        const info = (res.perStock && res.perStock[t]) || {};
                        const beta = info.beta;
                        const capm = info.expectedCAPM;
                        const cagr = info.cagrApprox;
                        return (
                          <tr key={t}>
                            <td>{t}</td>
                            <td>{(w * 100).toFixed(2)}%</td>
                            <td>{Number.isFinite(beta) ? beta.toFixed(3) : "—"}</td>
                            <td>{Number.isFinite(capm) ? (capm * 100).toFixed(2) + "%" : "—"}</td>
                            <td>{Number.isFinite(cagr) ? (cagr * 100).toFixed(2) + "%" : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Pie */}
                <div style={{ width: 320, height: 260 }}>
                  <PieChart width={320} height={260}>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                    >
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v) => (v * 100).toFixed(2) + "%"} />
                    <Legend />
                  </PieChart>
                </div>
              </div>
            ) : (
              <div style={{ color: "#6b7280" }}>No results yet.</div>
            )}
          </Section>

          <Section title="Portfolio Stats (annualized)">
            {res ? (
              <div style={{ display: "grid", gap: 8 }}>
                <div style={styles.kv}>
                  <div>Expected Return</div>
                  <div>{(res.stats.expected_return * 100).toFixed(2)}%</div>
                </div>
                <div style={styles.kv}>
                  <div>Volatility</div>
                  <div>{(res.stats.volatility * 100).toFixed(2)}%</div>
                </div>
                <div style={styles.kv}>
                  <div>Sharpe (Rf {(res.market.Rf * 100).toFixed(2)}%)</div>
                  <div>{Number(res.stats.sharpe).toFixed(3)}</div>
                </div>

                <hr style={{ border: "none", borderTop: "1px solid #e5e7eb" }} />

                <div style={{ color: "#6b7280" }}>Equal-Weight Baseline</div>
                <div style={styles.kv}>
                  <div>Expected Return</div>
                  <div>{(res.baseline.expected_return * 100).toFixed(2)}%</div>
                </div>
                <div style={styles.kv}>
                  <div>Volatility</div>
                  <div>{(res.baseline.volatility * 100).toFixed(2)}%</div>
                </div>
                <div style={styles.kv}>
                  <div>Sharpe</div>
                  <div>{Number(res.baseline.sharpe).toFixed(3)}</div>
                </div>
              </div>
            ) : (
              <div style={{ color: "#6b7280" }}>No stats yet.</div>
            )}
          </Section>

          <Section title="Market Snapshot">
            {res ? (
              <div style={{ display: "flex", gap: 24 }}>
                <div>Benchmark return: {(res.market.Rm * 100).toFixed(2)}%</div>
                <div>Risk-free rate: {(res.market.Rf * 100).toFixed(2)}%</div>
              </div>
            ) : (
              <div style={{ color: "#6b7280" }}>—</div>
            )}
          </Section>
        </div>
      </main>
    </div>
  );
}

// ---------- simple styles ----------
const styles = {
  page: {
    minHeight: "100vh",
    background: "#f6f7fb",
    color: "#111827",
    fontFamily:
      '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji"',
  },
  header: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: "24px 16px",
  },
  main: {
    display: "grid",
    gridTemplateColumns: "minmax(0, 360px) minmax(0, 1fr)",
    gap: 16,
    maxWidth: 1100,
    margin: "0 auto",
    padding: "0 16px 32px",
  },
  card: {
    minWidth: 0,
    background: "white",
    border: "1px solid #e5e7eb",
    borderRadius: 16,
    padding: 16,
    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
  },
  cardTitle: {
    fontSize: 13,
    color: "#6b7280",
    fontWeight: 600,
    marginBottom: 8,
  },
  label: { fontSize: 12, color: "#6b7280", marginBottom: 6 },
  formGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: 8,
  },
  tickerEntry: {
    display: "flex",
    gap: 8,
    minWidth: 0,
  },
  input: {
    width: "100%",
    minWidth: 0,
    boxSizing: "border-box",
    padding: "10px 12px",
    borderRadius: 12,
    border: "1px solid #d1d5db",
    background: "white",
  },
  primaryBtn: {
    boxSizing: "border-box",
    padding: "10px 12px",
    borderRadius: 12,
    border: "1px solid #111827",
    background: "#111827",
    color: "white",
    cursor: "pointer",
  },
  segmentedControl: {
    display: "flex",
    border: "1px solid #d1d5db",
    borderRadius: 8,
    overflow: "hidden",
  },
  segmentButton: {
    flex: 1,
    minWidth: 0,
    padding: "8px 10px",
    border: "none",
    background: "white",
    color: "#4b5563",
    cursor: "pointer",
  },
  segmentButtonActive: {
    background: "#111827",
    color: "white",
  },
  chip: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    borderRadius: 999,
    background: "#e5e7eb",
    fontSize: 13,
  },
  chipX: {
    background: "transparent",
    border: "none",
    cursor: "pointer",
    color: "#6b7280",
    fontSize: 16,
    lineHeight: 1,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 14,
  },
  kv: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
  },
  error: {
    color: "#b91c1c",
    background: "#fee2e2",
    border: "1px solid #fecaca",
    padding: "8px 10px",
    borderRadius: 10,
  },
};
