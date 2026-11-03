import React from "react";

const API =
  process.env.REACT_APP_API_URL || "http://localhost:8000/api/optimize";
const dashboardURL = API.replace(/\/api\/optimize\/?$/, "/backtest");

export default function Backtest() {
  return (
    <section aria-label="Portfolio backtesting">
      <p style={{ padding: "0 16px", color: "#617584", fontSize: 14 }}>
        Historical, out-of-sample comparisons with zero transaction costs. The backend may take a moment to wake up. {" "}
        <a href={dashboardURL} target="_blank" rel="noopener noreferrer">
          Open dashboard in a separate tab
        </a>
      </p>
      <iframe
        title="Portfolio backtest dashboard"
        src={dashboardURL}
        style={{ width: "100%", height: "calc(100vh - 125px)", minHeight: 650, border: 0 }}
      />
    </section>
  );
}