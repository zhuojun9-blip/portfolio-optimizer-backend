import React from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import Optimizer from "./Optimizer";
import Definitions from "./Definitions";
import Backtest from "./Backtest";
export default function App() {
  return <Router><header className="app-header"><div className="app-nav-inner"><span className="wordmark">Portfolio Optimizer</span><nav aria-label="Main navigation"><NavLink to="/" end>Optimizer</NavLink><NavLink to="/backtest">Backtest</NavLink><NavLink to="/definitions">Definitions</NavLink></nav><span className="nav-context">Allocation & research</span></div></header><Routes><Route path="/" element={<Optimizer/>}/><Route path="/definitions" element={<Definitions/>}/><Route path="/backtest" element={<Backtest/>}/></Routes></Router>;
}
