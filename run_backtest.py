"""Download once and replay: python run_backtest.py config.json --output results/run1.
For an offline replay add --data results/run1/data.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from backtest import BacktestRequest, run_backtest
from backtest_api import download_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    req = BacktestRequest.model_validate_json(args.config.read_text())
    if args.data:
        prices = pd.read_csv(args.data / "prices.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        market = pd.read_csv(args.data / "market.csv", index_col=0, parse_dates=True, float_precision="round_trip").iloc[:, 0]
        rf = pd.read_csv(args.data / "risk_free.csv", index_col=0, parse_dates=True, float_precision="round_trip").iloc[:, 0]
    else:
        prices, market, rf = download_data(req)
    result = run_backtest(req, prices, market, rf)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "data"
    snapshot.mkdir(exist_ok=True)
    prices.to_csv(snapshot / "prices.csv")
    market.to_csv(snapshot / "market.csv")
    rf.to_csv(snapshot / "risk_free.csv")
    (args.output / "config.json").write_text(req.model_dump_json(indent=2))
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    summaries, navs, weights = {}, {}, []
    for name, strategy in result["strategies"].items():
        if strategy["status"] == "ok":
            summaries[name] = strategy["metrics"]
            navs[name] = strategy["nav"]
            for day, row in zip(result["dates"], strategy["weights"]):
                weights.append({"date": day, "strategy": name, **dict(zip(req.tickers, row))})
        else:
            summaries[name] = {"status": "failed", "error": strategy["error"]}
    pd.DataFrame.from_dict(summaries, orient="index").to_csv(args.output / "summary.csv")
    pd.DataFrame(navs, index=result["dates"]).to_csv(args.output / "equity.csv", index_label="date")
    pd.DataFrame(weights).to_csv(args.output / "weights.csv", index=False)
    print(f"Saved results and replayable data to {args.output}")


if __name__ == "__main__":
    main()
