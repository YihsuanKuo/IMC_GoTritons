from __future__ import annotations

import argparse
import importlib.util
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from backtester import HistoricalData, ReplayBacktester
from backtester.reporting import save_outputs, summarize_result


def load_trader(strategy_path: str):
    path = Path(strategy_path)
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_strategy"] = module
    spec.loader.exec_module(module)
    return module.Trader()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay-style backtester for round-0 CSV data."
    )
    parser.add_argument(
        "--prices",
        required=True,
        help="Path to prices CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--trades",
        required=True,
        help="Path to trades CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Path to a strategy Python file exposing Trader.",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_output",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to run the backtest.",
    )

    args = parser.parse_args()

    trader = load_trader(args.strategy)
    data = HistoricalData(args.prices, args.trades, max_rows=args.max_steps)
    engine = ReplayBacktester(data, trader)
    result = engine.run()

    summary = summarize_result(result)
    print(summary.to_string(index=False))
    print("final_positions =", result.final_positions)

    save_outputs(result, args.output_dir)
    print(f"Outputs saved to: {args.output_dir}")

    df = pd.read_csv(f"{args.output_dir}/equity_curve.csv")

    if "timestamp" not in df.columns or "total_pnl" not in df.columns:
        raise ValueError(
            "equity_curve.csv must contain 'timestamp' and 'total_pnl' columns"
        )

    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["total_pnl"])
    plt.xlabel("Timestamp")
    plt.ylabel("Total PnL")
    plt.title("PnL Curve")
    plt.grid(True)
    plt.tight_layout()

    plot_path = Path(args.output_dir) / "pnl_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Plot midprice and position for each product
    products = sorted(
        {
            col.replace("mid_", "")
            for col in df.columns
            if col.startswith("mid_")
        }
    )

    for product in products:
        mid_col = f"mid_{product}"
        pos_col = f"pos_{product}"

        if mid_col not in df.columns or pos_col not in df.columns:
            continue

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(df["timestamp"], df[mid_col], label=f"{product} Midprice")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Midprice")
        ax1.set_title(f"{product}: Midprice and Position Over Time")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(
            df["timestamp"],
            df[pos_col],
            linestyle="--",
            label=f"{product} Position",
            color = "red"
        )
        ax2.set_ylabel("Position")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.tight_layout()
        fig.savefig(
            Path(args.output_dir) / f"{product.lower()}_midprice_position.png",
            dpi=150,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
