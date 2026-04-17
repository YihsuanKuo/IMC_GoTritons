from pathlib import Path

import matplotlib.pyplot as plt

from backtester import Backtester, load_strategy, make_summary


def add_global_time(df):
    if df.empty:
        return df
    df = df.sort_values(["day", "timestamp"]).copy()
    min_day = int(df["day"].min())
    day_span = int(df["timestamp"].max()) + 1
    df["global_time"] = (df["day"] - min_day) * day_span + df["timestamp"]
    return df


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    project_root = here.parent

    strategy_path = project_root / "strategy.py"
    if not strategy_path.exists():
        strategy_path = project_root / "pepper_root.py"
    data_dir = project_root / "data"

    strategy = load_strategy(strategy_path)
    backtester = Backtester(strategy=strategy, reset_each_day=False)

    pnl_df, fills_df = backtester.run_many_days(
        data_dir=data_dir,
        round_number=1,
        days=[-2, -1, 0],
    )
    pnl_df = add_global_time(pnl_df)
    summary_df = make_summary(pnl_df, fills_df)

    out_dir = here / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    pnl_df.to_csv(out_dir / "pnl_timeseries.csv", index=False)
    fills_df.to_csv(out_dir / "fills.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print(summary_df.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df["global_time"], pnl_df["total_pnl"])
    plt.xlabel("Global Time")
    plt.ylabel("PnL")
    plt.title("PnL Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
