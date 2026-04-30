#!/usr/bin/env python3
"""
PEBBLES_XS / PEBBLES_L diagnosis tool

Goal:
    Help us find the real alpha before writing another Trader version.

What it does:
    1. Reads Round 5 price CSV files for day 2 / day 3 / day 4.
    2. Builds the XS/L residual spread:
            spread = PEBBLES_XS - (ALPHA + BETA * PEBBLES_L)
    3. Reconstructs baseline z-score pair-trading cycles.
    4. Compares good vs bad trades using only market-derived signals:
            z-score, drift, acceleration, volatility, entry/exit behavior.
    5. Runs a small grid search over legal signal filters:
            entry_z, exit_z, drift filter, adverse z stop, max hold bars.
       No timestamp cutoff. No PnL lock. No short-only assumption.
    6. Writes CSV summaries and charts into an output folder.

How to run:
    python3 pebbles_xs_l_diagnosis.py \
        --prices prices_round_5_day_2.csv prices_round_5_day_3.csv prices_round_5_day_4.csv

Optional:
    python3 pebbles_xs_l_diagnosis.py \
        --prices prices_round_5_day_2.csv prices_round_5_day_3.csv prices_round_5_day_4.csv \
        --out pebbles_xs_l_diagnosis_output \
        --alpha 13768.699562 \
        --beta -0.625515
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib is only used for saved charts. The script still writes CSVs if plotting fails.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


XS = "PEBBLES_XS"
L = "PEBBLES_L"

DEFAULT_ALPHA = 13768.699562
DEFAULT_BETA = -0.625515


# -----------------------------
# Reading and preprocessing
# -----------------------------

def read_csv_auto(path: str | Path) -> pd.DataFrame:
    """Read CSV with automatic separator detection.

    IMC files are often semicolon-separated, but local exports may use commas.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path)


def infer_day_from_filename(path: str | Path, fallback: int) -> int:
    name = Path(path).name
    for token in ["day_", "day"]:
        if token in name:
            tail = name.split(token, 1)[1]
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            if digits:
                return int(digits)
    return fallback


def normalize_price_frame(df: pd.DataFrame, day: int) -> pd.DataFrame:
    """Keep only XS/L rows and normalize column names/types."""
    needed = {"timestamp", "product"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found columns: {list(df.columns)}")

    d = df[df["product"].isin([XS, L])].copy()
    if d.empty:
        raise ValueError(f"No {XS}/{L} rows found.")

    d["day"] = day
    for c in [
        "bid_price_1", "bid_volume_1", "ask_price_1", "ask_volume_1", "mid_price", "profit_and_loss"
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "mid_price" not in d.columns or d["mid_price"].isna().all():
        d["mid_price"] = (d["bid_price_1"] + d["ask_price_1"]) / 2.0

    return d


def make_pair_frame(price_files: List[str], alpha: float, beta: float) -> pd.DataFrame:
    frames = []
    for i, path in enumerate(price_files):
        day = infer_day_from_filename(path, fallback=i)
        raw = read_csv_auto(path)
        d = normalize_price_frame(raw, day)

        # Pivot mid/bid/ask so each timestamp has XS and L side by side.
        base_cols = ["day", "timestamp", "product", "mid_price"]
        for c in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1", "profit_and_loss"]:
            if c in d.columns:
                base_cols.append(c)
        d = d[base_cols]

        piv = d.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price", aggfunc="last")
        piv.columns = [f"mid_{c}" for c in piv.columns]

        extra = []
        for c in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1", "profit_and_loss"]:
            if c in d.columns:
                p = d.pivot_table(index=["day", "timestamp"], columns="product", values=c, aggfunc="last")
                p.columns = [f"{c}_{prod}" for prod in p.columns]
                extra.append(p)

        day_df = pd.concat([piv] + extra, axis=1).reset_index().sort_values("timestamp")
        day_df = day_df.dropna(subset=[f"mid_{XS}", f"mid_{L}"]).copy()
        day_df["spread"] = day_df[f"mid_{XS}"] - (alpha + beta * day_df[f"mid_{L}"])
        day_df["raw_L_minus_XS"] = day_df[f"mid_{L}"] - day_df[f"mid_{XS}"]
        frames.append(day_df)

    if not frames:
        raise ValueError("No price files loaded.")
    return pd.concat(frames, ignore_index=True)


def add_features(df: pd.DataFrame, window: int = 300, short_window: int = 80) -> pd.DataFrame:
    """Add rolling features per day."""
    out = []
    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("timestamp").copy()
        s = g["spread"]
        g["roll_mean"] = s.rolling(window, min_periods=window).mean()
        g["roll_std"] = s.rolling(window, min_periods=window).std(ddof=0)
        g["short_mean"] = s.rolling(short_window, min_periods=short_window).mean()
        g["long_mean"] = g["roll_mean"]
        g["z"] = (s - g["roll_mean"]) / g["roll_std"]
        g["drift"] = g["short_mean"] - g["long_mean"]
        g["drift_z"] = g["drift"] / g["roll_std"]
        g["spread_change_20"] = s - s.shift(20)
        g["spread_change_80"] = s - s.shift(80)
        # Acceleration: recent 20-bar move compared with the prior 60 bars, normalized by std.
        g["accel"] = (s - s.shift(20)) - (s.shift(20) - s.shift(80))
        g["accel_z"] = g["accel"] / g["roll_std"]
        g["vol_short"] = s.diff().rolling(short_window, min_periods=short_window).std(ddof=0)
        g["vol_long"] = s.diff().rolling(window, min_periods=window).std(ddof=0)
        g["vol_ratio"] = g["vol_short"] / g["vol_long"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# -----------------------------
# Baseline cycle reconstruction
# -----------------------------

def reconstruct_cycles(
    df: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.4,
    max_forward: int = 200,
) -> pd.DataFrame:
    """Reconstruct baseline mode cycles from z-score.

    direction = -1 means short spread: sell XS / buy L.
    direction = +1 means long spread: buy XS / sell L.

    Profit here is spread-unit profit, not exact exchange PnL:
        short spread profit = entry_spread - exit_spread
        long spread profit  = exit_spread - entry_spread
    This is enough to diagnose alpha direction and bad regimes.
    """
    rows = []
    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        mode = 0
        entry_i = None
        entry = None

        for i, r in g.iterrows():
            z = r["z"]
            if not np.isfinite(z):
                continue

            if mode == 0:
                if z > entry_z:
                    mode = -1
                    entry_i = i
                    entry = r.copy()
                elif z < -entry_z:
                    mode = 1
                    entry_i = i
                    entry = r.copy()
            else:
                if abs(z) < exit_z:
                    exit_r = r.copy()
                    segment = g.iloc[entry_i : i + 1]
                    direction = mode
                    entry_spread = float(entry["spread"])
                    exit_spread = float(exit_r["spread"])
                    spread_profit = direction * (exit_spread - entry_spread)
                    # direction=+1 long spread, profit=exit-entry
                    # direction=-1 short spread, profit=entry-exit -> -1*(exit-entry)

                    # Future/within-trade diagnostics.
                    if direction == 1:
                        favorable = segment["spread"].max() - entry_spread
                        adverse = entry_spread - segment["spread"].min()
                    else:
                        favorable = entry_spread - segment["spread"].min()
                        adverse = segment["spread"].max() - entry_spread

                    rows.append({
                        "day": day,
                        "direction": direction,
                        "direction_name": "LONG_SPREAD_buy_XS_sell_L" if direction == 1 else "SHORT_SPREAD_sell_XS_buy_L",
                        "entry_timestamp": int(entry["timestamp"]),
                        "exit_timestamp": int(exit_r["timestamp"]),
                        "duration_bars": int(i - entry_i + 1),
                        "entry_spread": entry_spread,
                        "exit_spread": exit_spread,
                        "entry_z": float(entry["z"]),
                        "exit_z": float(exit_r["z"]),
                        "entry_drift_z": float(entry.get("drift_z", np.nan)),
                        "entry_accel_z": float(entry.get("accel_z", np.nan)),
                        "entry_vol_ratio": float(entry.get("vol_ratio", np.nan)),
                        "spread_profit_units": float(spread_profit),
                        "max_favorable_units": float(favorable),
                        "max_adverse_units": float(adverse),
                        "mfe_mae_ratio": float(favorable / adverse) if adverse and adverse > 1e-9 else np.nan,
                    })
                    mode = 0
                    entry_i = None
                    entry = None

        # If still open at end, close synthetically at last row for diagnostics.
        if mode != 0 and entry_i is not None and entry is not None:
            exit_r = g.iloc[-1]
            direction = mode
            entry_spread = float(entry["spread"])
            exit_spread = float(exit_r["spread"])
            spread_profit = direction * (exit_spread - entry_spread)
            rows.append({
                "day": day,
                "direction": direction,
                "direction_name": "LONG_SPREAD_buy_XS_sell_L" if direction == 1 else "SHORT_SPREAD_sell_XS_buy_L",
                "entry_timestamp": int(entry["timestamp"]),
                "exit_timestamp": int(exit_r["timestamp"]),
                "duration_bars": int(len(g) - entry_i),
                "entry_spread": entry_spread,
                "exit_spread": exit_spread,
                "entry_z": float(entry["z"]),
                "exit_z": float(exit_r["z"]),
                "entry_drift_z": float(entry.get("drift_z", np.nan)),
                "entry_accel_z": float(entry.get("accel_z", np.nan)),
                "entry_vol_ratio": float(entry.get("vol_ratio", np.nan)),
                "spread_profit_units": float(spread_profit),
                "max_favorable_units": np.nan,
                "max_adverse_units": np.nan,
                "mfe_mae_ratio": np.nan,
                "forced_end_close": True,
            })

    cycles = pd.DataFrame(rows)
    if not cycles.empty:
        cycles["is_good"] = cycles["spread_profit_units"] > 0
    return cycles


# -----------------------------
# Signal strategy grid search
# -----------------------------

def simulate_signal_strategy(
    df: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.4,
    drift_limit: Optional[float] = None,
    adverse_z_stop: Optional[float] = None,
    max_hold_bars: Optional[int] = None,
    require_turn: bool = False,
    turn_confirm: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simulate signal-only rules in spread units.

    This does NOT use timestamp cutoffs, PnL locks, or short-only assumptions.
    It is intentionally a diagnosis simulator, not exact exchange matching.
    """
    trades = []
    cumulative = 0.0
    equity_points = []

    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        mode = 0
        entry_i = None
        entry_r = None
        best_recent_high_z = -np.inf
        best_recent_low_z = np.inf

        for i, r in g.iterrows():
            z = r["z"]
            if not np.isfinite(z):
                continue

            # Track recent extremes using a rolling-ish reset each flat period.
            best_recent_high_z = max(best_recent_high_z, z)
            best_recent_low_z = min(best_recent_low_z, z)

            if mode == 0:
                dz = z - g.loc[i - 1, "z"] if i > 0 and np.isfinite(g.loc[i - 1, "z"]) else 0.0
                drift_z = r.get("drift_z", np.nan)
                allow_short = z > entry_z
                allow_long = z < -entry_z

                # Drift filter: avoid fading a spread that is drifting further away.
                if drift_limit is not None and np.isfinite(drift_z):
                    # If spread is high and short_mean is also far above long_mean, the upward regime is still active.
                    allow_short = allow_short and (drift_z < drift_limit)
                    # If spread is low and short_mean is also far below long_mean, the downward regime is still active.
                    allow_long = allow_long and (drift_z > -drift_limit)

                # Optional turn filter: only enter after the extreme starts moving back.
                if require_turn:
                    allow_short = allow_short and (best_recent_high_z - z >= turn_confirm) and (dz < 0)
                    allow_long = allow_long and (z - best_recent_low_z >= turn_confirm) and (dz > 0)

                if allow_short:
                    mode = -1
                    entry_i = i
                    entry_r = r.copy()
                    best_recent_high_z = -np.inf
                    best_recent_low_z = np.inf
                elif allow_long:
                    mode = 1
                    entry_i = i
                    entry_r = r.copy()
                    best_recent_high_z = -np.inf
                    best_recent_low_z = np.inf

            else:
                hold = i - entry_i
                exit_reason = None

                if abs(z) < exit_z:
                    exit_reason = "mean_revert_exit"
                elif max_hold_bars is not None and hold >= max_hold_bars:
                    exit_reason = "max_hold_exit"
                elif adverse_z_stop is not None:
                    entry_z_val = float(entry_r["z"])
                    # For short spread, bad means z goes even higher after entry.
                    if mode == -1 and z > entry_z_val + adverse_z_stop:
                        exit_reason = "adverse_z_exit"
                    # For long spread, bad means z goes even lower after entry.
                    elif mode == 1 and z < entry_z_val - adverse_z_stop:
                        exit_reason = "adverse_z_exit"

                if exit_reason:
                    direction = mode
                    entry_spread = float(entry_r["spread"])
                    exit_spread = float(r["spread"])
                    profit = direction * (exit_spread - entry_spread)
                    cumulative += profit
                    trades.append({
                        "day": int(day),
                        "direction": direction,
                        "direction_name": "LONG_SPREAD_buy_XS_sell_L" if direction == 1 else "SHORT_SPREAD_sell_XS_buy_L",
                        "entry_timestamp": int(entry_r["timestamp"]),
                        "exit_timestamp": int(r["timestamp"]),
                        "duration_bars": int(hold + 1),
                        "entry_z": float(entry_r["z"]),
                        "exit_z": float(r["z"]),
                        "entry_drift_z": float(entry_r.get("drift_z", np.nan)),
                        "entry_accel_z": float(entry_r.get("accel_z", np.nan)),
                        "profit_units": float(profit),
                        "exit_reason": exit_reason,
                        "cum_profit_units": cumulative,
                    })
                    equity_points.append(cumulative)
                    mode = 0
                    entry_i = None
                    entry_r = None
                    best_recent_high_z = -np.inf
                    best_recent_low_z = np.inf

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {
            "total_profit_units": 0.0,
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_profit_units": np.nan,
            "max_drawdown_units": 0.0,
        }
        return trades_df, metrics

    eq = trades_df["cum_profit_units"].to_numpy()
    peaks = np.maximum.accumulate(eq)
    drawdowns = peaks - eq
    metrics = {
        "total_profit_units": float(trades_df["profit_units"].sum()),
        "num_trades": int(len(trades_df)),
        "win_rate": float((trades_df["profit_units"] > 0).mean()),
        "avg_profit_units": float(trades_df["profit_units"].mean()),
        "max_drawdown_units": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "short_trade_profit": float(trades_df.loc[trades_df["direction"] == -1, "profit_units"].sum()),
        "long_trade_profit": float(trades_df.loc[trades_df["direction"] == 1, "profit_units"].sum()),
    }
    return trades_df, metrics


def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    entry_values = [1.8, 2.0, 2.2, 2.4, 2.6]
    exit_values = [0.3, 0.4, 0.5, 0.6]
    drift_values = [None, 0.4, 0.6, 0.8, 1.0]
    adverse_values = [None, 0.5, 0.8, 1.1]
    max_hold_values = [None, 80, 120, 180]
    turn_options = [False, True]

    for entry_z in entry_values:
        for exit_z in exit_values:
            if exit_z >= entry_z:
                continue
            for drift_limit in drift_values:
                for adverse in adverse_values:
                    for max_hold in max_hold_values:
                        for require_turn in turn_options:
                            turn_confirm = 0.10 if require_turn else 0.0
                            _, m = simulate_signal_strategy(
                                df,
                                entry_z=entry_z,
                                exit_z=exit_z,
                                drift_limit=drift_limit,
                                adverse_z_stop=adverse,
                                max_hold_bars=max_hold,
                                require_turn=require_turn,
                                turn_confirm=turn_confirm,
                            )
                            rows.append({
                                "entry_z": entry_z,
                                "exit_z": exit_z,
                                "drift_limit": "None" if drift_limit is None else drift_limit,
                                "adverse_z_stop": "None" if adverse is None else adverse,
                                "max_hold_bars": "None" if max_hold is None else max_hold,
                                "require_turn": require_turn,
                                **m,
                            })
    res = pd.DataFrame(rows)
    # A simple robustness score: profit minus drawdown penalty, and require some trades.
    res["robust_score"] = res["total_profit_units"] - 0.75 * res["max_drawdown_units"]
    res = res.sort_values(["robust_score", "total_profit_units"], ascending=False)
    return res


# -----------------------------
# Charts and summaries
# -----------------------------

def write_feature_summary(cycles: pd.DataFrame, out_dir: Path) -> None:
    if cycles.empty:
        return
    feature_cols = ["entry_z", "entry_drift_z", "entry_accel_z", "entry_vol_ratio", "duration_bars", "max_adverse_units", "max_favorable_units"]
    summary = cycles.groupby(["direction_name", "is_good"])[feature_cols].agg(["count", "mean", "median", "std", "min", "max"])
    summary.to_csv(out_dir / "entry_feature_summary_by_outcome.csv")


def save_day_charts(df: pd.DataFrame, cycles: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("timestamp")
        c = cycles[cycles["day"] == day] if not cycles.empty else pd.DataFrame()

        plt.figure(figsize=(14, 6))
        plt.plot(g["timestamp"], g["z"], linewidth=1.1)
        plt.axhline(2.0, linestyle="--", linewidth=0.8)
        plt.axhline(-2.0, linestyle="--", linewidth=0.8)
        plt.axhline(0.0, linewidth=0.8)
        if not c.empty:
            for _, r in c.iterrows():
                color = "green" if r["spread_profit_units"] > 0 else "red"
                marker = "v" if r["direction"] == -1 else "^"
                plt.scatter(r["entry_timestamp"], r["entry_z"], s=45, marker=marker, c=color)
        plt.title(f"Day {day}: PEBBLES_XS/L residual z-score with baseline entries")
        plt.xlabel("timestamp")
        plt.ylabel("z-score")
        plt.tight_layout()
        plt.savefig(chart_dir / f"day{day}_zscore_entries.png", dpi=160)
        plt.close()

        plt.figure(figsize=(14, 6))
        plt.plot(g["timestamp"], g["spread"], linewidth=1.1, label="spread")
        plt.plot(g["timestamp"], g["roll_mean"], linewidth=1.0, label="rolling mean")
        plt.title(f"Day {day}: residual spread and rolling mean")
        plt.xlabel("timestamp")
        plt.ylabel("spread")
        plt.legend()
        plt.tight_layout()
        plt.savefig(chart_dir / f"day{day}_spread_mean.png", dpi=160)
        plt.close()

        plt.figure(figsize=(14, 6))
        plt.plot(g["timestamp"], g["drift_z"], linewidth=1.1)
        plt.axhline(0.8, linestyle="--", linewidth=0.8)
        plt.axhline(-0.8, linestyle="--", linewidth=0.8)
        plt.axhline(0.0, linewidth=0.8)
        plt.title(f"Day {day}: drift_z = (short_mean - long_mean) / std")
        plt.xlabel("timestamp")
        plt.ylabel("drift_z")
        plt.tight_layout()
        plt.savefig(chart_dir / f"day{day}_drift_z.png", dpi=160)
        plt.close()


def write_markdown_report(out_dir: Path, cycles: pd.DataFrame, grid: pd.DataFrame) -> None:
    lines = []
    lines.append("# PEBBLES_XS / PEBBLES_L Diagnosis Report\n")
    lines.append("## What to inspect first\n")
    lines.append("1. `baseline_cycle_summary.csv`: good vs bad baseline cycles.\n")
    lines.append("2. `entry_feature_summary_by_outcome.csv`: market features that separate winners from losers.\n")
    lines.append("3. `filter_grid_results.csv`: legal signal filters ranked by robustness.\n")
    lines.append("4. `charts/day*_zscore_entries.png`: visual inspection of entries and z-score behavior.\n")
    lines.append("\n## Baseline cycle overview\n")
    if cycles.empty:
        lines.append("No cycles found under the baseline rules.\n")
    else:
        overview = cycles.groupby("direction_name").agg(
            trades=("spread_profit_units", "count"),
            total_profit_units=("spread_profit_units", "sum"),
            win_rate=("is_good", "mean"),
            avg_duration_bars=("duration_bars", "mean"),
            avg_entry_drift_z=("entry_drift_z", "mean"),
            avg_entry_accel_z=("entry_accel_z", "mean"),
        ).reset_index()
        lines.append(overview.to_string(index=False))
        lines.append("\n")

    lines.append("\n## Top signal-filter candidates\n")
    if not grid.empty:
        top = grid.head(15)
        lines.append(top.to_string(index=False))
        lines.append("\n")

    lines.append("\n## Important note\n")
    lines.append("This diagnosis intentionally avoids timestamp cutoffs, PnL locks, and short-only assumptions. ")
    lines.append("The purpose is to discover market-state rules that can generalize.\n")

    (out_dir / "diagnosis_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", nargs="+", required=True, help="Price CSV files, e.g. prices_round_5_day_2.csv ...")
    parser.add_argument("--out", default="pebbles_xs_l_diagnosis_output", help="Output directory")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--short-window", type=int, default=80)
    parser.add_argument("--baseline-entry-z", type=float, default=2.0)
    parser.add_argument("--baseline-exit-z", type=float, default=0.4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading price files...")
    pair = make_pair_frame(args.prices, alpha=args.alpha, beta=args.beta)
    print("Adding features...")
    feat = add_features(pair, window=args.window, short_window=args.short_window)
    feat.to_csv(out_dir / "pebbles_xs_l_features_all_days.csv", index=False)

    print("Reconstructing baseline cycles...")
    cycles = reconstruct_cycles(feat, entry_z=args.baseline_entry_z, exit_z=args.baseline_exit_z)
    cycles.to_csv(out_dir / "baseline_cycle_summary.csv", index=False)
    write_feature_summary(cycles, out_dir)

    print("Running legal signal-filter grid search...")
    grid = run_grid_search(feat)
    grid.to_csv(out_dir / "filter_grid_results.csv", index=False)

    if not grid.empty:
        best = grid.iloc[0]
        drift_limit = None if best["drift_limit"] == "None" else float(best["drift_limit"])
        adverse = None if best["adverse_z_stop"] == "None" else float(best["adverse_z_stop"])
        max_hold = None if best["max_hold_bars"] == "None" else int(best["max_hold_bars"])
        trades, metrics = simulate_signal_strategy(
            feat,
            entry_z=float(best["entry_z"]),
            exit_z=float(best["exit_z"]),
            drift_limit=drift_limit,
            adverse_z_stop=adverse,
            max_hold_bars=max_hold,
            require_turn=bool(best["require_turn"]),
            turn_confirm=0.10 if bool(best["require_turn"]) else 0.0,
        )
        trades.to_csv(out_dir / "best_filter_trades.csv", index=False)
        pd.DataFrame([metrics]).to_csv(out_dir / "best_filter_metrics.csv", index=False)

    print("Saving charts...")
    save_day_charts(feat, cycles, out_dir)
    write_markdown_report(out_dir, cycles, grid)

    print("Done.")
    print(f"Output folder: {out_dir.resolve()}")
    print("Start with:")
    print(f"  {out_dir / 'diagnosis_report.txt'}")
    print(f"  {out_dir / 'baseline_cycle_summary.csv'}")
    print(f"  {out_dir / 'filter_grid_results.csv'}")


if __name__ == "__main__":
    main()
