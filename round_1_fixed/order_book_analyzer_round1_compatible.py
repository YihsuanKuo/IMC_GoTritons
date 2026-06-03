#!/usr/bin/env python3
"""
Order book analyzer for IMC/Prosperity-style CSV files.
Compatible with sparse books where one side may be missing at a snapshot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRICE_LEVELS = (1, 2, 3)
PEPPER_ROOT_PRODUCT = "INTARIAN_PEPPER_ROOT"


def infer_day_from_path(path: Path) -> Optional[int]:
    m = re.search(r"day_(-?\d+)", path.name)
    return int(m.group(1)) if m else None


def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def _expand_globs(raw_paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_paths:
        raw = Path(raw)
        matches = sorted(raw.parent.glob(raw.name)) if any(ch in raw.name for ch in "*?[]") else [raw]
        out.extend(matches)
    return out


def load_prices(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = safe_read_csv(path)
        if "day" not in df.columns:
            df["day"] = infer_day_from_path(path)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise ValueError("No prices files were provided.")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return out


def load_trades(paths: Iterable[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = safe_read_csv(path)
        if "day" not in df.columns:
            df["day"] = infer_day_from_path(path)
        if "product" not in df.columns and "symbol" in df.columns:
            df["product"] = df["symbol"]
        df["source_file"] = path.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if not out.empty:
        out = out.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return out


def _first_non_null(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.where(a.notna(), b)


def add_book_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    best_bid = df.get("bid_price_1", pd.Series(np.nan, index=df.index))
    best_ask = df.get("ask_price_1", pd.Series(np.nan, index=df.index))

    df["best_bid_observed"] = best_bid
    df["best_ask_observed"] = best_ask
    df["book_state"] = np.select(
        [best_bid.notna() & best_ask.notna(), best_bid.notna() & best_ask.isna(), best_bid.isna() & best_ask.notna()],
        ["two_sided", "bid_only", "ask_only"],
        default="empty",
    )

    # Spread only makes sense when both sides exist.
    df["spread"] = np.where(best_bid.notna() & best_ask.notna(), best_ask - best_bid, np.nan)

    # Keep provided mid_price when available. Otherwise fallback sensibly for one-sided books.
    if "mid_price" not in df.columns:
        df["mid_price"] = np.nan
    fallback_mid = np.where(best_bid.notna() & best_ask.notna(), (best_bid + best_ask) / 2.0, np.where(best_bid.notna(), best_bid, best_ask))
    df["mid_price"] = pd.Series(df["mid_price"]).where(pd.Series(df["mid_price"]).notna(), fallback_mid)

    bid_v1 = df.get("bid_volume_1", pd.Series(0, index=df.index)).fillna(0)
    ask_v1 = df.get("ask_volume_1", pd.Series(0, index=df.index)).fillna(0)
    top_denom = bid_v1 + ask_v1
    df["imbalance_l1"] = np.where(top_denom > 0, (bid_v1 - ask_v1) / top_denom, np.nan)

    bid_depth = pd.Series(0.0, index=df.index)
    ask_depth = pd.Series(0.0, index=df.index)
    for level in PRICE_LEVELS:
        weight = 1.0 / level
        bid_depth += weight * df.get(f"bid_volume_{level}", pd.Series(0, index=df.index)).fillna(0)
        ask_depth += weight * df.get(f"ask_volume_{level}", pd.Series(0, index=df.index)).fillna(0)
    df["bid_depth_weighted"] = bid_depth
    df["ask_depth_weighted"] = ask_depth
    total_depth = bid_depth + ask_depth
    df["depth_imbalance"] = np.where(total_depth > 0, (bid_depth - ask_depth) / total_depth, np.nan)

    # Microprice: standard when two-sided, otherwise fall back to visible side / mid.
    standard_micro = np.where(
        top_denom > 0,
        (best_ask * bid_v1 + best_bid * ask_v1) / top_denom,
        np.nan,
    )
    fallback_micro = np.where(best_bid.notna() & best_ask.isna(), best_bid, np.where(best_ask.notna() & best_bid.isna(), best_ask, df["mid_price"]))
    df["microprice"] = np.where(best_bid.notna() & best_ask.notna(), standard_micro, fallback_micro)

    group_keys = ["product", "day"]
    df["mid_return_1"] = df.groupby(group_keys)["mid_price"].pct_change()
    df["mid_change_1"] = df.groupby(group_keys)["mid_price"].diff()
    df["mid_change_5"] = df.groupby(group_keys)["mid_price"].diff(5)
    df["micro_minus_mid"] = df["microprice"] - df["mid_price"]
    df["log_mid"] = np.log(df["mid_price"].replace(0, np.nan))
    df["log_ret_1"] = df.groupby(group_keys)["log_mid"].diff()
    return df


def add_pepper_root_linear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["linear_fair_value"] = np.nan
    df["linear_spread"] = np.nan
    df["linear_spread_zscore"] = np.nan

    pepper_mask = df["product"] == PEPPER_ROOT_PRODUCT
    if not pepper_mask.any():
        return df

    valid_mask = pepper_mask & df["mid_price"].notna() & (df["mid_price"] > 0)
    if valid_mask.sum() < 3:
        return df

    valid = df.loc[valid_mask, ["day", "timestamp", "mid_price"]].copy()
    design = np.column_stack(
        [
            np.ones(len(valid)),
            valid["day"].to_numpy(dtype=float),
            valid["timestamp"].to_numpy(dtype=float),
        ]
    )
    response = valid["mid_price"].to_numpy(dtype=float)
    intercept, day_coef, timestamp_coef = np.linalg.lstsq(design, response, rcond=None)[0]

    fair_value = intercept + day_coef * df["day"].to_numpy(dtype=float) + timestamp_coef * df["timestamp"].to_numpy(dtype=float)
    df.loc[pepper_mask, "linear_fair_value"] = fair_value[pepper_mask]
    df.loc[valid_mask, "linear_spread"] = df.loc[valid_mask, "mid_price"] - df.loc[valid_mask, "linear_fair_value"]

    spread_std = df.loc[valid_mask, "linear_spread"].std()
    if pd.notna(spread_std) and spread_std > 0:
        df.loc[valid_mask, "linear_spread_zscore"] = df.loc[valid_mask, "linear_spread"] / spread_std

    return df


def summarize_products(book: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in book.groupby("product"):
        row = {
            "product": product,
            "rows": int(len(g)),
            "days": int(g["day"].nunique()) if "day" in g.columns else np.nan,
            "two_sided_rows": int((g["book_state"] == "two_sided").sum()),
            "bid_only_rows": int((g["book_state"] == "bid_only").sum()),
            "ask_only_rows": int((g["book_state"] == "ask_only").sum()),
            "avg_spread": g["spread"].mean(),
            "spread_std": g["spread"].std(),
            "avg_mid_price": g["mid_price"].mean(),
            "mid_price_std": g["mid_price"].std(),
            "avg_abs_mid_change_1": g["mid_change_1"].abs().mean(),
            "avg_abs_mid_change_5": g["mid_change_5"].abs().mean(),
            "avg_imbalance_l1": g["imbalance_l1"].mean(),
            "avg_depth_imbalance": g["depth_imbalance"].mean(),
            "realized_vol_logret": g["log_ret_1"].std(),
        }
        if product == PEPPER_ROOT_PRODUCT and "linear_spread" in g.columns:
            valid = g[g["linear_spread"].notna()]
            if not valid.empty:
                design = np.column_stack(
                    [
                        np.ones(len(valid)),
                        valid["day"].to_numpy(dtype=float),
                        valid["timestamp"].to_numpy(dtype=float),
                    ]
                )
                response = valid["mid_price"].to_numpy(dtype=float)
                intercept, day_coef, timestamp_coef = np.linalg.lstsq(design, response, rcond=None)[0]
                residuals = response - design @ np.array([intercept, day_coef, timestamp_coef])
                total_ss = np.square(response - response.mean()).sum()
                residual_ss = np.square(residuals).sum()
                row.update(
                    {
                        "linear_pattern_intercept": intercept,
                        "linear_pattern_day_coef": day_coef,
                        "linear_pattern_timestamp_coef": timestamp_coef,
                        "linear_spread_mean": valid["linear_spread"].mean(),
                        "linear_spread_std": valid["linear_spread"].std(),
                        "linear_spread_abs_mean": valid["linear_spread"].abs().mean(),
                        "linear_spread_max_abs": valid["linear_spread"].abs().max(),
                        "linear_pattern_r2": np.nan if total_ss == 0 else 1.0 - (residual_ss / total_ss),
                        "linear_pattern_observations": int(len(valid)),
                    }
                )
        if not trades.empty:
            tg = trades[trades["product"] == product].copy()
            row["trade_count"] = int(len(tg)) if not tg.empty else 0
            row["total_trade_volume"] = float(tg["quantity"].sum()) if not tg.empty else 0.0
            row["vwap"] = float(np.average(tg["price"], weights=tg["quantity"])) if not tg.empty else np.nan
            row["avg_trade_size"] = float(tg["quantity"].mean()) if not tg.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def build_trade_join(book: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    keep_cols = [
        "product", "day", "timestamp", "bid_price_1", "ask_price_1", "mid_price", "spread",
        "imbalance_l1", "microprice", "depth_imbalance", "book_state", "linear_fair_value", "linear_spread",
    ]
    merged_parts = []
    for (product, day), tg in trades.groupby(["product", "day"], sort=False):
        bg = book[(book["product"] == product) & (book["day"] == day)][keep_cols].sort_values("timestamp").copy()
        if bg.empty:
            continue
        tg = tg.sort_values("timestamp").copy()
        mg = pd.merge_asof(
            tg, bg,
            on="timestamp",
            by=["product", "day"],
            direction="backward",
        )
        merged_parts.append(mg)
    if not merged_parts:
        return pd.DataFrame()
    merged = pd.concat(merged_parts, ignore_index=True).sort_values(["product", "day", "timestamp"])
    merged["trade_minus_mid"] = merged["price"] - merged["mid_price"]
    merged["trade_minus_micro"] = merged["price"] - merged["microprice"]
    if "linear_fair_value" in merged.columns:
        merged["trade_minus_linear_fair"] = merged["price"] - merged["linear_fair_value"]
    return merged


def plot_product(book: pd.DataFrame, trades: pd.DataFrame, product: str, outdir: Path) -> None:
    g = book[book["product"] == product].sort_values(["day", "timestamp"]).copy()
    if g.empty:
        return
    x = np.arange(len(g))

    plt.figure(figsize=(12, 5))
    plt.plot(x, g["mid_price"], label="mid_price")
    plt.plot(x, g["microprice"], label="microprice", alpha=0.8)
    if "linear_fair_value" in g.columns and g["linear_fair_value"].notna().any():
        plt.plot(x, g["linear_fair_value"], label="linear_fair_value", alpha=0.9)
    plt.title(f"{product}: Mid-price vs Microprice")
    plt.xlabel("snapshot index")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{product.lower()}_mid_microprice.png", dpi=150)
    plt.close()

    if g["spread"].notna().any():
        plt.figure(figsize=(12, 4))
        plt.plot(x, g["spread"])
        plt.title(f"{product}: Spread over time (two-sided only)")
        plt.xlabel("snapshot index")
        plt.ylabel("spread")
        plt.tight_layout()
        plt.savefig(outdir / f"{product.lower()}_spread.png", dpi=150)
        plt.close()

    if "linear_spread" in g.columns and g["linear_spread"].notna().any():
        plt.figure(figsize=(12, 4))
        plt.plot(x, g["linear_spread"], label="linear spread")
        plt.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        plt.title(f"{product}: Linear-pattern spread")
        plt.xlabel("snapshot index")
        plt.ylabel("mid - fitted linear fair value")
        plt.tight_layout()
        plt.savefig(outdir / f"{product.lower()}_linear_spread.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(x, g["imbalance_l1"], label="imbalance_l1")
    plt.plot(x, g["depth_imbalance"], label="depth_imbalance", alpha=0.8)
    plt.title(f"{product}: Book imbalance")
    plt.xlabel("snapshot index")
    plt.ylabel("imbalance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{product.lower()}_imbalance.png", dpi=150)
    plt.close()

    tg = trades[trades["product"] == product].copy() if not trades.empty else pd.DataFrame()
    if not tg.empty:
        plt.figure(figsize=(10, 4))
        plt.hist(tg["price"], bins=30)
        plt.title(f"{product}: Trade price distribution")
        plt.xlabel("trade price")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / f"{product.lower()}_trade_price_hist.png", dpi=150)
        plt.close()


def write_text_report(summary: pd.DataFrame, outpath: Path) -> None:
    lines = ["ORDER BOOK ANALYSIS REPORT", "=" * 80, ""]
    for _, row in summary.iterrows():
        lines += [
            f"Product: {row['product']}",
            f"  rows: {int(row['rows'])}",
            f"  days: {int(row['days'])}",
            f"  two-sided rows: {int(row['two_sided_rows'])}",
            f"  bid-only rows: {int(row['bid_only_rows'])}",
            f"  ask-only rows: {int(row['ask_only_rows'])}",
            f"  avg spread: {row['avg_spread']:.4f}" if pd.notna(row['avg_spread']) else "  avg spread: n/a",
            f"  spread std: {row['spread_std']:.4f}" if pd.notna(row['spread_std']) else "  spread std: n/a",
            f"  avg mid price: {row['avg_mid_price']:.4f}",
            f"  mid price std: {row['mid_price_std']:.4f}",
            f"  avg |mid change(1)|: {row['avg_abs_mid_change_1']:.4f}",
            f"  avg |mid change(5)|: {row['avg_abs_mid_change_5']:.4f}",
            f"  avg L1 imbalance: {row['avg_imbalance_l1']:.4f}",
            f"  avg depth imbalance: {row['avg_depth_imbalance']:.4f}",
            f"  realized vol (log ret std): {row['realized_vol_logret']:.6f}",
        ]
        if row["product"] == PEPPER_ROOT_PRODUCT and pd.notna(row.get("linear_pattern_intercept", np.nan)):
            lines += [
                "  detected linear pattern:",
                f"    fair value ~= {row['linear_pattern_intercept']:.4f} + ({row['linear_pattern_day_coef']:.4f} * day) + ({row['linear_pattern_timestamp_coef']:.9f} * timestamp)",
                f"  linear spread mean: {row['linear_spread_mean']:.4f}",
                f"  linear spread std: {row['linear_spread_std']:.4f}",
                f"  linear spread mean abs: {row['linear_spread_abs_mean']:.4f}",
                f"  linear spread max abs: {row['linear_spread_max_abs']:.4f}",
                f"  linear pattern R^2: {row['linear_pattern_r2']:.6f}",
                f"  fitted observations: {int(row['linear_pattern_observations'])}",
            ]
        if "trade_count" in row:
            lines += [
                f"  trade count: {int(row['trade_count'])}",
                f"  total trade volume: {row['total_trade_volume']:.0f}",
                f"  average trade size: {row['avg_trade_size']:.4f}" if pd.notna(row['avg_trade_size']) else "  average trade size: n/a",
                f"  vwap: {row['vwap']:.4f}" if pd.notna(row['vwap']) else "  vwap: n/a",
            ]
        lines.append("")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Prosperity-style order book files.")
    parser.add_argument("--prices", nargs="+", required=True, help="One or more prices CSV files. Globs supported.")
    parser.add_argument("--trades", nargs="*", default=[], help="Optional trades CSV files. Globs supported.")
    parser.add_argument("--outdir", default="analysis_output", help="Directory for outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_paths = _expand_globs(args.prices)
    trade_paths = _expand_globs(args.trades)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    book = add_pepper_root_linear_features(add_book_features(load_prices(price_paths)))
    trades = load_trades(trade_paths)
    summary = summarize_products(book, trades)
    trade_join = build_trade_join(book, trades)

    book.to_csv(outdir / "book_signals.csv", index=False)
    summary.to_csv(outdir / "product_summary.csv", index=False)
    if not trade_join.empty:
        trade_join.to_csv(outdir / "trades_with_book_context.csv", index=False)
    write_text_report(summary, outdir / "report.txt")

    for product in sorted(book["product"].dropna().unique()):
        plot_product(book, trades, product, outdir)

    print(f"Saved analysis to: {outdir.resolve()}")
    for path in sorted(outdir.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
