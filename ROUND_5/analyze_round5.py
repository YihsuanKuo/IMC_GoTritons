"""
analyze_round5.py

Round 5 product-group analysis for IMC-style price and trade CSV files.

It reads:
  1) prices CSV: order book snapshots, usually separated by semicolon (;)
  2) trades CSV: market trades, usually separated by semicolon (;)

It outputs:
  analysis_output/
    report.md
    summary_tables/
    group_charts/
    pair_analysis/
    rankings/

How to run:
  python analyze_round5.py --prices prices_round_5_day_2.csv --trades trades_round_5_day_2.csv

Optional:
  python analyze_round5.py --prices prices_round_5_day_2.csv --trades trades_round_5_day_2.csv --out analysis_output_day2
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PRODUCT_GROUPS: Dict[str, List[str]] = {
    "Galaxy Sounds Recorders": [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "Vertical Sleeping Pods": [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ],
    "Organic Microchips": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_TRIANGLE",
    ],
    "Purification Pebbles": [
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    ],
    "Domestic Robots": [
        "ROBOT_VACUUMING",
        "ROBOT_MOPPING",
        "ROBOT_DISHES",
        "ROBOT_LAUNDRY",
        "ROBOT_IRONING",
    ],
    "UV-Visors": [
        "UV_VISOR_YELLOW",
        "UV_VISOR_AMBER",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_MAGENTA",
    ],
    "Instant Translators": [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "Construction Panels": [
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "Liquid Breath Oxygen Shakes": [
        "OXYGEN_SHAKE_MORNING_BREATH",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_GARLIC",
    ],
    "Protein Snack Packs": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ],
}


GROUP_SHORT_NAMES = {
    "Galaxy Sounds Recorders": "galaxy_sounds",
    "Vertical Sleeping Pods": "sleep_pods",
    "Organic Microchips": "microchips",
    "Purification Pebbles": "pebbles",
    "Domestic Robots": "robots",
    "UV-Visors": "uv_visors",
    "Instant Translators": "translators",
    "Construction Panels": "panels",
    "Liquid Breath Oxygen Shakes": "oxygen_shakes",
    "Protein Snack Packs": "snackpacks",
}


ORDERED_HINTS: Dict[str, List[str]] = {
    "Purification Pebbles": ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "UV-Visors": ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_MAGENTA"],
    "Construction Panels": ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
}


PANEL_AREAS = {
    "PANEL_1X2": 2,
    "PANEL_2X2": 4,
    "PANEL_1X4": 4,
    "PANEL_2X4": 8,
    "PANEL_4X4": 16,
}


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    """Read CSV with automatic delimiter detection, favoring semicolon for IMC files."""
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path)


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "tables": out_dir / "summary_tables",
        "charts": out_dir / "group_charts",
        "pairs": out_dir / "pair_analysis",
        "rankings": out_dir / "rankings",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    required = {"timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices file is missing columns: {sorted(missing)}")

    # If mid_price is missing or partially null, repair it from best bid/ask.
    df["mid_price"] = df["mid_price"].fillna((df["bid_price_1"] + df["ask_price_1"]) / 2)
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    bid_volume_cols = [c for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if c in df.columns]
    ask_volume_cols = [c for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if c in df.columns]

    df["bid_volume_total"] = df[bid_volume_cols].abs().sum(axis=1) if bid_volume_cols else np.nan
    df["ask_volume_total"] = df[ask_volume_cols].abs().sum(axis=1) if ask_volume_cols else np.nan
    volume_sum = df["bid_volume_total"] + df["ask_volume_total"]
    df["order_book_imbalance"] = np.where(
        volume_sum != 0,
        (df["bid_volume_total"] - df["ask_volume_total"]) / volume_sum,
        np.nan,
    )

    df = df.sort_values(["product", "timestamp"])
    df["mid_return"] = df.groupby("product")["mid_price"].pct_change()
    df["mid_diff"] = df.groupby("product")["mid_price"].diff()
    df["future_mid_diff_1"] = df.groupby("product")["mid_price"].shift(-1) - df["mid_price"]
    df["rolling_mean_50"] = df.groupby("product")["mid_price"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    df["rolling_std_50"] = df.groupby("product")["mid_price"].transform(lambda s: s.rolling(50, min_periods=10).std())
    df["z_score_50"] = (df["mid_price"] - df["rolling_mean_50"]) / df["rolling_std_50"]

    return df


def add_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    if df.empty:
        return df
    required = {"timestamp", "symbol", "price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trades file is missing columns: {sorted(missing)}")
    df["notional"] = df["price"] * df["quantity"]
    return df


def product_summary(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    p = prices.groupby("product").agg(
        rows=("timestamp", "count"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        mean_mid=("mid_price", "mean"),
        median_mid=("mid_price", "median"),
        std_mid=("mid_price", "std"),
        min_mid=("mid_price", "min"),
        max_mid=("mid_price", "max"),
        mean_spread=("spread", "mean"),
        median_spread=("spread", "median"),
        max_spread=("spread", "max"),
        mean_bid_volume=("bid_volume_total", "mean"),
        mean_ask_volume=("ask_volume_total", "mean"),
        mean_imbalance=("order_book_imbalance", "mean"),
        std_imbalance=("order_book_imbalance", "std"),
        return_std=("mid_return", "std"),
        diff_std=("mid_diff", "std"),
        mean_pnl=("profit_and_loss", "mean") if "profit_and_loss" in prices.columns else ("mid_price", "mean"),
    ).reset_index()

    if not trades.empty:
        t = trades.groupby("symbol").agg(
            trade_count=("timestamp", "count"),
            total_trade_quantity=("quantity", "sum"),
            avg_trade_price=("price", "mean"),
            trade_notional=("notional", "sum"),
        ).reset_index().rename(columns={"symbol": "product"})
        t["vwap"] = t["trade_notional"] / t["total_trade_quantity"].replace(0, np.nan)
        p = p.merge(t, on="product", how="left")
    else:
        for col in ["trade_count", "total_trade_quantity", "avg_trade_price", "trade_notional", "vwap"]:
            p[col] = np.nan

    p["price_range"] = p["max_mid"] - p["min_mid"]
    p["relative_volatility"] = p["std_mid"] / p["mean_mid"].replace(0, np.nan)
    p["spread_to_volatility"] = p["mean_spread"] / p["diff_std"].replace(0, np.nan)
    return p


def pivot_mid(prices: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    sub = prices[prices["product"].isin(products)]
    wide = sub.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="last")
    return wide.sort_index()


def pivot_feature(prices: pd.DataFrame, products: List[str], feature: str) -> pd.DataFrame:
    sub = prices[prices["product"].isin(products)]
    wide = sub.pivot_table(index="timestamp", columns="product", values=feature, aggfunc="last")
    return wide.sort_index()


def plot_lines(wide: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(13, 7))
    for col in wide.columns:
        plt.plot(wide.index, wide[col], label=col, linewidth=1.3)
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_box(data: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    clean = [data[col].dropna().values for col in data.columns]
    plt.figure(figsize=(13, 7))
    plt.boxplot(clean, labels=data.columns, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path, vmin: float = -1, vmax: float = 1) -> None:
    plt.figure(figsize=(9, 7))
    arr = matrix.values.astype(float)
    im = plt.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                plt.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def estimate_halflife(series: pd.Series) -> float:
    """Estimate mean-reversion half-life from residual_t and delta residual_t."""
    s = series.dropna()
    if len(s) < 20:
        return np.nan
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    common = lag.index.intersection(delta.index)
    if len(common) < 20:
        return np.nan
    x = lag.loc[common].values
    y = delta.loc[common].values
    if np.std(x) == 0:
        return np.nan
    beta = np.polyfit(x, y, 1)[0]
    if beta >= 0:
        return np.nan
    return float(-math.log(2) / beta)


def pair_analysis_for_group(wide: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    returns = wide.pct_change()
    diffs = wide.diff()

    for a, b in itertools.combinations(wide.columns, 2):
        pair = wide[[a, b]].dropna()
        if len(pair) < 30:
            continue

        x = pair[b].values
        y = pair[a].values
        if np.std(x) == 0 or np.std(y) == 0:
            beta = np.nan
            alpha = np.nan
            residual = pd.Series(index=pair.index, dtype=float)
        else:
            beta, alpha = np.polyfit(x, y, 1)
            residual = pair[a] - (alpha + beta * pair[b])

        resid_std = residual.std()
        resid_range = residual.max() - residual.min()
        half_life = estimate_halflife(residual)
        corr_price = pair[a].corr(pair[b])
        corr_return = returns[a].corr(returns[b])

        ratio = pair[a] / pair[b].replace(0, np.nan)
        rows.append(
            {
                "group": group_name,
                "product_a": a,
                "product_b": b,
                "price_corr": corr_price,
                "return_corr": corr_return,
                "beta_a_on_b": beta,
                "alpha_a_on_b": alpha,
                "residual_mean": residual.mean(),
                "residual_std": resid_std,
                "residual_range": resid_range,
                "half_life_estimate": half_life,
                "ratio_mean": ratio.mean(),
                "ratio_std": ratio.std(),
                "observations": len(pair),
            }
        )
    return pd.DataFrame(rows)


def lead_lag_analysis_for_group(wide: pd.DataFrame, group_name: str, max_lag: int = 5) -> pd.DataFrame:
    """
    Checks whether return of A at time t correlates with return of B at time t+lag.
    Positive lag means A may lead B by that lag.
    """
    rets = wide.diff()
    rows = []
    for a, b in itertools.permutations(wide.columns, 2):
        best_lag = None
        best_corr = np.nan
        for lag in range(1, max_lag + 1):
            corr = rets[a].corr(rets[b].shift(-lag))
            if pd.notna(corr) and (pd.isna(best_corr) or abs(corr) > abs(best_corr)):
                best_corr = corr
                best_lag = lag
        rows.append(
            {
                "group": group_name,
                "leader_candidate": a,
                "follower_candidate": b,
                "best_lag": best_lag,
                "lead_lag_corr": best_corr,
                "abs_lead_lag_corr": abs(best_corr) if pd.notna(best_corr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def imbalance_predictiveness(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, sub in prices.groupby("product"):
        s = sub.sort_values("timestamp")
        corr = s["order_book_imbalance"].corr(s["future_mid_diff_1"])
        rows.append({"product": product, "imbalance_vs_next_mid_diff_corr": corr})
    return pd.DataFrame(rows)


def plot_pair_residuals(wide: pd.DataFrame, pair_df: pd.DataFrame, group_slug: str, out_dir: Path, top_n: int = 3) -> List[str]:
    saved = []
    if pair_df.empty:
        return saved
    ranked = pair_df.copy()
    # High price corr + lower residual std is usually more tradable.
    ranked["pair_score"] = ranked["price_corr"].abs().fillna(0) / ranked["residual_std"].replace(0, np.nan).abs()
    ranked = ranked.sort_values(["price_corr", "pair_score"], ascending=[False, False]).head(top_n)

    for _, row in ranked.iterrows():
        a, b = row["product_a"], row["product_b"]
        pair = wide[[a, b]].dropna()
        if len(pair) < 30:
            continue
        beta = row["beta_a_on_b"]
        alpha = row["alpha_a_on_b"]
        if pd.isna(beta) or pd.isna(alpha):
            continue
        residual = pair[a] - (alpha + beta * pair[b])
        z = (residual - residual.rolling(100, min_periods=20).mean()) / residual.rolling(100, min_periods=20).std()

        base = f"{group_slug}__{a}__vs__{b}"
        path1 = out_dir / f"{base}_residual.png"
        plt.figure(figsize=(13, 6))
        plt.plot(residual.index, residual, linewidth=1.2)
        plt.axhline(residual.mean(), linestyle="--", linewidth=1)
        plt.title(f"{a} - ({alpha:.2f} + {beta:.4f} * {b}) residual")
        plt.xlabel("timestamp")
        plt.ylabel("residual")
        plt.tight_layout()
        plt.savefig(path1, dpi=150)
        plt.close()
        saved.append(str(path1))

        path2 = out_dir / f"{base}_zscore.png"
        plt.figure(figsize=(13, 6))
        plt.plot(z.index, z, linewidth=1.2)
        plt.axhline(2, linestyle="--", linewidth=1)
        plt.axhline(-2, linestyle="--", linewidth=1)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(f"Rolling z-score of pair residual: {a} vs {b}")
        plt.xlabel("timestamp")
        plt.ylabel("z-score")
        plt.tight_layout()
        plt.savefig(path2, dpi=150)
        plt.close()
        saved.append(str(path2))
    return saved


def special_structure_analysis(prices: pd.DataFrame, dirs: Dict[str, Path]) -> pd.DataFrame:
    """Add extra tests for groups whose names imply ordering or mathematical structure."""
    rows = []

    # Construction Panels: compare price per area and same-area pairs.
    panel_products = [p for p in PANEL_AREAS if p in set(prices["product"])]
    if panel_products:
        panel_wide = pivot_mid(prices, panel_products)
        price_per_area = panel_wide.copy()
        for p in price_per_area.columns:
            price_per_area[p] = price_per_area[p] / PANEL_AREAS[p]
        plot_lines(
            price_per_area,
            "Construction Panels: mid price divided by panel area",
            "mid price / area",
            dirs["charts"] / "panels_price_per_area.png",
        )

        # Same area pair: PANEL_2X2 vs PANEL_1X4.
        if "PANEL_2X2" in panel_wide.columns and "PANEL_1X4" in panel_wide.columns:
            diff = panel_wide["PANEL_2X2"] - panel_wide["PANEL_1X4"]
            rows.append(
                {
                    "group": "Construction Panels",
                    "test": "same_area_PANEL_2X2_minus_PANEL_1X4",
                    "mean": diff.mean(),
                    "std": diff.std(),
                    "min": diff.min(),
                    "max": diff.max(),
                    "comment": "If stable around 0, these two may form a clean pair trade.",
                }
            )

    # Ordered groups: adjacent spreads.
    for group_name, order in ORDERED_HINTS.items():
        available = [p for p in order if p in set(prices["product"])]
        if len(available) < 2:
            continue
        wide = pivot_mid(prices, available)
        plt.figure(figsize=(13, 7))
        for a, b in zip(available[:-1], available[1:]):
            spread = wide[b] - wide[a]
            plt.plot(spread.index, spread, label=f"{b} - {a}", linewidth=1.2)
            rows.append(
                {
                    "group": group_name,
                    "test": f"adjacent_spread_{b}_minus_{a}",
                    "mean": spread.mean(),
                    "std": spread.std(),
                    "min": spread.min(),
                    "max": spread.max(),
                    "comment": "Check whether adjacent product spread is stable or mean-reverting.",
                }
            )
        plt.title(f"{group_name}: adjacent product spreads")
        plt.xlabel("timestamp")
        plt.ylabel("mid price difference")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(dirs["charts"] / f"{safe_name(group_name)}_adjacent_spreads.png", dpi=150)
        plt.close()

    return pd.DataFrame(rows)


def rank_groups(product_sum: pd.DataFrame, pair_all: pd.DataFrame, lead_lag_all: pd.DataFrame) -> pd.DataFrame:
    group_map = {p: g for g, products in PRODUCT_GROUPS.items() for p in products}
    ps = product_sum.copy()
    ps["group"] = ps["product"].map(group_map)

    base = ps.groupby("group").agg(
        avg_relative_volatility=("relative_volatility", "mean"),
        avg_mean_spread=("mean_spread", "mean"),
        avg_trade_count=("trade_count", "mean"),
        avg_total_trade_quantity=("total_trade_quantity", "mean"),
        avg_abs_imbalance=("mean_imbalance", lambda s: s.abs().mean()),
    ).reset_index()

    if not pair_all.empty:
        pair_stats = pair_all.groupby("group").agg(
            max_price_corr=("price_corr", lambda s: s.abs().max()),
            avg_abs_price_corr=("price_corr", lambda s: s.abs().mean()),
            best_residual_std=("residual_std", "min"),
            best_half_life=("half_life_estimate", "min"),
        ).reset_index()
        base = base.merge(pair_stats, on="group", how="left")

    if not lead_lag_all.empty:
        ll = lead_lag_all.groupby("group").agg(
            max_abs_lead_lag_corr=("abs_lead_lag_corr", "max"),
            avg_abs_lead_lag_corr=("abs_lead_lag_corr", "mean"),
        ).reset_index()
        base = base.merge(ll, on="group", how="left")

    # Simple ranking score: prioritize strong relationships, lead-lag, activity, and tradable movement.
    def zscore(col: pd.Series) -> pd.Series:
        if col.std(skipna=True) == 0 or col.dropna().empty:
            return pd.Series(0, index=col.index)
        return (col - col.mean(skipna=True)) / col.std(skipna=True)

    base["pattern_score"] = (
        1.5 * zscore(base.get("avg_abs_price_corr", pd.Series(0, index=base.index)).fillna(0))
        + 1.5 * zscore(base.get("max_abs_lead_lag_corr", pd.Series(0, index=base.index)).fillna(0))
        + 1.0 * zscore(base["avg_relative_volatility"].fillna(0))
        + 0.7 * zscore(base["avg_trade_count"].fillna(0))
        - 0.4 * zscore(base["avg_mean_spread"].fillna(0))
    )
    return base.sort_values("pattern_score", ascending=False)


def write_report(
    report_path: Path,
    product_sum: pd.DataFrame,
    group_rank: pd.DataFrame,
    pair_all: pd.DataFrame,
    lead_lag_all: pd.DataFrame,
    special_df: pd.DataFrame,
) -> None:
    top_groups = group_rank.head(10)
    top_pairs = pair_all.copy()
    if not top_pairs.empty:
        top_pairs["abs_price_corr"] = top_pairs["price_corr"].abs()
        top_pairs = top_pairs.sort_values(["abs_price_corr", "residual_std"], ascending=[False, True]).head(20)
    top_ll = lead_lag_all.sort_values("abs_lead_lag_corr", ascending=False).head(20) if not lead_lag_all.empty else pd.DataFrame()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Round 5 Day Analysis Report\n\n")
        f.write("## 你应该先怎么看\n\n")
        f.write("1. 先打开 `rankings/group_pattern_ranking.csv`，看哪一组最值得优先研究。\n")
        f.write("2. 再看 `group_charts/` 每组的 `raw_mid` 和 `normalized_mid`，判断是否有同步走势、偏离和回归。\n")
        f.write("3. 对排名高的组，打开 `pair_analysis/` 的 residual 和 z-score 图，看 pair trade 是否可行。\n")
        f.write("4. 如果 lead-lag 相关性很高，考虑做“领先商品动了，滞后商品还没动”的预测策略。\n")
        f.write("5. Construction Panels / Pebbles / UV-Visors 这几组有天然顺序或尺寸关系，额外看 adjacent spread 和 price-per-area 图。\n\n")

        f.write("## Group Ranking\n\n")
        f.write(top_groups.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Top Pair Candidates\n\n")
        if not top_pairs.empty:
            cols = [
                "group", "product_a", "product_b", "price_corr", "return_corr",
                "beta_a_on_b", "residual_std", "half_life_estimate", "observations"
            ]
            f.write(top_pairs[cols].to_markdown(index=False))
        else:
            f.write("No pair analysis available.")
        f.write("\n\n")

        f.write("## Top Lead-Lag Candidates\n\n")
        if not top_ll.empty:
            f.write(top_ll[["group", "leader_candidate", "follower_candidate", "best_lag", "lead_lag_corr"]].to_markdown(index=False))
        else:
            f.write("No lead-lag analysis available.")
        f.write("\n\n")

        f.write("## Special Structure Tests\n\n")
        if not special_df.empty:
            f.write(special_df.to_markdown(index=False))
        else:
            f.write("No special structure tests available.")
        f.write("\n\n")

        f.write("## Product Summary Preview\n\n")
        f.write(product_sum.head(20).to_markdown(index=False))
        f.write("\n")


def analyze(prices_path: str, trades_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    dirs = ensure_dirs(out)

    print("Reading files...")
    prices_raw = read_csv_auto(prices_path)
    trades_raw = read_csv_auto(trades_path) if trades_path else pd.DataFrame()

    print(f"prices shape: {prices_raw.shape}")
    print(f"trades shape: {trades_raw.shape}")

    prices = add_price_features(prices_raw)
    trades = add_trade_features(trades_raw) if not trades_raw.empty else pd.DataFrame()

    # Save enriched data sample, not the full huge file.
    prices.head(2000).to_csv(dirs["tables"] / "enriched_prices_sample.csv", index=False)

    print("Creating product summary...")
    prod_sum = product_summary(prices, trades)
    group_map = {p: g for g, products in PRODUCT_GROUPS.items() for p in products}
    prod_sum["group"] = prod_sum["product"].map(group_map)
    prod_sum = prod_sum.sort_values(["group", "product"])
    prod_sum.to_csv(dirs["tables"] / "product_summary.csv", index=False)

    print("Running imbalance predictiveness...")
    imb = imbalance_predictiveness(prices)
    imb.to_csv(dirs["tables"] / "imbalance_predictiveness.csv", index=False)

    pair_frames = []
    lead_lag_frames = []

    print("Creating group charts and pair analysis...")
    for group_name, products in PRODUCT_GROUPS.items():
        existing_products = [p for p in products if p in set(prices["product"])]
        if not existing_products:
            print(f"Skipping {group_name}: no products found")
            continue

        slug = GROUP_SHORT_NAMES.get(group_name, safe_name(group_name))
        wide = pivot_mid(prices, existing_products)
        if wide.empty:
            continue

        # 1. Raw mid prices.
        plot_lines(wide, f"{group_name}: raw mid price", "mid price", dirs["charts"] / f"{slug}_raw_mid.png")

        # 2. Normalized mid prices.
        norm = wide / wide.ffill().bfill().iloc[0]
        plot_lines(norm, f"{group_name}: normalized mid price", "mid / first mid", dirs["charts"] / f"{slug}_normalized_mid.png")

        # 3. Mid differences / returns correlation.
        corr = wide.diff().corr()
        corr.to_csv(dirs["tables"] / f"{slug}_diff_correlation.csv")
        plot_heatmap(corr, f"{group_name}: mid-price-diff correlation", dirs["charts"] / f"{slug}_diff_correlation_heatmap.png")

        # 4. Spread over time and distribution.
        spread = pivot_feature(prices, existing_products, "spread")
        plot_lines(spread, f"{group_name}: spread over time", "ask1 - bid1", dirs["charts"] / f"{slug}_spread_time.png")
        plot_box(spread, f"{group_name}: spread distribution", "spread", dirs["charts"] / f"{slug}_spread_boxplot.png")

        # 5. Order book imbalance.
        imbalance = pivot_feature(prices, existing_products, "order_book_imbalance")
        plot_lines(imbalance, f"{group_name}: order book imbalance", "imbalance", dirs["charts"] / f"{slug}_imbalance_time.png")

        # 6. Rolling volatility.
        rolling_vol = wide.diff().rolling(100, min_periods=20).std()
        plot_lines(rolling_vol, f"{group_name}: rolling volatility of mid-price changes", "rolling std(diff)", dirs["charts"] / f"{slug}_rolling_volatility.png")

        # 7. Pair analysis and lead-lag analysis.
        pair_df = pair_analysis_for_group(wide, group_name)
        if not pair_df.empty:
            pair_df.to_csv(dirs["pairs"] / f"{slug}_pair_summary.csv", index=False)
            pair_frames.append(pair_df)
            plot_pair_residuals(wide, pair_df, slug, dirs["pairs"], top_n=3)

        ll_df = lead_lag_analysis_for_group(wide, group_name, max_lag=5)
        if not ll_df.empty:
            ll_df.to_csv(dirs["pairs"] / f"{slug}_lead_lag_summary.csv", index=False)
            lead_lag_frames.append(ll_df)

    pair_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    lead_lag_all = pd.concat(lead_lag_frames, ignore_index=True) if lead_lag_frames else pd.DataFrame()

    if not pair_all.empty:
        pair_all.to_csv(dirs["rankings"] / "all_pair_summary.csv", index=False)
    if not lead_lag_all.empty:
        lead_lag_all.to_csv(dirs["rankings"] / "all_lead_lag_summary.csv", index=False)

    print("Running special structure analysis...")
    special_df = special_structure_analysis(prices, dirs)
    if not special_df.empty:
        special_df.to_csv(dirs["tables"] / "special_structure_tests.csv", index=False)

    print("Ranking groups...")
    group_rank = rank_groups(prod_sum, pair_all, lead_lag_all)
    group_rank.to_csv(dirs["rankings"] / "group_pattern_ranking.csv", index=False)

    print("Writing report...")
    write_report(out / "report.md", prod_sum, group_rank, pair_all, lead_lag_all, special_df)

    print("\nDone.")
    print(f"Output folder: {out.resolve()}")
    print(f"Main report: {(out / 'report.md').resolve()}")
    print("Start with these files:")
    print(f"  1) {dirs['rankings'] / 'group_pattern_ranking.csv'}")
    print(f"  2) {out / 'report.md'}")
    print(f"  3) {dirs['charts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 5 price/trade CSV files by product group.")
    parser.add_argument("--prices", required=True, help="Path to prices CSV file")
    parser.add_argument("--trades", required=True, help="Path to trades CSV file")
    parser.add_argument("--out", default="analysis_output", help="Output folder name")
    args = parser.parse_args()
    analyze(args.prices, args.trades, args.out)


if __name__ == "__main__":
    main()
