#!/usr/bin/env python3
"""
Aether Crystal manual challenge fair-value simulator.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path("/tmp/matplotlib-aether")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# User settings
# =========================

OUTPUT_DIR = Path("output")

N_SIMS = 100              # official style: 100 simulations
SEED = None               # None = new random result every run. Use an int like 42 to reproduce.
OFFICIAL_PATHS_PER_SETTLEMENT = 100

S0 = 50.0
SIGMA = 2.51              # 251% annualized vol
R = 0.0                   # zero risk-neutral drift
STEPS_PER_TRADING_DAY = 4
TRADING_DAYS_PER_YEAR = 252
CONTRACT_SIZE = 3000

T2_TRADING_DAYS = 10      # 2 weeks = 10 trading days
T3_TRADING_DAYS = 15      # 3 weeks = 15 trading days

BINARY_PAYOUT = 10.0      # assumed for AC_40_BP; edit if the product detail gives another payout


# =========================
# Market quotes from screenshot
# =========================

@dataclass(frozen=True)
class Quote:
    product: str
    bid: float
    ask: float
    max_volume: int


QUOTES: dict[str, Quote] = {
    "AC": Quote("AC", 49.975, 50.025, 200),

    "AC_50_P": Quote("AC_50_P", 12.00, 12.05, 50),
    "AC_50_C": Quote("AC_50_C", 12.00, 12.05, 50),
    "AC_35_P": Quote("AC_35_P", 4.33, 4.35, 50),
    "AC_40_P": Quote("AC_40_P", 6.50, 6.55, 50),
    "AC_45_P": Quote("AC_45_P", 9.05, 9.10, 50),
    "AC_60_C": Quote("AC_60_C", 8.80, 8.85, 50),

    "AC_50_P_2": Quote("AC_50_P_2", 9.70, 9.75, 50),
    "AC_50_C_2": Quote("AC_50_C_2", 9.70, 9.75, 50),

    "AC_50_CO": Quote("AC_50_CO", 22.20, 22.30, 50),
    "AC_40_BP": Quote("AC_40_BP", 5.00, 5.10, 50),
    "AC_45_KO": Quote("AC_45_KO", 0.150, 0.175, 500),
}


# =========================
# Simulation
# =========================

def reset_output_folder(path: Path) -> None:
    """Delete old output and recreate it."""
    if path.exists():
        shutil.rmtree(path)
    (path / "plots").mkdir(parents=True, exist_ok=True)


def simulate_gbm_paths(n_sims: int, seed: int | None = SEED) -> np.ndarray:
    """
    Return array of shape (n_sims, max_steps + 1).
    paths[:, 0] is S0.
    """
    rng = np.random.default_rng(seed)

    max_steps = T3_TRADING_DAYS * STEPS_PER_TRADING_DAY
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_TRADING_DAY)

    z = rng.standard_normal((n_sims, max_steps))
    log_returns = (R - 0.5 * SIGMA**2) * dt + SIGMA * math.sqrt(dt) * z

    log_paths = np.cumsum(log_returns, axis=1)
    paths = np.empty((n_sims, max_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def idx_after_trading_days(days: int) -> int:
    return days * STEPS_PER_TRADING_DAY


# =========================
# Payoff functions
# =========================

def call_payoff(s: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(s - k, 0.0)


def put_payoff(s: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(k - s, 0.0)


def build_payoffs(paths: np.ndarray) -> dict[str, np.ndarray]:
    i2 = idx_after_trading_days(T2_TRADING_DAYS)
    i3 = idx_after_trading_days(T3_TRADING_DAYS)

    s2 = paths[:, i2]
    s3 = paths[:, i3]

    payoffs: dict[str, np.ndarray] = {}

    # Underlying marked at final 3-week value.
    payoffs["AC"] = s3

    # 3-week vanilla options
    payoffs["AC_50_P"] = put_payoff(s3, 50)
    payoffs["AC_50_C"] = call_payoff(s3, 50)
    payoffs["AC_35_P"] = put_payoff(s3, 35)
    payoffs["AC_40_P"] = put_payoff(s3, 40)
    payoffs["AC_45_P"] = put_payoff(s3, 45)
    payoffs["AC_60_C"] = call_payoff(s3, 60)

    # 2-week vanilla options
    payoffs["AC_50_P_2"] = put_payoff(s2, 50)
    payoffs["AC_50_C_2"] = call_payoff(s2, 50)

    # Chooser: at 2 weeks choose call if S2 >= 50, else put.
    # Then the chosen option settles at the 3-week final price.
    choose_call = s2 >= 50
    payoffs["AC_50_CO"] = np.where(
        choose_call,
        call_payoff(s3, 50),
        put_payoff(s3, 50),
    )

    # Binary put: pays fixed amount if final S3 < 40
    payoffs["AC_40_BP"] = np.where(s3 < 40, BINARY_PAYOUT, 0.0)

    # Knock-out put: K=45, barrier=35, T=3 weeks.
    # If the path ever falls strictly below 35, payoff is zero.
    min_s_until_expiry = np.min(paths[:, : i3 + 1], axis=1)
    knocked_out = min_s_until_expiry < 35
    payoffs["AC_45_KO"] = np.where(knocked_out, 0.0, put_payoff(s3, 45))

    return payoffs


# =========================
# Analysis/output
# =========================

def summarize(payoffs: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for product, x in payoffs.items():
        q = QUOTES[product]
        fair = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

        buy_edge_per_unit = fair - q.ask
        sell_edge_per_unit = q.bid - fair

        best_side = "BUY" if buy_edge_per_unit > sell_edge_per_unit else "SELL"
        best_edge_per_unit = max(buy_edge_per_unit, sell_edge_per_unit)
        best_edge_full_size = best_edge_per_unit * q.max_volume * CONTRACT_SIZE

        rows.append({
            "product": product,
            "bid": q.bid,
            "ask": q.ask,
            "mid": (q.bid + q.ask) / 2,
            "max_volume": q.max_volume,
            "fair_value_per_unit": fair,
            "std_payoff_per_unit": std,
            "p05": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "buy_edge_per_unit_after_ask": buy_edge_per_unit,
            "sell_edge_per_unit_after_bid": sell_edge_per_unit,
            "best_side_by_fair_value": best_side,
            "best_edge_per_unit": best_edge_per_unit,
            "best_edge_if_max_volume_x_contract_size": best_edge_full_size,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("best_edge_if_max_volume_x_contract_size", ascending=False)


def save_distribution(payoffs: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(payoffs)
    df.index.name = "simulation"
    return df


def fair_values_from_payoffs(payoffs: dict[str, np.ndarray]) -> dict[str, float]:
    return {product: float(np.mean(values)) for product, values in payoffs.items()}


def run_official_fair_value_settlements(n_settlements: int) -> pd.DataFrame:
    """Return one row of simulated official fair values per 100-path settlement."""
    master_rng = np.random.default_rng(SEED)
    rows = []

    for settlement_id in range(1, n_settlements + 1):
        settlement_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        paths = simulate_gbm_paths(OFFICIAL_PATHS_PER_SETTLEMENT, seed=settlement_seed)
        fair_values = fair_values_from_payoffs(build_payoffs(paths))
        row = {"settlement_id": settlement_id, "seed": settlement_seed}
        row.update(fair_values)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_histograms(payoffs: dict[str, np.ndarray], out_dir: Path) -> None:
    for product, x in payoffs.items():
        plt.figure(figsize=(8, 5))
        plt.hist(x, bins=min(20, max(5, len(x) // 5)), edgecolor="black")
        plt.axvline(np.mean(x), linestyle="--", linewidth=2, label=f"fair value = {np.mean(x):.4f}")
        plt.title(f"{product} payoff distribution, {len(x)} simulations")
        plt.xlabel("Payoff per unit")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / f"{product}_hist.png", dpi=160)
        plt.close()


def plot_paths(paths: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    max_to_plot = min(100, paths.shape[0])
    for i in range(max_to_plot):
        plt.plot(paths[i], alpha=0.35, linewidth=1)
    plt.axhline(50, linestyle="--", linewidth=1, label="S0 / K=50")
    plt.axhline(35, linestyle=":", linewidth=1, label="KO barrier=35")
    plt.title(f"Aether Crystal simulated paths, {paths.shape[0]} simulations")
    plt.xlabel("Simulation step")
    plt.ylabel("AC price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "AC_paths.png", dpi=160)
    plt.close()


def main() -> None:
    reset_output_folder(OUTPUT_DIR)

    paths = simulate_gbm_paths(N_SIMS)
    payoffs = build_payoffs(paths)

    fair_df = summarize(payoffs)
    dist_df = save_distribution(payoffs)
    official_fair_values_df = run_official_fair_value_settlements(N_SIMS)

    fair_df.to_csv(OUTPUT_DIR / "fair_values.csv", index=False)
    dist_df.to_csv(OUTPUT_DIR / "payoff_distribution.csv", index=True)
    official_fair_values_df.to_csv(OUTPUT_DIR / "official_fair_value_settlements.csv", index=False)

    pnl_cols = [
        "product",
        "bid",
        "ask",
        "max_volume",
        "fair_value_per_unit",
        "buy_edge_per_unit_after_ask",
        "sell_edge_per_unit_after_bid",
        "best_side_by_fair_value",
        "best_edge_per_unit",
        "best_edge_if_max_volume_x_contract_size",
    ]
    fair_df[pnl_cols].to_csv(OUTPUT_DIR / "pnl_edges.csv", index=False)

    plot_paths(paths, OUTPUT_DIR)
    plot_histograms(payoffs, OUTPUT_DIR)

    print("\nSaved outputs to:", OUTPUT_DIR.resolve())
    print("\nPredicted fair prices:")
    print(fair_df[pnl_cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\nReminder:")
    print("- BUY edge = fair value - ask")
    print("- SELL edge = bid - fair value")
    print("- Full-size edge multiplies by max_volume and contract size =", CONTRACT_SIZE)
    print("- This uses only", N_SIMS, "simulations, so results can be noisy. Increase N_SIMS for smoother estimates.")


if __name__ == "__main__":
    main()
