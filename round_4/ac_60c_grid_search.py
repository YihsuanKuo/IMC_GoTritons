#!/usr/bin/env python3
"""
Grid search AC and AC_60_C hedges around the fixed Aether Crystal portfolio.

This script intentionally imports the existing GBM and payoff logic from
aether_fair_value_simulator.py instead of redefining product payoffs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aether_fair_value_simulator import (
    CONTRACT_SIZE,
    OFFICIAL_PATHS_PER_SETTLEMENT,
    OUTPUT_DIR,
    build_payoffs,
    fair_values_from_payoffs,
    simulate_gbm_paths,
)


N_SETTLEMENTS = 1000
Q_AC_RANGE = range(-200, 201)
Q_60C_RANGE = range(-50, 51)


@dataclass(frozen=True)
class Position:
    product: str
    side: str
    quantity: int
    entry_price: float


def position_pnl(position: Position, fair_values: dict[str, float]) -> float:
    fair_value = fair_values[position.product]
    if position.side == "BUY":
        edge = fair_value - position.entry_price
    elif position.side == "SELL":
        edge = position.entry_price - fair_value
    else:
        raise ValueError(f"Unknown position side: {position.side}")
    return edge * position.quantity * CONTRACT_SIZE


def portfolio_pnl(positions: list[Position], fair_values: dict[str, float]) -> float:
    return sum(position_pnl(position, fair_values) for position in positions)


BASE_PORTFOLIO = [
    # KO combo
    Position("AC_45_KO", "BUY", 500, 0.175),
    Position("AC_35_P", "BUY", 22, 4.35),
    Position("AC_45_P", "SELL", 22, 9.05),

    # BP combo
    Position("AC_40_BP", "SELL", 50, 5.00),
    Position("AC_40_P", "BUY", 38, 6.55),

    # Chooser combo
    Position("AC_50_CO", "SELL", 50, 22.20),
    Position("AC_50_C", "BUY", 25, 12.05),
    Position("AC_50_P_2", "BUY", 25, 9.75),
    Position("AC_50_P", "BUY", 25, 12.05),
    Position("AC_50_C_2", "BUY", 25, 9.75),
]


def run_official_fair_value_settlements(n_settlements: int) -> pd.DataFrame:
    rows = []
    master_rng = np.random.default_rng()

    for settlement_id in range(1, n_settlements + 1):
        settlement_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        paths = simulate_gbm_paths(OFFICIAL_PATHS_PER_SETTLEMENT, seed=settlement_seed)
        fair_values = fair_values_from_payoffs(build_payoffs(paths))
        rows.append({
            "settlement_id": settlement_id,
            "seed": settlement_seed,
            **fair_values,
        })

    return pd.DataFrame(rows)


def signed_hedge_pnl(fair_values: np.ndarray, quantity: int, bid: float, ask: float) -> np.ndarray:
    if quantity > 0:
        return (fair_values - ask) * quantity * CONTRACT_SIZE
    if quantity < 0:
        return (bid - fair_values) * abs(quantity) * CONTRACT_SIZE
    return np.zeros_like(fair_values)


def summarize_total_pnl(
    q_ac: int,
    q_60c: int,
    base_pnl: np.ndarray,
    hedge_pnl: np.ndarray,
) -> dict[str, float | int]:
    total_pnl = base_pnl + hedge_pnl
    return {
        "q_ac": q_ac,
        "q_60c": q_60c,
        "mean_total_pnl": float(np.mean(total_pnl)),
        "std_total_pnl": float(np.std(total_pnl, ddof=1)),
        "profit_probability": float(np.mean(total_pnl > 0)),
        "min_total_pnl": float(np.min(total_pnl)),
        "p05_total_pnl": float(np.percentile(total_pnl, 5)),
        "median_total_pnl": float(np.percentile(total_pnl, 50)),
        "p95_total_pnl": float(np.percentile(total_pnl, 95)),
        "max_total_pnl": float(np.max(total_pnl)),
        "mean_base_pnl": float(np.mean(base_pnl)),
        "mean_hedge_pnl": float(np.mean(hedge_pnl)),
    }


def grid_search(fair_values: pd.DataFrame) -> pd.DataFrame:
    base_pnl = np.array([
        portfolio_pnl(BASE_PORTFOLIO, row)
        for row in fair_values.to_dict(orient="records")
    ])

    ac_fair = fair_values["AC"].to_numpy()
    c60_fair = fair_values["AC_60_C"].to_numpy()

    ac_pnl_by_q = {
        q_ac: signed_hedge_pnl(ac_fair, q_ac, bid=49.975, ask=50.025)
        for q_ac in Q_AC_RANGE
    }
    c60_pnl_by_q = {
        q_60c: signed_hedge_pnl(c60_fair, q_60c, bid=8.80, ask=8.85)
        for q_60c in Q_60C_RANGE
    }

    rows = []
    for q_ac, ac_pnl in ac_pnl_by_q.items():
        for q_60c, c60_pnl in c60_pnl_by_q.items():
            rows.append(summarize_total_pnl(
                q_ac=q_ac,
                q_60c=q_60c,
                base_pnl=base_pnl,
                hedge_pnl=ac_pnl + c60_pnl,
            ))

    return pd.DataFrame(rows)


def print_best(label: str, row: pd.Series) -> None:
    print(f"\n{label}:")
    print(f"- q_ac={int(row['q_ac'])}, q_60c={int(row['q_60c'])}")
    print(f"- mean_total_pnl={row['mean_total_pnl']:,.2f}")
    print(f"- std_total_pnl={row['std_total_pnl']:,.2f}")
    print(f"- profit_probability={row['profit_probability']:.1%}")
    print(f"- p05_total_pnl={row['p05_total_pnl']:,.2f}")
    print(f"- median_total_pnl={row['median_total_pnl']:,.2f}")
    print(f"- p95_total_pnl={row['p95_total_pnl']:,.2f}")
    print(f"- min/max={row['min_total_pnl']:,.2f} / {row['max_total_pnl']:,.2f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fair_values = run_official_fair_value_settlements(N_SETTLEMENTS)
    results = grid_search(fair_values)

    by_mean = results.sort_values(
        ["mean_total_pnl", "profit_probability", "p05_total_pnl"],
        ascending=[False, False, False],
    )
    by_profit_probability = results.sort_values(
        ["profit_probability", "mean_total_pnl", "p05_total_pnl"],
        ascending=[False, False, False],
    )
    by_p05 = results.sort_values(
        ["p05_total_pnl", "mean_total_pnl", "profit_probability"],
        ascending=[False, False, False],
    )

    results.to_csv(OUTPUT_DIR / "ac_60c_grid_search_results.csv", index=False)
    by_mean.head(20).to_csv(OUTPUT_DIR / "top20_by_mean_pnl.csv", index=False)
    by_profit_probability.head(20).to_csv(OUTPUT_DIR / "top20_by_profit_probability.csv", index=False)
    by_p05.head(20).to_csv(OUTPUT_DIR / "top20_by_p05_pnl.csv", index=False)

    base_row = results[(results["q_ac"] == 0) & (results["q_60c"] == 0)].iloc[0]

    print("\nSaved AC / AC_60_C hedge grid outputs to:", OUTPUT_DIR.resolve())
    print_best("Best pair by mean PnL", by_mean.iloc[0])
    print_best("Best pair by profit probability", by_profit_probability.iloc[0])
    print_best("Best pair by 5th percentile", by_p05.iloc[0])
    print_best("Base portfolio stats, q_ac=0 and q_60c=0", base_row)


if __name__ == "__main__":
    main()
