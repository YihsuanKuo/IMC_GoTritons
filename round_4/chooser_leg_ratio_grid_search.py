#!/usr/bin/env python3
"""
Chooser leg ratio grid search.

The product payoff logic is imported from aether_fair_value_simulator.py so this
script does not redefine the chooser or option payoff rules.
"""

from __future__ import annotations

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
COARSE_STEP = 5
REFINE_RADIUS = 5
REFINE_TOP_N = 20
BATCH_SIZE = 5000


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


def coarse_candidates() -> set[tuple[int, int, int, int]]:
    values = range(0, 51, COARSE_STEP)
    return {
        (q_c3, q_p2, q_p3, q_c2)
        for q_c3 in values
        for q_p2 in values
        for q_p3 in values
        for q_c2 in values
        if q_c3 + q_p2 + q_p3 + q_c2 == 100
    }


def refinement_candidates(top_coarse: pd.DataFrame) -> set[tuple[int, int, int, int]]:
    candidates: set[tuple[int, int, int, int]] = set()
    for row in top_coarse.itertuples(index=False):
        centers = [int(row.q_C3), int(row.q_P2), int(row.q_P3), int(row.q_C2)]
        ranges = [
            range(max(0, center - REFINE_RADIUS), min(50, center + REFINE_RADIUS) + 1)
            for center in centers
        ]
        for q_c3 in ranges[0]:
            for q_p2 in ranges[1]:
                for q_p3 in ranges[2]:
                    for q_c2 in ranges[3]:
                        if q_c3 + q_p2 + q_p3 + q_c2 == 100:
                            candidates.add((q_c3, q_p2, q_p3, q_c2))
    return candidates


def evaluate_candidates(
    candidates: set[tuple[int, int, int, int]],
    fair_values: pd.DataFrame,
) -> pd.DataFrame:
    candidate_array = np.array(sorted(candidates), dtype=np.int16)

    base_pnl = (22.20 - fair_values["AC_50_CO"].to_numpy()) * 50 * CONTRACT_SIZE
    unit_pnls = np.vstack([
        (fair_values["AC_50_C"].to_numpy() - 12.05) * CONTRACT_SIZE,
        (fair_values["AC_50_P_2"].to_numpy() - 9.75) * CONTRACT_SIZE,
        (fair_values["AC_50_P"].to_numpy() - 12.05) * CONTRACT_SIZE,
        (fair_values["AC_50_C_2"].to_numpy() - 9.75) * CONTRACT_SIZE,
    ])

    rows = []
    for start in range(0, len(candidate_array), BATCH_SIZE):
        batch = candidate_array[start:start + BATCH_SIZE].astype(float)
        pnl = base_pnl[np.newaxis, :] + batch @ unit_pnls

        batch_df = pd.DataFrame({
            "q_C3": batch[:, 0].astype(int),
            "q_P2": batch[:, 1].astype(int),
            "q_P3": batch[:, 2].astype(int),
            "q_C2": batch[:, 3].astype(int),
            "mean_pnl": np.mean(pnl, axis=1),
            "std_pnl": np.std(pnl, axis=1, ddof=1),
            "profit_probability": np.mean(pnl > 0, axis=1),
            "min_pnl": np.min(pnl, axis=1),
            "p05_pnl": np.percentile(pnl, 5, axis=1),
            "median_pnl": np.percentile(pnl, 50, axis=1),
            "p95_pnl": np.percentile(pnl, 95, axis=1),
            "max_pnl": np.max(pnl, axis=1),
        })
        rows.append(batch_df)

    return pd.concat(rows, ignore_index=True)


def add_required_reference_candidates(candidates: set[tuple[int, int, int, int]]) -> None:
    for combo in [(25, 25, 25, 25), (50, 50, 0, 0), (0, 0, 50, 50)]:
        if sum(combo) == 100:
            candidates.add(combo)


def print_combo(label: str, row: pd.Series) -> None:
    print(f"\n{label}:")
    print(f"- q_C3={int(row['q_C3'])}, q_P2={int(row['q_P2'])}, q_P3={int(row['q_P3'])}, q_C2={int(row['q_C2'])}")
    print(f"- mean_pnl={row['mean_pnl']:,.2f}")
    print(f"- std_pnl={row['std_pnl']:,.2f}")
    print(f"- profit_probability={row['profit_probability']:.1%}")
    print(f"- p05_pnl={row['p05_pnl']:,.2f}")
    print(f"- median_pnl={row['median_pnl']:,.2f}")
    print(f"- p95_pnl={row['p95_pnl']:,.2f}")
    print(f"- min/max={row['min_pnl']:,.2f} / {row['max_pnl']:,.2f}")


def find_exact(results: pd.DataFrame, combo: tuple[int, int, int, int]) -> pd.Series:
    q_c3, q_p2, q_p3, q_c2 = combo
    match = results[
        (results["q_C3"] == q_c3)
        & (results["q_P2"] == q_p2)
        & (results["q_P3"] == q_p3)
        & (results["q_C2"] == q_c2)
    ]
    return match.iloc[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fair_values = run_official_fair_value_settlements(N_SETTLEMENTS)

    coarse = evaluate_candidates(coarse_candidates(), fair_values)
    top_coarse = coarse.sort_values(
        ["mean_pnl", "profit_probability", "p05_pnl"],
        ascending=[False, False, False],
    ).head(REFINE_TOP_N)

    final_candidates = coarse_candidates() | refinement_candidates(top_coarse)
    add_required_reference_candidates(final_candidates)
    results = evaluate_candidates(final_candidates, fair_values)

    by_mean = results.sort_values(
        ["mean_pnl", "profit_probability", "p05_pnl"],
        ascending=[False, False, False],
    )
    by_profit_probability = results.sort_values(
        ["profit_probability", "mean_pnl", "p05_pnl"],
        ascending=[False, False, False],
    )
    by_p05 = results.sort_values(
        ["p05_pnl", "mean_pnl", "profit_probability"],
        ascending=[False, False, False],
    )

    by_mean.to_csv(OUTPUT_DIR / "chooser_leg_ratio_grid_search.csv", index=False)
    by_mean.head(20).to_csv(OUTPUT_DIR / "chooser_top20_by_mean_pnl.csv", index=False)
    by_profit_probability.head(20).to_csv(OUTPUT_DIR / "chooser_top20_by_profit_probability.csv", index=False)
    by_p05.head(20).to_csv(OUTPUT_DIR / "chooser_top20_by_p05_pnl.csv", index=False)

    print("\nSaved chooser ratio grid outputs to:", OUTPUT_DIR.resolve())
    print(f"Coarse candidates evaluated: {len(coarse):,}")
    print(f"Final refined candidates evaluated: {len(results):,}")
    print_combo("Best combo by mean_pnl", by_mean.iloc[0])
    print_combo("Best combo by profit_probability", by_profit_probability.iloc[0])
    print_combo("Best combo by p05_pnl", by_p05.iloc[0])
    print_combo("Original 25A+25B combo", find_exact(results, (25, 25, 25, 25)))
    print_combo("Pure A combo", find_exact(results, (50, 50, 0, 0)))
    print_combo("Pure B combo", find_exact(results, (0, 0, 50, 50)))


if __name__ == "__main__":
    main()
