import numpy as np

TOTAL_BUDGET = 50_000


def research(x: float) -> float:
    """Research outcome for percentage x in [0, 100]."""
    return 200_000 * np.log(1 + x) / np.log(1 + 100)


def scale(x: float) -> float:
    """Scale outcome for percentage x in [0, 100]."""
    return 7 * x / 100


def speed_multiplier_uniform(my_speed: float, comp_low: float, comp_high: float) -> float:
    """
    Assume competitors' Speed investments are uniformly distributed on [comp_low, comp_high].
    """
    if comp_low > comp_high:
        raise ValueError("comp_low must be <= comp_high")
    if not (0 <= my_speed <= 100 and 0 <= comp_low <= 100 and 0 <= comp_high <= 100):
        raise ValueError("All speed values must be between 0 and 100")

    if comp_low == comp_high:
        if my_speed < comp_low:
            return 0.1
        elif my_speed > comp_high:
            return 0.9
        else:
            return 0.9  # tying the top investment gets the top rank

    if my_speed <= comp_low:
        return 0.1
    if my_speed >= comp_high:
        return 0.9

    percentile = (my_speed - comp_low) / (comp_high - comp_low)
    return 0.1 + 0.8 * percentile


def compute_final_pnl(r_pct: int, s_pct: int, speed_pct: int, speed_multiplier: float) -> float:
    """
    Final PnL:
        research_outcome * scale_outcome * speed_multiplier - budget_used
    """
    total_used_pct = r_pct + s_pct + speed_pct
    if min(r_pct, s_pct, speed_pct) < 0 or total_used_pct > 100:
        return float("-inf")

    budget_used = TOTAL_BUDGET * total_used_pct / 100
    gross = research(r_pct) * scale(s_pct) * speed_multiplier
    return gross - budget_used


def find_best_research_scale_for_speed(speed_pct: int, multiplier: float):
    """
    For a fixed Speed percentage and its multiplier, search all valid
    Research/Scale splits in the remaining budget.
    """
    best_pnl = float("-inf")
    best_combo = None

    remaining = 100 - speed_pct
    for r in range(remaining + 1):
        for s in range(remaining - r + 1):
            pnl = compute_final_pnl(r, s, speed_pct, multiplier)
            if pnl > best_pnl:
                best_pnl = pnl
                best_combo = (r, s)

    return best_combo, best_pnl


def find_best_overall(comp_low: float, comp_high: float):
    best_result = None

    for my_speed in range(0, 101):
        multiplier = speed_multiplier_uniform(my_speed, comp_low, comp_high)
        (best_r, best_s), final_pnl = find_best_research_scale_for_speed(my_speed, multiplier)

        result = {
            "speed_pct": my_speed,
            "research_pct": best_r,
            "scale_pct": best_s,
            "unused_pct": 100 - my_speed - best_r - best_s,
            "speed_multiplier": multiplier,
            "final_pnl": final_pnl,
        }

        if best_result is None or result["final_pnl"] > best_result["final_pnl"]:
            best_result = result

    return best_result


def print_examples(comp_low: float, comp_high: float):
    print("\nSample multiplier points under your uniform assumption:")
    samples = [0, int(comp_low), int((comp_low + comp_high) / 2), int(comp_high), 100]
    shown = set()
    for x in samples:
        x = max(0, min(100, x))
        if x not in shown:
            shown.add(x)
            m = speed_multiplier_uniform(x, comp_low, comp_high)
            print(f"  Speed {x:>3}% -> multiplier {m:.4f}")


if __name__ == "__main__":
    print("=== Invest & Expand Optimizer (Uniform Speed Range Model) ===")
    print("Assumption: competitors' Speed choices are evenly distributed in your input range.\n")

    comp_low = float(input("Competitor Speed lower bound (e.g. 10): ").strip())
    comp_high = float(input("Competitor Speed upper bound (e.g. 50): ").strip())

    best = find_best_overall(comp_low, comp_high)

    print_examples(comp_low, comp_high)

    print("\n=== Best overall allocation found ===")
    print(f"Speed:      {best['speed_pct']}%")
    print(f"Research:   {best['research_pct']}%")
    print(f"Scale:      {best['scale_pct']}%")
    print(f"Unused:     {best['unused_pct']}%")
    print(f"Multiplier: {best['speed_multiplier']:.6f}")
    print(f"Final PnL:  {best['final_pnl']:,.2f}")
