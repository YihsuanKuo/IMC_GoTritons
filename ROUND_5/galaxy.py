from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math


class Trader:
    """
    Galaxy Sounds v3: Rings core + heavily gated scouts.

    Motivation from previous run:
      - GALAXY_SOUNDS_PLANETARY_RINGS short produced almost all profit.
      - The other four products were small negative / noisy.
      - Therefore we keep Rings as the core alpha and allow other products
        to trade only when momentum is strong, persistent, and current.

    This is not timestamp hardcoding:
      - Rings size changes with its own trend regime.
      - Non-rings products are dynamically ranked, but only extreme signals trade.
    """

    POSITION_LIMIT = 10

    PRODUCTS: List[str] = [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ]

    RINGS = "GALAXY_SOUNDS_PLANETARY_RINGS"
    SCOUT_PRODUCTS: List[str] = [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ]

    # =====================================================
    # Core Rings leg
    # =====================================================
    # Base belief: Rings is weak. Keep it as the only structural leg.
    RINGS_BASE_SHORT = -8
    RINGS_STRONG_SHORT = -10
    RINGS_REDUCED_SHORT = -4
    RINGS_FLAT = 0

    RINGS_SCORE_LOOKBACK = 160
    RINGS_VOL_LOOKBACK = 90

    # Rings score is normal momentum. Negative = Rings falling = short works.
    RINGS_STRONG_SHORT_SCORE = -0.60
    RINGS_REDUCE_SHORT_SCORE = 0.90
    RINGS_FLAT_SCORE = 1.50

    # =====================================================
    # Non-rings scout overlay
    # =====================================================
    ENABLE_SCOUTS = True

    FAST_LOOKBACK = 80
    SLOW_LOOKBACK = 240
    VOL_LOOKBACK = 100
    REBALANCE_EVERY = 25

    # Much stricter than v2. Scouts were not real alpha unless signal is extreme.
    SCOUT_ENTRY_SCORE = 1.20
    SCOUT_STRONG_SCORE = 1.80
    SCOUT_EXIT_SCORE = 0.35
    MIN_SCORE_SPREAD = 0.90

    # Require score sign to persist across multiple rebalances.
    CONFIRM_BARS = 3

    # Keep scouts small. Rings is the actual alpha.
    SCOUT_SMALL_SIZE = 1
    SCOUT_STRONG_SIZE = 2
    MAX_SCOUT_PRODUCTS = 1  # only one long OR one short scout at a time

    # Overall risk cap.
    MAX_GROSS_EXPOSURE = 14
    MAX_TRADE_SIZE = 3
    SCOUT_MAX_TRADE_SIZE = 1

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.score_history: Dict[str, List[float]] = {
            p: [] for p in self.SCOUT_PRODUCTS
        }
        self.cached_targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

    def bid(self):
        return 15

    # =====================================================
    # Helpers
    # =====================================================

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if (
            len(order_depth.buy_orders) == 0
            or len(order_depth.sell_orders) == 0
        ):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def mean(self, xs: List[float]) -> float:
        return sum(xs) / len(xs)

    def std(self, xs: List[float]) -> float:
        if len(xs) <= 1:
            return 0.0
        m = self.mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def clamp_position(self, x: float) -> int:
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    def update_history(self, state: TradingState) -> None:
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue
            self.mid_history[product].append(mid)
            if len(self.mid_history[product]) > 2000:
                self.mid_history[product] = self.mid_history[product][-2000:]

    # =====================================================
    # Scores
    # =====================================================

    def z_momentum_score(
        self, product: str, lookback: int, vol_lookback: int
    ) -> Optional[float]:
        hist = self.mid_history[product]
        need = max(lookback, vol_lookback) + 1
        if len(hist) < need:
            return None

        now = hist[-1]
        old = hist[-lookback]
        if old <= 0:
            return None

        ret = (now - old) / old

        recent = hist[-vol_lookback:]
        pct_changes: List[float] = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                pct_changes.append((recent[i] - recent[i - 1]) / recent[i - 1])

        vol = self.std(pct_changes)
        if vol <= 1e-12:
            return None

        return ret / (vol * math.sqrt(lookback))

    def combined_score(self, product: str) -> Optional[float]:
        fast = self.z_momentum_score(
            product, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        slow = self.z_momentum_score(
            product, self.SLOW_LOOKBACK, self.VOL_LOOKBACK
        )

        if fast is None and slow is None:
            return None
        if slow is None:
            return fast
        if fast is None:
            return slow

        # Slow trend is primary; fast trend gives responsiveness.
        return 0.70 * slow + 0.30 * fast

    def rings_target(self) -> int:
        score = self.z_momentum_score(
            self.RINGS,
            self.RINGS_SCORE_LOOKBACK,
            self.RINGS_VOL_LOOKBACK,
        )

        # Before history is available, enter moderate short immediately.
        if score is None:
            return self.RINGS_BASE_SHORT

        if score >= self.RINGS_FLAT_SCORE:
            return self.RINGS_FLAT
        if score >= self.RINGS_REDUCE_SHORT_SCORE:
            return self.RINGS_REDUCED_SHORT
        if score <= self.RINGS_STRONG_SHORT_SCORE:
            return self.RINGS_STRONG_SHORT
        return self.RINGS_BASE_SHORT

    def update_score_history(self, scores: Dict[str, float]) -> None:
        for product, score in scores.items():
            self.score_history[product].append(score)
            if len(self.score_history[product]) > 20:
                self.score_history[product] = self.score_history[product][-20:]

    def confirmed_direction(self, product: str, direction: int) -> bool:
        """
        direction = +1 for long, -1 for short.
        Require recent score signs to agree with direction.
        """
        hist = self.score_history.get(product, [])
        if len(hist) < self.CONFIRM_BARS:
            return False
        recent = hist[-self.CONFIRM_BARS :]
        if direction > 0:
            return all(x > 0 for x in recent)
        return all(x < 0 for x in recent)

    # =====================================================
    # Target generation
    # =====================================================

    def build_targets(self, state: TradingState) -> Dict[str, int]:
        targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

        # 1) Core Rings short.
        targets[self.RINGS] = self.rings_target()

        if not self.ENABLE_SCOUTS:
            return targets

        # 2) Scores for non-rings products.
        scores: Dict[str, float] = {}
        for product in self.SCOUT_PRODUCTS:
            s = self.combined_score(product)
            if s is not None:
                scores[product] = s

        if len(scores) < 3:
            return targets

        self.update_score_history(scores)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_product, best_score = ranked[0]
        worst_product, worst_score = ranked[-1]

        # If dispersion is weak, keep scouts flat.
        if best_score - worst_score < self.MIN_SCORE_SPREAD:
            return targets

        # Pick the single stronger absolute opportunity: long strongest or short weakest.
        candidates = []

        if best_score >= self.SCOUT_ENTRY_SCORE and self.confirmed_direction(
            best_product, +1
        ):
            size = (
                self.SCOUT_STRONG_SIZE
                if best_score >= self.SCOUT_STRONG_SCORE
                else self.SCOUT_SMALL_SIZE
            )
            candidates.append((abs(best_score), best_product, size))

        if (
            worst_score <= -self.SCOUT_ENTRY_SCORE
            and self.confirmed_direction(worst_product, -1)
        ):
            size = (
                -self.SCOUT_STRONG_SIZE
                if worst_score <= -self.SCOUT_STRONG_SCORE
                else -self.SCOUT_SMALL_SIZE
            )
            candidates.append((abs(worst_score), worst_product, size))

        candidates.sort(reverse=True)
        for _, product, size in candidates[: self.MAX_SCOUT_PRODUCTS]:
            targets[product] = size

        # If a current scout position exists but its score has faded, flatten it.
        # This prevents stale scout positions from lingering.
        for product in self.SCOUT_PRODUCTS:
            current_pos = state.position.get(product, 0)
            if current_pos != 0 and product not in [
                c[1] for c in candidates[: self.MAX_SCOUT_PRODUCTS]
            ]:
                s = scores.get(product, 0.0)
                if abs(s) > self.SCOUT_EXIT_SCORE:
                    # keep tiny existing position only if signal still agrees
                    if current_pos > 0 and s > 0:
                        targets[product] = min(
                            current_pos, self.SCOUT_SMALL_SIZE
                        )
                    elif current_pos < 0 and s < 0:
                        targets[product] = max(
                            current_pos, -self.SCOUT_SMALL_SIZE
                        )
                    else:
                        targets[product] = 0
                else:
                    targets[product] = 0

        # Clamp individual limits.
        for product in self.PRODUCTS:
            targets[product] = self.clamp_position(targets[product])

        # Gross cap, reducing scouts first.
        gross = sum(abs(x) for x in targets.values())
        if gross > self.MAX_GROSS_EXPOSURE:
            excess = gross - self.MAX_GROSS_EXPOSURE
            non_rings = sorted(
                [p for p in self.SCOUT_PRODUCTS if targets[p] != 0],
                key=lambda p: abs(targets[p]),
                reverse=True,
            )
            for product in non_rings:
                if excess <= 0:
                    break
                reduce_by = min(abs(targets[product]), excess)
                if targets[product] > 0:
                    targets[product] -= reduce_by
                else:
                    targets[product] += reduce_by
                excess -= reduce_by

        return targets

    def get_targets(self, state: TradingState) -> Dict[str, int]:
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets(state)
        return self.cached_targets

    # =====================================================
    # Execution
    # =====================================================

    def product_trade_cap(self, product: str) -> int:
        if product == self.RINGS:
            return self.MAX_TRADE_SIZE
        return self.SCOUT_MAX_TRADE_SIZE

    def move_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ) -> List[Order]:
        orders: List[Order] = []

        target_pos = self.clamp_position(target_pos)
        needed = target_pos - current_pos
        if needed == 0:
            return orders

        trade_cap = self.product_trade_cap(product)

        if needed > 0:
            remaining = min(
                needed, self.POSITION_LIMIT - current_pos, trade_cap
            )
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                ask_volume = -order_depth.sell_orders[ask_price]
                if ask_volume <= 0:
                    continue
                qty = min(remaining, ask_volume)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    remaining -= qty
        else:
            remaining = min(
                -needed, self.POSITION_LIMIT + current_pos, trade_cap
            )
            for bid_price in sorted(
                order_depth.buy_orders.keys(), reverse=True
            ):
                if remaining <= 0:
                    break
                bid_volume = order_depth.buy_orders[bid_price]
                if bid_volume <= 0:
                    continue
                qty = min(remaining, bid_volume)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    remaining -= qty

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {p: [] for p in self.PRODUCTS}
        conversions = 0
        traderData = ""

        self.tick += 1
        self.update_history(state)
        targets = self.get_targets(state)

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            current_pos = state.position.get(product, 0)
            target_pos = targets.get(product, 0)
            result[product] = self.move_to_target(
                product=product,
                order_depth=state.order_depths[product],
                current_pos=current_pos,
                target_pos=target_pos,
            )

        return result, conversions, traderData
