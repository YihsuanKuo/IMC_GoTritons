from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math


class Trader:
    """
    Round 5 Construction Panels strategy.

    Design:
        1. Static prior from historical cross-sectional alpha:
              long  PANEL_2X4
              short PANEL_2X2
              short PANEL_4X4

        2. Dynamic trend/ranking overlay:
              rank all five panels by recent risk-adjusted momentum.
              long strongest / short weakest, but only when dispersion is meaningful.

        3. Regime guard:
              if the static panel basket is working, keep full size.
              if the static panel basket weakens, reduce the static part.
              if an individual static leg strongly contradicts its prior, reduce that leg.

    This avoids pure hardcoding, but also avoids over-reactive flipping.
    """

    POSITION_LIMIT = 10

    PRODUCTS = [
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",
    ]

    # =====================================================
    # Static prior alpha
    # =====================================================
    # Historical robust read:
    #   PANEL_2X4 consistently strong.
    #   PANEL_2X2 and PANEL_4X4 weaker as short hedges.
    #   PANEL_1X2 and PANEL_1X4 are mixed; leave to dynamic overlay only.

    PRIOR_WEIGHTS: Dict[str, int] = {
        "PANEL_2X4": 1,
        "PANEL_2X2": -1,
        "PANEL_4X4": -1,
    }

    # Base size of static prior. This is intentionally not always 10,
    # because the dynamic overlay can add/subtract around it.
    PRIOR_BASE_SIZE = 7

    # =====================================================
    # Dynamic ranking parameters
    # =====================================================

    LOOKBACK = 250
    VOL_LOOKBACK = 120
    REBALANCE_EVERY = 25

    # If all scores are close, the ranking is noisy; avoid dynamic overlay.
    MIN_SCORE_SPREAD = 0.25

    # Dynamic overlay sizes. Added on top of the static prior.
    DYN_TOP_SIZE = 4
    DYN_SECOND_SIZE = 2
    DYN_BOTTOM_SIZE = -4
    DYN_SECOND_BOTTOM_SIZE = -2

    # Do not let dynamic overlay fully dominate the static prior too quickly.
    MAX_TARGET_ABS = 10

    # =====================================================
    # Regime guard parameters
    # =====================================================

    BASKET_LOOKBACK = 300
    LEG_LOOKBACK = 250
    REGIME_VOL_LOOKBACK = 120

    # Static basket scaling by health score.
    # health > 0 means prior basket has been working.
    # health < 0 means prior basket has been failing.
    BASKET_WEAK_SCORE = -1.25
    BASKET_BAD_SCORE = -2.25

    # Individual leg guard. If a prior leg is moving against its intended direction,
    # reduce its static contribution, but don't flip because of this alone.
    LEG_WEAK_SCORE = -1.50
    LEG_BAD_SCORE = -2.50

    # Execution
    MAX_TRADE_SIZE = 3

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.basket_history: List[float] = []
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

    def clamp_position(self, x: int) -> int:
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    # =====================================================
    # History update
    # =====================================================

    def update_history(self, state: TradingState) -> None:
        mids: Dict[str, float] = {}

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue
            mids[product] = mid
            self.mid_history[product].append(mid)
            if len(self.mid_history[product]) > 2000:
                self.mid_history[product] = self.mid_history[product][-2000:]

        # Static-prior basket value. If this rises, the static prior is making PnL.
        if all(p in mids for p in self.PRIOR_WEIGHTS):
            val = 0.0
            for product, weight in self.PRIOR_WEIGHTS.items():
                val += weight * mids[product]
            self.basket_history.append(val)
            if len(self.basket_history) > 2000:
                self.basket_history = self.basket_history[-2000:]

    # =====================================================
    # Signal functions
    # =====================================================

    def risk_adjusted_momentum(self, product: str) -> Optional[float]:
        hist = self.mid_history[product]
        if len(hist) < max(self.LOOKBACK, self.VOL_LOOKBACK) + 1:
            return None

        now = hist[-1]
        old = hist[-self.LOOKBACK]
        if old <= 0:
            return None

        ret = (now - old) / old

        recent = hist[-self.VOL_LOOKBACK :]
        pct_changes: List[float] = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            if prev > 0:
                pct_changes.append((recent[i] - prev) / prev)

        vol = self.std(pct_changes)
        if vol <= 1e-9:
            return None

        return ret / (vol * math.sqrt(self.LOOKBACK))

    def basket_health_score(self) -> float:
        hist = self.basket_history
        if len(hist) < max(self.BASKET_LOOKBACK, self.REGIME_VOL_LOOKBACK) + 1:
            return 0.0

        move = hist[-1] - hist[-self.BASKET_LOOKBACK]
        recent = hist[-self.REGIME_VOL_LOOKBACK :]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        vol = self.std(changes)
        if vol <= 1e-9:
            return 0.0

        return move / (vol * math.sqrt(self.BASKET_LOOKBACK))

    def leg_support_score(
        self, product: str, intended_direction: int
    ) -> float:
        """
        intended_direction = +1 for long prior, -1 for short prior.
        Positive score means the leg has moved in favor of the prior.
        Negative score means it has moved against the prior.
        """
        hist = self.mid_history[product]
        if len(hist) < max(self.LEG_LOOKBACK, self.REGIME_VOL_LOOKBACK) + 1:
            return 0.0

        raw_move = hist[-1] - hist[-self.LEG_LOOKBACK]
        favorable_move = intended_direction * raw_move

        recent = hist[-self.REGIME_VOL_LOOKBACK :]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        vol = self.std(changes)
        if vol <= 1e-9:
            return 0.0

        return favorable_move / (vol * math.sqrt(self.LEG_LOOKBACK))

    # =====================================================
    # Target construction
    # =====================================================

    def static_scale(self) -> float:
        score = self.basket_health_score()

        if score <= self.BASKET_BAD_SCORE:
            return 0.40
        if score <= self.BASKET_WEAK_SCORE:
            return 0.70
        return 1.00

    def leg_scale(self, product: str, intended_direction: int) -> float:
        score = self.leg_support_score(product, intended_direction)

        if score <= self.LEG_BAD_SCORE:
            return 0.40
        if score <= self.LEG_WEAK_SCORE:
            return 0.70
        return 1.00

    def static_prior_targets(self) -> Dict[str, int]:
        targets = {p: 0 for p in self.PRODUCTS}
        basket_scale = self.static_scale()

        for product, direction in self.PRIOR_WEIGHTS.items():
            scale = basket_scale * self.leg_scale(product, direction)
            targets[product] = self.clamp_position(
                direction * self.PRIOR_BASE_SIZE * scale
            )

        return targets

    def dynamic_ranking_targets(self) -> Dict[str, int]:
        targets = {p: 0 for p in self.PRODUCTS}

        scores: Dict[str, float] = {}
        for product in self.PRODUCTS:
            s = self.risk_adjusted_momentum(product)
            if s is not None:
                scores[product] = s

        if len(scores) < len(self.PRODUCTS):
            return targets

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        best_product, best_score = ranked[0]
        second_product, second_score = ranked[1]
        middle_product, middle_score = ranked[2]
        second_worst_product, second_worst_score = ranked[3]
        worst_product, worst_score = ranked[4]

        # If rankings are too compressed, do not add dynamic overlay.
        if best_score - worst_score < self.MIN_SCORE_SPREAD:
            return targets

        targets[best_product] += self.DYN_TOP_SIZE
        targets[second_product] += self.DYN_SECOND_SIZE
        targets[middle_product] += 0
        targets[second_worst_product] += self.DYN_SECOND_BOTTOM_SIZE
        targets[worst_product] += self.DYN_BOTTOM_SIZE

        return {p: self.clamp_position(v) for p, v in targets.items()}

    def build_targets(self) -> Dict[str, int]:
        # Immediate static entry at the beginning. Dynamic overlay only starts after warmup.
        targets = self.static_prior_targets()
        dyn = self.dynamic_ranking_targets()

        for product in self.PRODUCTS:
            targets[product] = self.clamp_position(
                targets.get(product, 0) + dyn.get(product, 0)
            )

        return targets

    def get_targets(self) -> Dict[str, int]:
        # Enter immediately, then rebalance slowly.
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets()
        return self.cached_targets

    # =====================================================
    # Execution
    # =====================================================

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

        if needed > 0:
            remaining = min(
                needed, self.POSITION_LIMIT - current_pos, self.MAX_TRADE_SIZE
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
                -needed, self.POSITION_LIMIT + current_pos, self.MAX_TRADE_SIZE
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
        targets = self.get_targets()

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
