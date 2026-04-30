from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math


class Trader:
    POSITION_LIMIT = 10

    PRODUCTS = [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ]

    LOOKBACK = 250
    VOL_LOOKBACK = 120
    REBALANCE_EVERY = 25
    MAX_TRADE_SIZE = 3

    # Dynamic ranking sizes
    TOP_SIZE = 10
    SECOND_SIZE = 5
    BOTTOM_SIZE = -10
    SECOND_BOTTOM_SIZE = -5

    # If scores are too close, don't force a trade.
    MIN_SCORE_GAP = 0.20

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
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

    def clamp_position(self, x: int) -> int:
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    # =====================================================
    # History
    # =====================================================

    def update_history(self, state: TradingState):
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue

            self.mid_history[product].append(mid)

            if len(self.mid_history[product]) > 1500:
                self.mid_history[product] = self.mid_history[product][-1500:]

    # =====================================================
    # Dynamic cross-sectional signal
    # =====================================================

    def product_score(self, product: str) -> Optional[float]:
        hist = self.mid_history[product]

        if len(hist) < max(self.LOOKBACK, self.VOL_LOOKBACK) + 1:
            return None

        now = hist[-1]
        old = hist[-self.LOOKBACK]

        if old <= 0:
            return None

        # Recent return
        ret = (now - old) / old

        # Volatility of percentage changes
        recent = hist[-self.VOL_LOOKBACK :]
        pct_changes = []

        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                pct_changes.append((recent[i] - recent[i - 1]) / recent[i - 1])

        vol = self.std(pct_changes)

        if vol <= 1e-9:
            return None

        return ret / (vol * math.sqrt(self.LOOKBACK))

    def build_targets(self) -> Dict[str, int]:
        targets = {p: 0 for p in self.PRODUCTS}

        scores = {}

        for product in self.PRODUCTS:
            score = self.product_score(product)
            if score is not None:
                scores[product] = score

        if len(scores) < len(self.PRODUCTS):
            return targets

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_product, best_score = ranked[0]
        second_product, second_score = ranked[1]
        middle_product, middle_score = ranked[2]
        second_worst_product, second_worst_score = ranked[3]
        worst_product, worst_score = ranked[4]

        # Avoid trading if the whole ranking is too compressed.
        if best_score - worst_score < self.MIN_SCORE_GAP:
            return targets

        targets[best_product] = self.TOP_SIZE
        targets[second_product] = self.SECOND_SIZE
        targets[middle_product] = 0
        targets[second_worst_product] = self.SECOND_BOTTOM_SIZE
        targets[worst_product] = self.BOTTOM_SIZE

        for product in targets:
            targets[product] = self.clamp_position(targets[product])

        return targets

    def get_targets(self) -> Dict[str, int]:
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
                needed,
                self.POSITION_LIMIT - current_pos,
                self.MAX_TRADE_SIZE,
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
                -needed,
                self.POSITION_LIMIT + current_pos,
                self.MAX_TRADE_SIZE,
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

