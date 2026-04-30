from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math


class Trader:
    """
    Vertical Sleeping Pods v2: Cotton/Wool core + adaptive risk guard.

    Why this version:
      - The previous all-product dynamic strategy lost because Suede/Polyester/Nylon
        were noisy and caused churn.
      - In the public log, Cotton and Lamb Wool were the only consistently positive
        contributors.
      - This version isolates that edge:
            COTTON: core long
            LAMB_WOOL: core short
            SUEDE / POLYESTER / NYLON: flat by default
      - It is still dynamic: Cotton/Wool targets reduce or flatten if their own
        price regime moves strongly against the core view.

    This is a standalone Sleeping Pods strategy file.
    """

    POSITION_LIMIT = 10

    PRODUCTS: List[str] = [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ]

    COTTON = "SLEEP_POD_COTTON"
    WOOL = "SLEEP_POD_LAMB_WOOL"

    # Core targets.
    COTTON_BASE_LONG = 8
    COTTON_STRONG_LONG = 10
    COTTON_REDUCED_LONG = 4
    COTTON_FLAT = 0
    COTTON_SMALL_SHORT = -2

    WOOL_BASE_SHORT = -8
    WOOL_STRONG_SHORT = -10
    WOOL_REDUCED_SHORT = -4
    WOOL_FLAT = 0
    WOOL_SMALL_LONG = 2

    # Signal settings.
    FAST_LOOKBACK = 50
    SLOW_LOOKBACK = 180
    VOL_LOOKBACK = 85
    SHOCK_LOOKBACK = 18
    Z_LOOKBACK = 260

    REBALANCE_EVERY = 20

    # Cotton score thresholds:
    # positive score = uptrend, negative score = downtrend.
    COTTON_STRONG_SCORE = 0.65
    COTTON_REDUCE_SCORE = -0.55
    COTTON_FLAT_SCORE = -1.05
    COTTON_SHORT_SCORE = -1.75

    # Wool score thresholds:
    # negative score = downtrend, positive score = uptrend.
    WOOL_STRONG_SHORT_SCORE = -0.65
    WOOL_REDUCE_SHORT_SCORE = 0.55
    WOOL_FLAT_SCORE = 1.05
    WOOL_LONG_SCORE = 1.75

    # Stretched-position guard.
    # If a core position is very overextended and fast momentum turns against us,
    # reduce one level instead of flipping aggressively.
    EXTREME_Z = 2.40

    MAX_GROSS_EXPOSURE = 18
    MAX_NET_EXPOSURE = 6

    MAX_TRADE_SIZE = 3

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
            if len(self.mid_history[product]) > 2500:
                self.mid_history[product] = self.mid_history[product][-2500:]

    # =====================================================
    # Signal construction
    # =====================================================

    def pct_vol(self, product: str, lookback: int) -> Optional[float]:
        hist = self.mid_history[product]
        if len(hist) < lookback + 1:
            return None

        recent = hist[-lookback:]
        changes: List[float] = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                changes.append((recent[i] - recent[i - 1]) / recent[i - 1])

        vol = self.std(changes)
        if vol <= 1e-12:
            return None
        return vol

    def momentum_score(
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
        vol = self.pct_vol(product, vol_lookback)
        if vol is None:
            return None

        return ret / (vol * math.sqrt(lookback))

    def shock_score(self, product: str) -> float:
        hist = self.mid_history[product]
        if len(hist) < self.SHOCK_LOOKBACK + self.VOL_LOOKBACK + 1:
            return 0.0

        now = hist[-1]
        old = hist[-self.SHOCK_LOOKBACK]
        if old <= 0:
            return 0.0

        ret = (now - old) / old
        vol = self.pct_vol(product, self.VOL_LOOKBACK)
        if vol is None:
            return 0.0

        raw = ret / (vol * math.sqrt(self.SHOCK_LOOKBACK))
        return max(-2.0, min(2.0, raw))

    def combined_score(self, product: str) -> Optional[float]:
        fast = self.momentum_score(
            product, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        slow = self.momentum_score(
            product, self.SLOW_LOOKBACK, self.VOL_LOOKBACK
        )

        if fast is None and slow is None:
            return None
        if slow is None:
            base = fast
        elif fast is None:
            base = slow
        else:
            base = 0.70 * slow + 0.30 * fast

        shock = self.shock_score(product)
        return base + 0.15 * shock

    def rolling_z(self, product: str) -> Optional[float]:
        hist = self.mid_history[product]
        if len(hist) < self.Z_LOOKBACK:
            return None
        window = hist[-self.Z_LOOKBACK :]
        s = self.std(window)
        if s <= 1e-9:
            return None
        return (window[-1] - self.mean(window)) / s

    # =====================================================
    # Target logic
    # =====================================================

    def cotton_target(self) -> int:
        score = self.combined_score(self.COTTON)

        # Before enough history: use core long.
        if score is None:
            target = self.COTTON_BASE_LONG
        elif score >= self.COTTON_STRONG_SCORE:
            target = self.COTTON_STRONG_LONG
        elif score <= self.COTTON_SHORT_SCORE:
            target = self.COTTON_SMALL_SHORT
        elif score <= self.COTTON_FLAT_SCORE:
            target = self.COTTON_FLAT
        elif score <= self.COTTON_REDUCE_SCORE:
            target = self.COTTON_REDUCED_LONG
        else:
            target = self.COTTON_BASE_LONG

        # Overextension guard: if long cotton is very stretched and short-term
        # momentum turns down, reduce but do not immediately flip.
        z = self.rolling_z(self.COTTON)
        fast = self.momentum_score(
            self.COTTON, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        if target > 0 and z is not None and fast is not None:
            if z > self.EXTREME_Z and fast < -0.10:
                target = min(target, self.COTTON_REDUCED_LONG)

        return self.clamp_position(target)

    def wool_target(self) -> int:
        score = self.combined_score(self.WOOL)

        # Before enough history: use core short.
        if score is None:
            target = self.WOOL_BASE_SHORT
        elif score <= self.WOOL_STRONG_SHORT_SCORE:
            target = self.WOOL_STRONG_SHORT
        elif score >= self.WOOL_LONG_SCORE:
            target = self.WOOL_SMALL_LONG
        elif score >= self.WOOL_FLAT_SCORE:
            target = self.WOOL_FLAT
        elif score >= self.WOOL_REDUCE_SHORT_SCORE:
            target = self.WOOL_REDUCED_SHORT
        else:
            target = self.WOOL_BASE_SHORT

        # Overextension guard: if short wool is very stretched downward and
        # short-term momentum bounces, reduce but do not immediately flip.
        z = self.rolling_z(self.WOOL)
        fast = self.momentum_score(
            self.WOOL, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        if target < 0 and z is not None and fast is not None:
            if z < -self.EXTREME_Z and fast > 0.10:
                target = max(target, self.WOOL_REDUCED_SHORT)

        return self.clamp_position(target)

    def apply_risk_caps(self, targets: Dict[str, int]) -> Dict[str, int]:
        # Gross exposure cap.
        gross = sum(abs(x) for x in targets.values())
        while gross > self.MAX_GROSS_EXPOSURE:
            nonzero = [p for p in self.PRODUCTS if targets[p] != 0]
            if not nonzero:
                break
            p = max(nonzero, key=lambda x: abs(targets[x]))
            if targets[p] > 0:
                targets[p] -= 1
            else:
                targets[p] += 1
            gross = sum(abs(x) for x in targets.values())

        # Net exposure cap.
        net = sum(targets.values())
        while abs(net) > self.MAX_NET_EXPOSURE:
            if net > 0:
                longs = [p for p in self.PRODUCTS if targets[p] > 0]
                if not longs:
                    break
                p = max(longs, key=lambda x: targets[x])
                targets[p] -= 1
                net -= 1
            else:
                shorts = [p for p in self.PRODUCTS if targets[p] < 0]
                if not shorts:
                    break
                p = min(shorts, key=lambda x: targets[x])
                targets[p] += 1
                net += 1

        for p in self.PRODUCTS:
            targets[p] = self.clamp_position(targets[p])
        return targets

    def build_targets(self) -> Dict[str, int]:
        targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

        # Only trade the two names with evidence of edge.
        targets[self.COTTON] = self.cotton_target()
        targets[self.WOOL] = self.wool_target()

        # Suede / Polyester / Nylon are intentionally flat:
        # the previous log showed they generated churn and negative PnL.
        targets["SLEEP_POD_SUEDE"] = 0
        targets["SLEEP_POD_POLYESTER"] = 0
        targets["SLEEP_POD_NYLON"] = 0

        return self.apply_risk_caps(targets)

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
