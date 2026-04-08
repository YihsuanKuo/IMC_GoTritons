from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # -----------------------------
    # Utility helpers
    # -----------------------------
    def clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def get_best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = -order_depth.sell_orders[best_ask]  # positive size
        return best_bid, best_ask, best_bid_vol, best_ask_vol

    def calc_microprice(
        self, best_bid: int, best_ask: int, bid_vol: int, ask_vol: int
    ) -> float:
        denom = bid_vol + ask_vol
        if denom <= 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_vol + best_ask * bid_vol) / denom

    def calc_book_imbalance(
        self, order_depth: OrderDepth, depth: int = 3
    ) -> float:
        buy_levels = sorted(order_depth.buy_orders.items(), reverse=True)[
            :depth
        ]
        sell_levels = sorted(order_depth.sell_orders.items())[:depth]

        bid_vol = sum(v for _, v in buy_levels)
        ask_vol = sum(-v for _, v in sell_levels)

        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def calc_vol_estimate(
        self, saved_data, product: str, mid_price: float
    ) -> float:
        prev_mid = saved_data.get(f"{product}_prev_mid", mid_price)
        prev_vol = saved_data.get(f"{product}_vol", 1.0)

        ret = mid_price - prev_mid
        vol = 0.2 * abs(ret) + 0.8 * prev_vol
        return max(vol, 0.5)

    def inventory_skew(
        self, position: int, limit: int, strength: float
    ) -> float:
        inv = position / limit
        return strength * inv

    # -----------------------------
    # Strategy blocks
    # -----------------------------
    def trade_emeralds(
        self,
        product: str,
        state: TradingState,
        order_depth: OrderDepth,
        saved_data: dict,
        new_data: dict,
    ) -> List[Order]:
        orders: List[Order] = []

        info = self.get_best_bid_ask(order_depth)
        if info is None:
            return orders

        best_bid, best_ask, best_bid_vol, best_ask_vol = info
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        micro = self.calc_microprice(
            best_bid, best_ask, best_bid_vol, best_ask_vol
        )
        imbalance = self.calc_book_imbalance(order_depth, depth=3)

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        max_buy = limit - pos
        max_sell = limit + pos

        anchor = 10000.0

        fair = 0.65 * anchor + 0.35 * micro

        # weaker inventory penalty -> more activity
        fair -= self.inventory_skew(pos, limit, strength=4.0)

        fair += 0.8 * imbalance

        inv_frac = abs(pos) / limit

        # narrower width -> higher fill probability
        base_half_width = 1 if spread >= 3 else 0
        extra_width = 1 if inv_frac > 0.65 else 0
        half_width = base_half_width + extra_width

        buy_quote = min(best_bid + 1, int(fair - half_width))
        sell_quote = max(best_ask - 1, int(fair + half_width))

        # slightly larger passive size
        bid_size = 14
        ask_size = 14

        if inv_frac > 0.7:
            bid_size = 7
            ask_size = 7

        # inventory-aware bias kicks in later
        if pos > 60:
            bid_size = 4
            sell_quote = max(best_bid + 1, min(best_ask - 1, int(fair)))
        elif pos < -60:
            ask_size = 4
            buy_quote = min(best_ask - 1, max(best_bid + 1, int(fair)))

        if max_buy > 0:
            orders.append(Order(product, buy_quote, min(bid_size, max_buy)))
        if max_sell > 0:
            orders.append(Order(product, sell_quote, -min(ask_size, max_sell)))

        # slightly more aggressive anchor capture
        if max_buy > 0 and best_ask <= anchor - 1:
            take_qty = min(best_ask_vol, max_buy, 10)
            if take_qty > 0:
                orders.append(Order(product, best_ask, take_qty))

        if max_sell > 0 and best_bid >= anchor + 1:
            take_qty = min(best_bid_vol, max_sell, 10)
            if take_qty > 0:
                orders.append(Order(product, best_bid, -take_qty))

        new_data[f"{product}_prev_mid"] = mid
        return orders

    def trade_tomatoes(
        self,
        product: str,
        state: TradingState,
        order_depth: OrderDepth,
        saved_data: dict,
        new_data: dict,
    ) -> List[Order]:
        orders: List[Order] = []

        info = self.get_best_bid_ask(order_depth)
        if info is None:
            return orders

        best_bid, best_ask, best_bid_vol, best_ask_vol = info
        mid = (best_bid + best_ask) / 2
        micro = self.calc_microprice(
            best_bid, best_ask, best_bid_vol, best_ask_vol
        )
        imbalance = self.calc_book_imbalance(order_depth, depth=3)

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        max_buy = limit - pos
        max_sell = limit + pos

        prev_mid = saved_data.get(f"{product}_prev_mid", mid)
        prev_fast = saved_data.get(f"{product}_ema_fast", mid)
        prev_slow = saved_data.get(f"{product}_ema_slow", mid)
        prev_vol = saved_data.get(f"{product}_vol", 1.0)

        lam_fast = 0.25
        lam_slow = 0.08

        ema_fast = lam_fast * mid + (1 - lam_fast) * prev_fast
        ema_slow = lam_slow * mid + (1 - lam_slow) * prev_slow
        vol = 0.2 * abs(mid - prev_mid) + 0.8 * prev_vol

        new_data[f"{product}_prev_mid"] = mid
        new_data[f"{product}_ema_fast"] = ema_fast
        new_data[f"{product}_ema_slow"] = ema_slow
        new_data[f"{product}_vol"] = vol

        short_move = mid - prev_mid
        trend = ema_fast - ema_slow
        deviation = mid - ema_slow

        # weaker inventory penalty -> more activity
        fair = micro - self.inventory_skew(pos, limit, strength=5.0)
        fair += 1.5 * imbalance

        # less restrictive high-vol filter
        high_vol = vol > 3.5

        # lower thresholds -> more aggressive entry frequency
        extreme_dev_up = deviation > max(1.2, 0.8 * vol)
        extreme_dev_dn = deviation < -max(1.2, 0.8 * vol)

        inv_frac = abs(pos) / limit

        # narrower width
        base_half_width = 1
        if high_vol:
            base_half_width += 1
        if inv_frac > 0.65:
            base_half_width += 1

        bid_size = 8
        ask_size = 8
        if high_vol:
            bid_size = 5
            ask_size = 5
        if inv_frac > 0.7:
            bid_size = 4
            ask_size = 4

        # more frequent aggressive fades with controlled size
        if extreme_dev_up and short_move > 0 and max_sell > 0:
            sell_qty = min(best_bid_vol, max_sell, 6 if not high_vol else 4)
            if sell_qty > 0 and best_bid >= fair:
                orders.append(Order(product, best_bid, -sell_qty))

        if extreme_dev_dn and short_move < 0 and max_buy > 0:
            buy_qty = min(best_ask_vol, max_buy, 6 if not high_vol else 4)
            if buy_qty > 0 and best_ask <= fair:
                orders.append(Order(product, best_ask, buy_qty))

        buy_quote = min(best_bid + 1, int(fair - base_half_width))
        sell_quote = max(best_ask - 1, int(fair + base_half_width))

        # inventory pressure kicks in later
        if pos > 60:
            buy_quote = min(best_bid, int(fair - base_half_width - 1))
            sell_quote = max(best_bid + 1, min(best_ask - 1, int(fair)))
            bid_size = 3
            ask_size = max(ask_size, 9)

        elif pos < -60:
            buy_quote = min(best_ask - 1, max(best_bid + 1, int(fair)))
            sell_quote = max(best_ask, int(fair + base_half_width + 1))
            bid_size = max(bid_size, 9)
            ask_size = 3

        # lower trend threshold so strategy leans more often
        if not high_vol:
            if trend > 0.6 and max_buy > 0:
                buy_quote = min(best_ask - 1, max(buy_quote, int(fair)))
            elif trend < -0.6 and max_sell > 0:
                sell_quote = max(best_bid + 1, min(sell_quote, int(fair)))

        if max_buy > 0:
            orders.append(Order(product, buy_quote, min(bid_size, max_buy)))
        if max_sell > 0:
            orders.append(Order(product, sell_quote, -min(ask_size, max_sell)))

        return orders

    # -----------------------------
    # Main entry
    # -----------------------------
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if state.traderData:
            try:
                saved_data = json.loads(state.traderData)
            except Exception:
                saved_data = {}
        else:
            saved_data = {}

        new_data = {}

        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(
                    product, state, order_depth, saved_data, new_data
                )

            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(
                    product, state, order_depth, saved_data, new_data
                )

            else:
                result[product] = []

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
