from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # -----------------------------
    # Helpers
    # -----------------------------
    def get_best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = -order_depth.sell_orders[best_ask]
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

    # -----------------------------
    # Simple microprice strategy
    # -----------------------------
    def trade_product(
        self,
        product: str,
        state: TradingState,
        order_depth: OrderDepth,
        anchor: float = None,
    ) -> List[Order]:
        orders: List[Order] = []

        info = self.get_best_bid_ask(order_depth)
        if info is None:
            return orders

        best_bid, best_ask, best_bid_vol, best_ask_vol = info
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        micro = self.calc_microprice(
            best_bid, best_ask, best_bid_vol, best_ask_vol
        )
        imbalance = self.calc_book_imbalance(order_depth, depth=3)

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        max_buy = limit - pos
        max_sell = limit + pos

        # -----------------------------
        # Fair price
        # -----------------------------
        if anchor is not None:
            # anchored product like EMERALDS
            fair = 0.70 * anchor + 0.30 * micro
        else:
            # freer product like TOMATOES
            fair = 0.80 * mid + 0.20 * micro

        # light inventory penalty
        fair -= 4.0 * (pos / limit)

        micro_edge = micro - mid

        # -----------------------------
        # Regimes from micro edge
        # -----------------------------
        strong_up = micro_edge > 0.35
        strong_down = micro_edge < -0.35

        # base width
        half_width = 1
        if spread >= 3:
            half_width = 2

        # inventory bands
        very_long = pos > 0.65 * limit
        very_short = pos < -0.65 * limit

        # -----------------------------
        # Inventory override
        # -----------------------------
        if very_long:
            # flatten long inventory
            buy_quote = min(best_bid, int(fair - 2))
            sell_quote = max(best_bid + 1, min(best_ask - 1, int(fair)))
            bid_size = 2
            ask_size = 14

        elif very_short:
            # flatten short inventory
            buy_quote = min(best_ask - 1, max(best_bid + 1, int(fair)))
            sell_quote = max(best_ask, int(fair + 2))
            bid_size = 14
            ask_size = 2

        # -----------------------------
        # Strong upward pressure
        # -----------------------------
        elif strong_up:
            buy_quote = min(best_ask, int(fair))
            sell_quote = max(best_ask, int(fair + half_width + 1))
            bid_size = 10
            ask_size = 5

            # optional small aggressive buy
            if max_buy > 0 and best_ask <= fair + 1:
                take_qty = min(best_ask_vol, max_buy, 6)
                if take_qty > 0:
                    orders.append(Order(product, best_ask, take_qty))

        # -----------------------------
        # Strong downward pressure
        # -----------------------------
        elif strong_down:
            buy_quote = min(best_bid, int(fair - half_width - 1))
            sell_quote = max(best_bid, int(fair))
            bid_size = 5
            ask_size = 10

            # optional small aggressive sell
            if max_sell > 0 and best_bid >= fair - 1:
                take_qty = min(best_bid_vol, max_sell, 6)
                if take_qty > 0:
                    orders.append(Order(product, best_bid, -take_qty))

        # -----------------------------
        # Neutral pressure
        # -----------------------------
        else:
            buy_quote = min(best_bid + 1, int(fair - half_width))
            sell_quote = max(best_ask - 1, int(fair + half_width))
            bid_size = 8
            ask_size = 8

        if max_buy > 0:
            orders.append(Order(product, buy_quote, min(bid_size, max_buy)))

        if max_sell > 0:
            orders.append(Order(product, sell_quote, -min(ask_size, max_sell)))

        return orders

    # -----------------------------
    # Main
    # -----------------------------
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            if product == "EMERALDS":
                result[product] = self.trade_product(
                    product, state, order_depth, anchor=10000.0
                )

            elif product == "TOMATOES":
                result[product] = self.trade_product(
                    product, state, order_depth, anchor=None
                )

            else:
                result[product] = []

        traderData = json.dumps({})
        conversions = 0
        return result, conversions, traderData
