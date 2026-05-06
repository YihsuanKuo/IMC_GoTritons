<<<<<<< HEAD:backtester_project/strategies/strategy.py
from datamodel import OrderDepth, TradingState, Order
=======
from tutorial.datamodel import OrderDepth, TradingState, Order
>>>>>>> 7fecebb0fd61c4b965c21df7b5538cf029d3cb31:tutorial/test.py
from typing import Dict, List
import json
import math


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def __init__(self, lam=[0.6, 0.6], alpha=[None, None]):
        self.er_lam = lam[0]
        self.tom_lam = lam[1]
        self.er_alpha = alpha[0]
        self.tom_alpha = alpha[1]

    def bid(self):
        return 15

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)
        if abs_pos < 10:
            return 0.02
        elif abs_pos < 30:
            return 0.05
        else:
            return 0.10

    def get_quotes(self, best_bid: int, best_ask: int, fair: float):
        """
        Build passive quotes around fair value while preventing bid >= ask.
        """
        bid_quote = min(best_bid + 1, math.floor(fair))
        ask_quote = max(best_ask - 1, math.ceil(fair))

        if bid_quote >= ask_quote:
            bid_quote = best_bid
            ask_quote = best_ask

        return bid_quote, ask_quote

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
            orders: List[Order] = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]

            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = abs(order_depth.sell_orders[best_ask])

            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            max_buy = limit - current_position
            max_sell = limit + current_position

            # ---------------- EMERALDS ----------------
            if product == "EMERALDS":
                # EMERALDS is very stable; use a near-constant anchor
                base_fair = 10000.0

                # inventory penalty should be recomputed every step unless user fixed it
                alpha = (
                    self.er_alpha
                    if self.er_alpha is not None
                    else self.get_alpha(current_position)
                )

                fair_price = base_fair - alpha * current_position

                # aggressive taking only when clearly favorable
                if best_ask < fair_price and max_buy > 0:
                    buy_volume = min(best_ask_volume, max_buy, 12)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if best_bid > fair_price and max_sell > 0:
                    sell_volume = min(best_bid_volume, max_sell, 12)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # passive MM, but size should depend on inventory
                bid_quote, ask_quote = self.get_quotes(
                    best_bid, best_ask, fair_price
                )

                buy_size = min(12, max_buy)
                sell_size = min(12, max_sell)

                if current_position > 40:
                    buy_size = min(4, max_buy)
                    sell_size = min(16, max_sell)
                elif current_position < -40:
                    buy_size = min(16, max_buy)
                    sell_size = min(4, max_sell)

                if buy_size > 0:
                    orders.append(Order(product, bid_quote, buy_size))
                if sell_size > 0:
                    orders.append(Order(product, ask_quote, -sell_size))

                new_data[product] = {
                    "fair": fair_price,
                    "mid": mid_price,
                }

            # ---------------- TOMATOES ----------------
            elif product == "TOMATOES":
                prev = saved_data.get(product, {})
                prev_fair = prev.get("fair", mid_price)
                prev_mid = prev.get("mid", mid_price)

                alpha = (
                    self.tom_alpha
                    if self.tom_alpha is not None
                    else self.get_alpha(current_position)
                )

                # true EWMA fair value
                raw_fair = (
                    self.tom_lam * prev_fair + (1 - self.tom_lam) * mid_price
                )

                # simple trend estimate
                trend = mid_price - prev_mid

                # inventory-skewed fair
                fair_price = raw_fair - alpha * current_position

                # dynamic edge
                edge = max(1.0, 0.5 * spread)

                # ---------------- risk-off logic ----------------
                # if inventory is already large and trend is against it, flatten some
                if current_position >= 40 and trend < -1.0:
                    reduce_qty = min(12, current_position, best_bid_volume)
                    if reduce_qty > 0:
                        orders.append(Order(product, best_bid, -reduce_qty))
                    result[product] = orders
                    new_data[product] = {
                        "fair": raw_fair,
                        "mid": mid_price,
                    }
                    continue

                if current_position <= -40 and trend > 1.0:
                    reduce_qty = min(12, -current_position, best_ask_volume)
                    if reduce_qty > 0:
                        orders.append(Order(product, best_ask, reduce_qty))
                    result[product] = orders
                    new_data[product] = {
                        "fair": raw_fair,
                        "mid": mid_price,
                    }
                    continue

                # ---------------- aggressive taking ----------------
                # do not keep buying in a clear downtrend
                allow_buy = not (trend < -1.0 and current_position >= 0)
                allow_sell = not (trend > 1.0 and current_position <= 0)

                if allow_buy and best_ask <= fair_price - edge and max_buy > 0:
                    buy_volume = min(best_ask_volume, max_buy, 10)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if (
                    allow_sell
                    and best_bid >= fair_price + edge
                    and max_sell > 0
                ):
                    sell_volume = min(best_bid_volume, max_sell, 10)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # ---------------- passive quoting ----------------
                bid_quote, ask_quote = self.get_quotes(
                    best_bid, best_ask, fair_price
                )

                buy_size = min(8, max_buy)
                sell_size = min(8, max_sell)

                # inventory-aware sizing
                if current_position > 20:
                    buy_size = min(3, max_buy)
                    sell_size = min(12, max_sell)
                elif current_position < -20:
                    buy_size = min(12, max_buy)
                    sell_size = min(3, max_sell)

                # trend-aware quoting
                if trend < -1.0 and current_position >= 0:
                    buy_size = 0
                    sell_size = min(12, max_sell)
                elif trend > 1.0 and current_position <= 0:
                    sell_size = 0
                    buy_size = min(12, max_buy)

                if buy_size > 0:
                    orders.append(Order(product, bid_quote, buy_size))
                if sell_size > 0:
                    orders.append(Order(product, ask_quote, -sell_size))

                new_data[product] = {
                    "fair": raw_fair,
                    "mid": mid_price,
                }

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
