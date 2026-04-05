from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 20,
    }

    def bid(self):
        return 15

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)
        if abs_pos < 10:
            return 0.02
        elif abs_pos < 30:
            return 0.05
        else:
            return 0.1

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

            if (
                len(order_depth.buy_orders) == 0
                or len(order_depth.sell_orders) == 0
            ):
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]

            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]

            mid_price = (best_bid + best_ask) / 2

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            alpha = self.get_alpha(current_position)

            max_buy = limit - current_position
            max_sell = limit + current_position

            # ---------------- EMERALDS ----------------
            if product == "EMERALDS":
                fair_price = 10000 - alpha * current_position

                buy_quote = int(fair_price - 1)
                sell_quote = int(fair_price + 1)

                if max_buy > 0:
                    orders.append(Order(product, buy_quote, min(10, max_buy)))

                if max_sell > 0:
                    orders.append(
                        Order(product, sell_quote, -min(10, max_sell))
                    )

            # ---------------- TOMATOES ----------------
            elif product == "TOMATOES":
                prev_ema = saved_data.get("TOMATOES_ema", mid_price)
                prev_mid = saved_data.get("TOMATOES_prev_mid", mid_price)

                lam = 0.2
                ema_price = lam * mid_price + (1 - lam) * prev_ema

                volatility = abs(mid_price - prev_mid)

                fair_price = ema_price - alpha * current_position

                # stricter edge to improve trade quality
                edge = 1.0

                # volatility-adjusted width
                base_width = 1.0
                gamma = 0.3
                width = base_width + gamma * volatility

                # aggressive taking only when clearly favorable
                if best_ask < fair_price - edge and max_buy > 0:
                    buy_volume = min(-best_ask_volume, max_buy, 5)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if best_bid > fair_price + edge and max_sell > 0:
                    sell_volume = min(best_bid_volume, max_sell, 5)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # small passive quotes
                bid_quote = int(fair_price - width)
                ask_quote = int(fair_price + width)

                if max_buy > 0:
                    orders.append(Order(product, bid_quote, min(3, max_buy)))

                if max_sell > 0:
                    orders.append(Order(product, ask_quote, -min(3, max_sell)))

                new_data["TOMATOES_ema"] = ema_price
                new_data["TOMATOES_prev_mid"] = mid_price

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
