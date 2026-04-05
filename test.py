from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import numpy as np


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def __init__(self):
        self.price_history = {"EMERALDS": [], "TOMATOES": []}

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
        gmma = 0.5

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
            new_data[product] = mid_price

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            alpha = self.get_alpha(current_position)

            max_buy = limit - current_position
            max_sell = limit + current_position

            if product == "EMERALDS":
                self.price_history["EMERALDS"].append(mid_price)
                volatility = np.mean(np.diff(self.price_history["EMERALDS"]))
                base_width = alpha * current_position
                width = base_width + gmma * volatility

                spread = best_ask - best_bid

                if spread >= 2:
                    buy_quote = min(best_bid + 1, int(mid_price - width))
                    sell_quote = max(best_ask - 1, int(mid_price + width))
                else:
                    buy_quote = best_bid
                    sell_quote = best_ask

                if max_buy > 0:
                    orders.append(Order(product, buy_quote, min(15, max_buy)))

                if max_sell > 0:
                    orders.append(Order(product, sell_quote, -min(15, max_sell)))

            elif product == "TOMATOES":
                self.price_history["TOMATOES"].append(mid_price)
                volatility = np.mean(np.diff(self.price_history["TOMATOES"]))
                base_width = alpha * current_position
                width = base_width + gmma * volatility

                ask_price = mid_price + width
                bid_price = mid_price - width
                edge = 0

                if best_ask <= ask_price - edge:
                    buy_volume = min(-best_ask_volume, max_buy)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if best_bid >= bid_price + edge:
                    sell_volume = min(best_bid_volume, max_sell)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                passive_buy = min(best_bid + 1, int(bid_price))
                passive_sell = max(best_ask - 1, int(ask_price))

                if max_buy > 0:
                    orders.append(Order(product, passive_buy, min(8, max_buy)))

                if max_sell > 0:
                    orders.append(Order(product, passive_sell, -min(8, max_sell)))

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
