from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 20,
        "TOMATOES": 20,
    }

    def bid(self):
        return 15

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)
        if abs_pos < 20:
            return 0.05
        elif abs_pos < 50:
            return 0.1
        else:
            return 0.2

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
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

            current_position = state.position.get(product, 0)
            position_limit = self.POSITION_LIMITS.get(product, 20)

            if product == "EMERALDS":
                fair_old = 10000
            else:
                result[product] = orders
                continue

            alpha = self.get_alpha(current_position)
            fair_new = fair_old - alpha * current_position

            edge = 1

            if best_ask < fair_new - edge:
                max_buy = position_limit - current_position
                buy_volume = min(-best_ask_volume, max_buy)
                if buy_volume > 0:
                    orders.append(Order(product, best_ask, buy_volume))

            if best_bid > fair_new + edge:
                max_sell = position_limit + current_position
                sell_volume = min(best_bid_volume, max_sell)
                if sell_volume > 0:
                    orders.append(Order(product, best_bid, -sell_volume))

            result[product] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData
