from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

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

            best_ask = min(order_depth.sell_orders.keys())

            mid_price = (best_bid + best_ask) / 2
            new_data[product] = mid_price

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            max_buy = limit - current_position

            # ---------------- INTARIAN_PEPPER_ROOT ----------------
            # Price always trends up linearly. Strategy: buy the full position limit (80)
            # as aggressively as possible on the first opportunity, then hold forever.
            if product == "INTARIAN_PEPPER_ROOT":
                if current_position < limit and max_buy > 0:
                    # Sweep ask levels greedily until position limit is reached
                    remaining = max_buy
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if remaining <= 0:
                            break
                        vol = min(remaining, -order_depth.sell_orders[ask_price])
                        orders.append(Order(product, ask_price, vol))
                        remaining -= vol
                # Never sell — no sell orders placed

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
