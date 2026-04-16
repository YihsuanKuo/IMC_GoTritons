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
            if product == "INTARIAN_PEPPER_ROOT":
                if current_position < limit and max_buy > 0:
                    remaining_to_buy = max_buy
                    
                    # 1. AGGRESSIVE BUYING (with a price cap)
                    # Only buy asks that are at or just 1 tick above the best_ask
                    acceptable_ask = best_ask + 1 
                    
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if remaining_to_buy <= 0:
                            break
                        # Stop buying if the order book gets too expensive
                        if ask_price > acceptable_ask:
                            break
                            
                        vol = min(remaining_to_buy, -order_depth.sell_orders[ask_price])
                        orders.append(Order(product, ask_price, vol))
                        remaining_to_buy -= vol

                    # 2. PASSIVE BUYING
                    # If we haven't reached our limit of 80, place a passive bid
                    # to try and get filled at a cheaper price
                    if remaining_to_buy > 0:
                        # Place a bid 1 tick below the best ask (or at the best bid)
                        passive_bid_price = best_ask - 1
                        orders.append(Order(product, passive_bid_price, remaining_to_buy))

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
