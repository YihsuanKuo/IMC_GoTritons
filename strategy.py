from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def __init__(self, lam=[0.8, 0.8], alpha=[0.1, 0.05]):
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
            new_data[product] = mid_price

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            alpha = self.get_alpha(current_position)

            max_buy = limit - current_position
            max_sell = limit + current_position

            # ---------------- EMERALDS ----------------
            if product == "EMERALDS":
                prev_mid = saved_data.get("EMERALDS", mid_price)

                fair_price = (
                    self.er_lam * prev_mid + (1 - self.er_lam) * mid_price - self.er_alpha * current_position
                )

                spread = best_ask - best_bid

                # step inside the spread to improve fill probability
                if spread >= 2:
                    buy_quote = min(best_bid + 1, int(fair_price))
                    sell_quote = max(best_ask - 1, int(fair_price))
                else:
                    buy_quote = best_bid
                    sell_quote = best_ask

                if max_buy > 0:
                    orders.append(Order(product, buy_quote, min(15, max_buy)))

                if max_sell > 0:
                    orders.append(
                        Order(product, sell_quote, -min(15, max_sell))
                    )

            # ---------------- TOMATOES ----------------
            elif product == "TOMATOES":
                prev_mid = saved_data.get("TOMATOES", mid_price)

                fair_price = (
                    self.tom_lam * prev_mid + (1 - self.tom_lam) * mid_price - self.tom_alpha * current_position
                )

                # smaller threshold => more aggressive trading
                edge = 0

                # aggressive order taking
                if best_ask <= fair_price - edge:
                    buy_volume = min(-best_ask_volume, max_buy)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if best_bid >= fair_price + edge:
                    sell_volume = min(best_bid_volume, max_sell)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # tighter passive quotes
                passive_buy = min(best_bid + 1, int(fair_price))
                passive_sell = max(best_ask - 1, int(fair_price))

                if max_buy > 0:
                    orders.append(Order(product, passive_buy, min(8, max_buy)))

                if max_sell > 0:
                    orders.append(
                        Order(product, passive_sell, -min(8, max_sell))
                    )

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
