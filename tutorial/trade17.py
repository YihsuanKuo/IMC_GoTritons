from tutorial.datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    FAIR_PRICE = {
        "EMERALDS": 10000,
    }

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)

        if abs_pos < 30:
            return 0.03
        elif abs_pos < 60:
            return 0.06
        else:
            return 0.10

    def get_order_size(self, position: int, limit: int) -> int:
        abs_pos = abs(position)

        if abs_pos < 20:
            return 16
        elif abs_pos < 50:
            return 10
        else:
            return 6

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            orders: List[Order] = []

            if product not in self.FAIR_PRICE:
                result[product] = orders
                continue

            order_depth: OrderDepth = state.order_depths[product]

            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            current_position = state.position.get(product, 0)
            position_limit = self.POSITION_LIMITS[product]
            fair_price = self.FAIR_PRICE[product]

            alpha = self.get_alpha(current_position)
            adjusted_fair = fair_price - alpha * current_position

            max_buy = position_limit - current_position
            max_sell = position_limit + current_position

            base_size = self.get_order_size(current_position, position_limit)

            # Take liquidity when the top of book is already favorable.
            if max_buy > 0 and best_ask <= adjusted_fair + 1:
                ask_volume = abs(order_depth.sell_orders[best_ask])
                take_buy_qty = min(max_buy, max(base_size, ask_volume))
                if take_buy_qty > 0:
                    orders.append(Order(product, best_ask, take_buy_qty))
                    current_position += take_buy_qty
                    max_buy = position_limit - current_position
                    max_sell = position_limit + current_position

            if max_sell > 0 and best_bid >= adjusted_fair - 1:
                bid_volume = abs(order_depth.buy_orders[best_bid])
                take_sell_qty = min(max_sell, max(base_size, bid_volume))
                if take_sell_qty > 0:
                    orders.append(Order(product, best_bid, -take_sell_qty))
                    current_position -= take_sell_qty
                    max_buy = position_limit - current_position
                    max_sell = position_limit + current_position

            alpha = self.get_alpha(current_position)
            adjusted_fair = fair_price - alpha * current_position
            base_size = self.get_order_size(current_position, position_limit)

            spread = best_ask - best_bid
            if spread >= 3:
                my_bid = min(best_bid + 1, int(adjusted_fair))
                my_ask = max(best_ask - 1, int(adjusted_fair))
            else:
                my_bid = min(best_ask - 1, int(adjusted_fair))
                my_ask = max(best_bid + 1, int(adjusted_fair))

            if my_bid >= my_ask:
                my_bid = best_bid
                my_ask = best_ask

            buy_qty = min(base_size, max_buy)
            sell_qty = min(base_size, max_sell)

            if current_position > 65:
                buy_qty = 0
                sell_qty = min(max_sell, base_size + 8)
            elif current_position < -65:
                buy_qty = min(max_buy, base_size + 8)
                sell_qty = 0
            elif current_position > 35:
                buy_qty = min(max_buy, max(1, base_size // 2))
                sell_qty = min(max_sell, base_size + 4)
            elif current_position < -35:
                buy_qty = min(max_buy, base_size + 4)
                sell_qty = min(max_sell, max(1, base_size // 2))
            else:
                buy_qty = min(max_buy, base_size + 2)
                sell_qty = min(max_sell, base_size + 2)

            if buy_qty > 0:
                orders.append(Order(product, my_bid, buy_qty))

            if sell_qty > 0:
                orders.append(Order(product, my_ask, -sell_qty))

            result[product] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData
