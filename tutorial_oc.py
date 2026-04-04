from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)
        if abs_pos < 20:
            return 0.05
        elif abs_pos < 50:
            return 0.10
        else:
            return 0.20

    def get_order_size(self, position: int) -> int:
        abs_pos = abs(position)
        if abs_pos < 20:
            return 10
        elif abs_pos < 50:
            return 6
        else:
            return 3

    def get_half_spread(self, best_bid: int, best_ask: int, position: int) -> int:
        market_spread = best_ask - best_bid

        # 基于当前市场 spread 的基础值
        base_half_spread = max(1, market_spread // 4)

        # 根据仓位风险做调整
        abs_pos = abs(position)
        if abs_pos < 20:
            return base_half_spread
        elif abs_pos < 50:
            return base_half_spread + 1
        else:
            return base_half_spread + 2

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            orders: List[Order] = []

            if product != "EMERALDS":
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

            # 1. 定 fair
            fair_price = 10000

            # 2. inventory-aware fair
            alpha = self.get_alpha(current_position)
            adjusted_fair = fair_price - alpha * current_position

            # 3. 动态 spread
            half_spread = self.get_half_spread(best_bid, best_ask, current_position)

            raw_bid = int(adjusted_fair - half_spread)
            raw_ask = int(adjusted_fair + half_spread)

            # 4. 不要挂穿盘口
            my_bid = min(raw_bid, best_ask - 1)
            my_ask = max(raw_ask, best_bid + 1)

            if my_bid >= my_ask:
                my_bid = best_bid
                my_ask = best_ask

            # 5. 下单量
            base_size = self.get_order_size(current_position)

            max_buy = position_limit - current_position
            max_sell = position_limit + current_position

            buy_qty = min(base_size, max_buy)
            sell_qty = min(base_size, max_sell)

            # 6. 仓位控制
            if current_position > 80:
                if sell_qty > 0:
                    orders.append(Order(product, my_ask, -sell_qty))

            elif current_position < -80:
                if buy_qty > 0:
                    orders.append(Order(product, my_bid, buy_qty))

            else:
                if current_position > 20:
                    buy_qty = min(max(1, buy_qty // 2), max_buy)
                    sell_qty = min(base_size + 2, max_sell)
                elif current_position < -20:
                    buy_qty = min(base_size + 2, max_buy)
                    sell_qty = min(max(1, sell_qty // 2), max_sell)

                if buy_qty > 0:
                    orders.append(Order(product, my_bid, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, my_ask, -sell_qty))

            result[product] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData