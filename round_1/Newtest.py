from datamodel import OrderDepth, TradingState, Order
import json
import math


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    HISTORY_LENGTH = 3
    ENTRY_Z = 1.0
    EXIT_Z = 0.4

    BASE_QUOTE_OFFSET = 1
    INVENTORY_SKEW = 0.02
    STD_FLOOR = 0.5

    def __init__(self):
        self.position = 0
        self.mid_history = []

    def send_sell_order(self, orders, product, price, amount):
        if amount < 0:
            orders.append(Order(product, int(price), int(amount)))

    def send_buy_order(self, orders, product, price, amount):
        if amount > 0:
            orders.append(Order(product, int(price), int(amount)))

    def get_product_pos(self, state, product):
        return state.position.get(product, 0)

    def load_history(self, state):
        if self.mid_history:
            return
        if not state.traderData:
            return
        try:
            saved = json.loads(state.traderData)
            self.mid_history = saved.get("mid_history", [])
        except:
            self.mid_history = []

    def save_history(self):
        return json.dumps({
            "mid_history": self.mid_history[-self.HISTORY_LENGTH:]
        })

    def get_mid_price(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def update_mid_history(self, mid):
        self.mid_history.append(mid)
        if len(self.mid_history) > self.HISTORY_LENGTH:
            self.mid_history.pop(0)

    def get_mean_std(self):
        if not self.mid_history:
            return None, None
        mean = sum(self.mid_history) / len(self.mid_history)
        var = sum((x - mean) ** 2 for x in self.mid_history) / len(self.mid_history)
        return mean, math.sqrt(var)

    def get_zscore(self, mid):
        mean, std = self.get_mean_std()
        if mean is None:
            return 0, mid
        std = max(std, self.STD_FLOOR)
        return (mid - mean) / std, mean

    def trade(self, state, orders):
        od = state.order_depths[self.PRODUCT]

        if not od.buy_orders or not od.sell_orders:
            return

        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_mid_history(mid)
        z, mean = self.get_zscore(mid)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        current_pos = self.position
        max_buy = self.POSITION_LIMIT - current_pos
        max_sell = self.POSITION_LIMIT + current_pos

        # 1. EXIT LOGIC: z-score 回到均值附近时主动平仓
        if abs(z) < self.EXIT_Z:
            if current_pos > 0:
                self.send_sell_order(orders, self.PRODUCT, best_bid, -current_pos)
            elif current_pos < 0:
                self.send_buy_order(orders, self.PRODUCT, best_ask, -current_pos)
            return

        # 2. MEAN REVERSION
        if z < -self.ENTRY_Z:
            # 价格低于均值太多，买入
            for ask, vol in sorted(od.sell_orders.items()):
                if ask <= mean and max_buy > 0:
                    available = -vol
                    size = min(available, max_buy)
                    if size > 0:
                        self.send_buy_order(orders, self.PRODUCT, ask, size)
                        max_buy -= size

        elif z > self.ENTRY_Z:
            # 价格高于均值太多，卖出
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid >= mean and max_sell > 0:
                    available = vol
                    size = min(available, max_sell)
                    if size > 0:
                        self.send_sell_order(orders, self.PRODUCT, bid, -size)
                        max_sell -= size

        # 3. MARKET MAKING WITH INVENTORY SKEW
        fair = mean - self.INVENTORY_SKEW * current_pos

        bid_price = min(best_bid + 1, int(fair - self.BASE_QUOTE_OFFSET))
        ask_price = max(best_ask - 1, int(fair + self.BASE_QUOTE_OFFSET))

        # 防止报价交叉
        if bid_price >= ask_price:
            bid_price = best_bid
            ask_price = best_ask

        # 仓位越大，做市单量越小
        mm_size = max(1, 10 - abs(current_pos) // 10)

        remaining_buy = self.POSITION_LIMIT - current_pos
        remaining_sell = self.POSITION_LIMIT + current_pos

        buy_size = min(mm_size, remaining_buy)
        sell_size = min(mm_size, remaining_sell)

        if buy_size > 0:
            self.send_buy_order(orders, self.PRODUCT, bid_price, buy_size)

        if sell_size > 0:
            self.send_sell_order(orders, self.PRODUCT, ask_price, -sell_size)

    def run(self, state: TradingState):
        result = {self.PRODUCT: []}

        self.position = self.get_product_pos(state, self.PRODUCT)
        self.load_history(state)

        if self.PRODUCT in state.order_depths:
            self.trade(state, result[self.PRODUCT])

        return result, 0, self.save_history()