from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 60

    HISTORY_LENGTH = 3
    ENTRY_Z = 0.6
    EXIT_Z = 0.4

    BASE_QUOTE_OFFSET = 1
    INVENTORY_SKEW = 0.02

    def __init__(self):
        self.position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.mid_history = []

    def send_sell_order(self, orders, product, price, amount):
        orders.append(Order(product, int(price), int(amount)))

    def send_buy_order(self, orders, product, price, amount):
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
        return json.dumps({"mid_history": self.mid_history[-self.HISTORY_LENGTH:]})

    def get_mid_price(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

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
        if mean is None or std < 1e-6:
            return 0, mid
        return (mid - mean) / std, mean

    def trade(self, state, orders):
        od = state.order_depths[self.PRODUCT]
        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_mid_history(mid)
        z, mean = self.get_zscore(mid)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        # mean reversion
        if z < -self.ENTRY_Z:
            for ask, vol in sorted(od.sell_orders.items()):
                if ask <= mean:
                    size = min(-vol, self.POSITION_LIMIT)
                    self.send_buy_order(orders, self.PRODUCT, ask, size)

        elif z > self.ENTRY_Z:
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid >= mean:
                    size = min(vol, self.POSITION_LIMIT)
                    self.send_sell_order(orders, self.PRODUCT, bid, -size)

        # market making
        fair = mean
        bid_price = min(best_bid + 1, int(fair - self.BASE_QUOTE_OFFSET))
        ask_price = max(best_ask - 1, int(fair + self.BASE_QUOTE_OFFSET))

        self.send_buy_order(orders, self.PRODUCT, bid_price, 10)
        self.send_sell_order(orders, self.PRODUCT, ask_price, -10)

    def run(self, state: TradingState):
        result = {self.PRODUCT: []}

        self.position = self.get_product_pos(state, self.PRODUCT)
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0

        self.load_history(state)

        if self.PRODUCT in state.order_depths:
            self.trade(state, result[self.PRODUCT])

        return result, 0, self.save_history()