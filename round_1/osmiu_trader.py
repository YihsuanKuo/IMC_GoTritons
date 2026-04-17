from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    def __init__(self):
        self.position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.price_history: List[float] = []

    def send_sell_order(self, orders: List[Order], product: str, price: int, amount: int):
        orders.append(Order(product, price, amount))

    def send_buy_order(self, orders: List[Order], product: str, price: int, amount: int):
        orders.append(Order(product, int(price), amount))

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def search_buys(
        self,
        state: TradingState,
        orders: List[Order],
        product: str,
        acceptable_price: float,
        depth: int = 1,
    ):
        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) != 0:
            book = list(order_depth.sell_orders.items())
            for ask, amount in book[0:min(len(book), depth)]:
                pos = self.get_product_pos(state, product)

                if int(ask) < acceptable_price or (
                    abs(ask - acceptable_price) < 1
                    and (pos < 0 and abs(pos - amount) < abs(pos))
                ):
                    size = min(
                        self.POSITION_LIMIT - self.position - self.buy_orders_sent,
                        -amount,
                    )
                    if size > 0:
                        self.buy_orders_sent += size
                        self.send_buy_order(orders, product, ask, size)

    def search_sells(
        self,
        state: TradingState,
        orders: List[Order],
        product: str,
        acceptable_price: float,
        depth: int = 1,
    ):
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) != 0:
            book = list(order_depth.buy_orders.items())
            for bid, amount in book[0:min(len(book), depth)]:
                pos = self.get_product_pos(state, product)

                if int(bid) > acceptable_price or (
                    abs(bid - acceptable_price) < 1
                    and (pos > 0 and abs(pos - amount) < abs(pos))
                ):
                    size = min(
                        self.position + self.POSITION_LIMIT - self.sell_orders_sent,
                        amount,
                    )
                    if size > 0:
                        self.sell_orders_sent += size
                        self.send_sell_order(orders, product, bid, -size)

    def get_bid(self, state: TradingState, product: str, price: int):
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) != 0:
            book = list(order_depth.buy_orders.items())
            for bid, _ in book:
                if bid < price:
                    return bid
        return None

    def get_ask(self, state: TradingState, product: str, price: int):
        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) != 0:
            book = list(order_depth.sell_orders.items())
            for ask, _ in book:
                if ask > price:
                    return ask
        return None

    def trade_osmium(self, state: TradingState, orders: List[Order]):
        product = self.PRODUCT
        order_book: OrderDepth = state.order_depths[product]
        sell_orders = order_book.sell_orders
        buy_orders = order_book.buy_orders

        if not sell_orders or not buy_orders:
            return

        best_ask = min(sell_orders.keys())
        best_ask_volume = sell_orders[best_ask]  # negative in the book
        best_bid = max(buy_orders.keys())
        best_bid_volume = buy_orders[best_bid]   # positive in the book
        mid_price = (best_ask + best_bid) / 2

        # Record current mid price in running history
        self.price_history.append(mid_price)

        # Need at least a few observations before trading on percentiles
        if len(self.price_history) < 10:
            return

        sorted_prices = sorted(self.price_history)
        n = len(sorted_prices)
        p25 = sorted_prices[int(n * 0.25)]
        p75 = sorted_prices[int(n * 0.75)]

        max_buy = self.POSITION_LIMIT - self.position
        max_sell = self.position + self.POSITION_LIMIT

        # Price below 25th percentile → buy
        if mid_price < p25:
            # Aggressive: take best ask
            aggressive_vol = min(-best_ask_volume, max_buy)
            if aggressive_vol > 0:
                orders.append(Order(product, best_ask, aggressive_vol))
                max_buy -= aggressive_vol
            # Passive: quote one tick inside the spread
            passive_vol = min(8, max_buy)
            if passive_vol > 0:
                orders.append(Order(product, best_bid + 1, passive_vol))

        # Price above 75th percentile → sell
        if mid_price > p75:
            # Aggressive: hit best bid
            aggressive_vol = min(best_bid_volume, max_sell)
            if aggressive_vol > 0:
                orders.append(Order(product, best_bid, -aggressive_vol))
                max_sell -= aggressive_vol
            # Passive: quote one tick inside the spread
            passive_vol = min(8, max_sell)
            if passive_vol > 0:
                orders.append(Order(product, best_ask - 1, -passive_vol))

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # Restore price history from previous ticks
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self.price_history = saved.get("price_history", [])
            except Exception:
                self.price_history = []

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            if (
                len(order_depth.buy_orders) == 0
                or len(order_depth.sell_orders) == 0
            ):
                result[product] = orders
                continue

            if product == self.PRODUCT:
                self.position = self.get_product_pos(state, product)
                self.buy_orders_sent = 0
                self.sell_orders_sent = 0
                self.trade_osmium(state, orders)

            result[product] = orders

        trader_data = json.dumps({"price_history": self.price_history})
        return result, 0, trader_data