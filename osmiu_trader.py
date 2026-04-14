from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import math


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    def __init__(self):
        self.position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0

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
        low = -self.POSITION_LIMIT
        high = self.POSITION_LIMIT
        product = self.PRODUCT

        position = state.position.get(product, 0)

        order_book: OrderDepth = state.order_depths[product]
        sell_orders = order_book.sell_orders
        buy_orders = order_book.buy_orders

        if len(sell_orders) != 0 and len(buy_orders) != 0:
            best_ask = min(sell_orders.keys())
            best_bid = max(buy_orders.keys())

            fair_price = int(math.ceil((best_ask + best_bid) / 2))
            decimal_fair_price = (best_ask + best_bid) / 2

            # TAKE good prices
            self.search_buys(state, orders, product, decimal_fair_price, depth=3)
            self.search_sells(state, orders, product, decimal_fair_price, depth=3)

            # FIND passive quote prices
            best_ask_above_fair = self.get_ask(state, product, fair_price)
            best_bid_below_fair = self.get_bid(state, product, fair_price)

            buy_price = math.floor(decimal_fair_price) - 17
            sell_price = math.ceil(decimal_fair_price) + 17

            if best_ask_above_fair is not None and best_bid_below_fair is not None:
                if best_ask_above_fair - 1 > decimal_fair_price:
                    sell_price = best_ask_above_fair - 1
                if best_bid_below_fair + 1 < decimal_fair_price:
                    buy_price = best_bid_below_fair + 1

            max_buy = high - self.position - self.buy_orders_sent
            max_sell = self.position - low - self.sell_orders_sent

            pos = self.get_product_pos(state, product)

            if max_buy > 0 and not (pos > 0 and float(buy_price) == decimal_fair_price):
                self.send_buy_order(
                    orders,
                    product,
                    buy_price,
                    max_buy,
                )

            if max_sell > 0 and not (pos < 0 and float(sell_price) == decimal_fair_price):
                self.send_sell_order(
                    orders,
                    product,
                    sell_price,
                    -max_sell,
                )

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {self.PRODUCT: []}

        self.position = self.get_product_pos(state, self.PRODUCT)
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0

        if self.PRODUCT in state.order_depths:
            self.trade_osmium(state, result[self.PRODUCT])

        return result, 0, ""
