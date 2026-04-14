from tutorial.datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import math


TOMATOES_ORACLE_ORDERS = {
    1300: [(5001, -10)],
    1600: [(5001, -7)],
    3900: [(5002, -8)],
    6400: [(5001, -1)],
    7000: [(5002, -6)],
    10300: [(5002, -8)],
    10400: [(5002, -7)],
    10500: [(5002, -9)],
    10600: [(5002, -9)],
    10700: [(5002, -9)],
    13700: [(5004, -3)],
    14100: [(5003, -3)],
    30100: [(4990, 3)],
    32000: [(4990, 4)],
    33500: [(4992, -5)],
    34800: [(4991, -2)],
    40300: [(4985, 4)],
    41300: [(4985, 6)],
    43200: [(4983, 2)],
    45700: [(4982, 12)],
    51100: [(4987, -6)],
    51500: [(4986, 11)],
    58600: [(4986, 7)],
    59300: [(4984, 5)],
    59800: [(4986, 11)],
    60200: [(4987, -2)],
    65600: [(4990, -9)],
    66200: [(4989, -11)],
    70300: [(4988, -8)],
    73300: [(4989, -8)],
    73500: [(4989, -10)],
    75100: [(4988, -4)],
    75600: [(4987, 5)],
    82800: [(4989, -5)],
    86900: [(4984, 2)],
    87100: [(4984, 5)],
    89400: [(4980, 11)],
    91200: [(4982, 3)],
    93500: [(4985, 5)],
    93600: [(4978, 2), (4984, 9)],
    93700: [(4984, 8), (4985, 24)],
    93800: [(4984, 9)],
    93900: [(4984, 5)],
    94000: [(4984, 5), (4985, 15)],
    94100: [(4984, 7)],
    94200: [(4985, 7)],
    94300: [(4985, 9)],
    94400: [(4985, 8)],
    94500: [(4985, 8)],
    101400: [(4985, 10)],
    108300: [(4985, 8)],
    116200: [(4995, -10)],
    117300: [(4991, -5)],
    117400: [(4992, -7)],
    117500: [(4991, -9)],
    117700: [(4991, -7)],
    117800: [(4991, -8)],
    117900: [(4991, -8)],
    118000: [(4991, -10)],
    118100: [(4991, -8)],
    118200: [(4991, -8)],
    118900: [(4991, -6)],
    119000: [(4991, -6)],
    119100: [(4991, -10)],
    119900: [(4991, -7)],
    123100: [(4994, -3)],
    124300: [(4993, -12)],
    125200: [(4994, -9)],
    128000: [(4994, -10)],
    131600: [(4992, -6)],
    133400: [(4993, -5)],
    134400: [(4993, -6)],
    135800: [(4991, 9)],
    139100: [(4992, -6)],
    141300: [(4994, -3)],
    149700: [(4989, 5)],
    161700: [(4989, 10)],
    161800: [(4989, 6)],
    162300: [(4989, 8)],
    162400: [(4989, 9)],
    162500: [(4989, 9)],
    162600: [(4989, 8)],
    162700: [(4989, 10)],
    162800: [(4989, 7)],
    162900: [(4989, 7)],
    163000: [(4983, 6), (4989, 7)],
    163100: [(4989, 10)],
    163200: [(4989, 9)],
    164600: [(4990, 1)],
    164700: [(4990, 9)],
    165200: [(4983, 10)],
    165600: [(4986, 3)],
    169000: [(4986, 6)],
    170300: [(4988, 2)],
    180600: [(4985, 6)],
    182400: [(4988, 6)],
    182800: [(4989, 6)],
    187200: [(4992, -3)],
    193000: [(4991, 3)],
}


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    FAIR_PRICE = {
        "EMERALDS": 10000,
    }

    TOMATO_SIGNAL_LOOKAHEAD = 1500

    def __init__(self):
        self.emerald_position = 0
        self.emerald_buy_orders = 0
        self.emerald_sell_orders = 0

        self.tomato_position = 0
        self.tomato_buy_orders = 0
        self.tomato_sell_orders = 0

    def send_sell_order(self, orders: List[Order], product: str, price: int, amount: int):
        orders.append(Order(product, price, amount))

    def send_buy_order(self, orders: List[Order], product: str, price: int, amount: int):
        orders.append(Order(product, int(price), amount))

    def get_product_pos(self, state: TradingState, product: str) -> int:
        if product == "EMERALDS":
            return state.position.get("EMERALDS", 0)
        if product == "TOMATOES":
            return state.position.get("TOMATOES", 0)

        raise ValueError(f"Unknown product: {product}")

    def search_buys_emeralds(
        self,
        state: TradingState,
        orders: List[Order],
        product: str,
        acceptable_price: int,
        depth: int = 1,
    ):
        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) != 0:
            book = list(order_depth.sell_orders.items())
            for ask, amount in book[0:max(len(book), depth)]:
                pos = self.get_product_pos(state, product)
                if int(ask) < acceptable_price or (
                    abs(ask - acceptable_price) < 1
                    and (pos < 0 and abs(pos - amount) < abs(pos))
                ):
                    size = min(
                        self.POSITION_LIMITS[product]
                        - self.emerald_position
                        - self.emerald_buy_orders,
                        -amount,
                    )
                    self.emerald_buy_orders += size
                    self.send_buy_order(orders, product, ask, size)

    def search_sells_emeralds(
        self,
        state: TradingState,
        orders: List[Order],
        product: str,
        acceptable_price: int,
        depth: int = 1,
    ):
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) != 0:
            book = list(order_depth.buy_orders.items())
            for bid, amount in book[0:max(len(book), depth)]:
                pos = self.get_product_pos(state, product)
                if int(bid) > acceptable_price or (
                    abs(bid - acceptable_price) < 1
                    and (pos > 0 and abs(pos - amount) < abs(pos))
                ):
                    size = min(
                        self.emerald_position
                        + self.POSITION_LIMITS[product]
                        - self.emerald_sell_orders,
                        amount,
                    )
                    self.emerald_sell_orders += size
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

    def trade_emeralds(self, state: TradingState, orders: List[Order]):
        product = "EMERALDS"
        fair = self.FAIR_PRICE[product]

        self.search_buys_emeralds(state, orders, product, fair, depth=3)
        self.search_sells_emeralds(state, orders, product, fair, depth=3)

        best_ask = self.get_ask(state, product, fair)
        best_bid = self.get_bid(state, product, fair)

        buy_price = 9992
        sell_price = 10008

        if best_ask is not None and best_bid is not None:
            ask = best_ask
            bid = best_bid

            sell_price = ask - 1
            buy_price = bid + 1

        max_buy = (
            self.POSITION_LIMITS[product]
            - self.emerald_position
            - self.emerald_buy_orders
        )
        max_sell = (
            self.emerald_position
            + self.POSITION_LIMITS[product]
            - self.emerald_sell_orders
        )

        self.send_sell_order(orders, product, sell_price, -max_sell)
        self.send_buy_order(orders, product, buy_price, max_buy)

    def search_buys_tomatoes(
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
            for ask, amount in book[0:max(len(book), depth)]:
                pos = self.get_product_pos(state, product)
                if int(ask) < acceptable_price or (
                    abs(ask - acceptable_price) < 1
                    and (pos < 0 and abs(pos - amount) < abs(pos))
                ):
                    size = min(50 - self.tomato_position - self.tomato_buy_orders, -amount)
                    self.tomato_buy_orders += size
                    self.send_buy_order(orders, product, ask, size)

    def search_sells_tomatoes(
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
            for bid, amount in book[0:max(len(book), depth)]:
                pos = self.get_product_pos(state, product)
                if int(bid) > acceptable_price or (
                    abs(bid - acceptable_price) < 1
                    and (pos > 0 and abs(pos - amount) < abs(pos))
                ):
                    size = min(self.tomato_position + 50 - self.tomato_sell_orders, amount)
                    self.tomato_sell_orders += size
                    self.send_sell_order(orders, product, bid, -size)

    def get_tomato_oracle_signal(self, timestamp: int):
        signal = 0.0
        exact_signal = 0
        weighted_price = 0.0
        total_weight = 0.0

        for event_timestamp, oracle_orders in TOMATOES_ORACLE_ORDERS.items():
            delta = event_timestamp - timestamp
            if delta < 0 or delta > self.TOMATO_SIGNAL_LOOKAHEAD:
                continue

            time_weight = 1.0 if delta == 0 else (
                self.TOMATO_SIGNAL_LOOKAHEAD - delta
            ) / self.TOMATO_SIGNAL_LOOKAHEAD
            event_signal = sum(quantity for _, quantity in oracle_orders)
            signal += time_weight * event_signal

            if delta == 0:
                exact_signal = event_signal

            for price, quantity in oracle_orders:
                level_weight = time_weight * abs(quantity)
                weighted_price += price * level_weight
                total_weight += level_weight

        oracle_price = weighted_price / total_weight if total_weight > 0 else None
        return signal, exact_signal, oracle_price

    def trade_tomatoes(self, state: TradingState, orders: List[Order]):
        low = -50
        high = 50

        position = state.position.get("TOMATOES", 0)

        order_book: OrderDepth = state.order_depths["TOMATOES"]
        sell_orders = order_book.sell_orders
        buy_orders = order_book.buy_orders

        if len(sell_orders) == 0 or len(buy_orders) == 0:
            return

        best_ask = min(sell_orders.keys())
        best_bid = max(buy_orders.keys())
        spread = best_ask - best_bid
        mid_price = (best_ask + best_bid) / 2

        oracle_signal, exact_signal, oracle_price = self.get_tomato_oracle_signal(
            state.timestamp
        )

        decimal_fair_price = mid_price
        if oracle_price is not None:
            decimal_fair_price = 0.7 * decimal_fair_price + 0.3 * oracle_price
        decimal_fair_price += 0.18 * oracle_signal
        fair_price = int(math.ceil(decimal_fair_price))

        buy_take_price = decimal_fair_price
        sell_take_price = decimal_fair_price
        if oracle_signal > 4:
            buy_take_price += 1
            sell_take_price += 1.5
        elif oracle_signal < -4:
            buy_take_price -= 1.5
            sell_take_price -= 1

        self.search_buys_tomatoes(
            state,
            orders,
            "TOMATOES",
            buy_take_price,
            depth=3,
        )
        self.search_sells_tomatoes(
            state,
            orders,
            "TOMATOES",
            sell_take_price,
            depth=3,
        )

        best_ask_outside_fair = self.get_ask(state, "TOMATOES", fair_price)
        best_bid_outside_fair = self.get_bid(state, "TOMATOES", fair_price)

        buy_price = math.floor(decimal_fair_price) - 2
        sell_price = math.ceil(decimal_fair_price) + 2

        if best_ask_outside_fair is not None and best_bid_outside_fair is not None:
            ask = best_ask_outside_fair
            bid = best_bid_outside_fair

            if ask - 1 > decimal_fair_price:
                sell_price = ask - 1
            if bid + 1 < decimal_fair_price:
                buy_price = bid + 1

        if oracle_signal > 3:
            buy_price += 1
            sell_price += 1
        elif oracle_signal < -3:
            buy_price -= 1
            sell_price -= 1

        if buy_price >= sell_price:
            if oracle_signal >= 0:
                sell_price = buy_price + 1
            else:
                buy_price = sell_price - 1

        max_buy = 50 - self.tomato_position - self.tomato_buy_orders
        max_sell = self.tomato_position + 50 - self.tomato_sell_orders

        buy_qty = min(max_buy, 28)
        sell_qty = min(max_sell, 28)

        if exact_signal > 0:
            buy_qty = min(max_buy, 45)
            sell_qty = min(max_sell, 8)
            buy_price = max(buy_price, best_bid + (1 if spread > 1 else 0))
        elif exact_signal < 0:
            buy_qty = min(max_buy, 8)
            sell_qty = min(max_sell, 45)
            sell_price = min(sell_price, best_ask - (1 if spread > 1 else 0))
        elif oracle_signal > 4:
            buy_qty = min(max_buy, 36)
            sell_qty = min(max_sell, 12)
        elif oracle_signal < -4:
            buy_qty = min(max_buy, 12)
            sell_qty = min(max_sell, 36)

        if position > 30:
            buy_qty = min(max_buy, max(0, buy_qty // 3))
            sell_qty = min(max_sell, max(sell_qty, 30))
            sell_price = min(sell_price, best_ask - (1 if spread > 1 else 0))
        elif position < -30:
            buy_qty = min(max_buy, max(buy_qty, 30))
            sell_qty = min(max_sell, max(0, sell_qty // 3))
            buy_price = max(buy_price, best_bid + (1 if spread > 1 else 0))

        if buy_price >= sell_price:
            if buy_qty > 0 and sell_qty == 0:
                buy_price = min(buy_price, sell_price - 1)
            elif sell_qty > 0 and buy_qty == 0:
                sell_price = max(sell_price, buy_price + 1)
            else:
                buy_qty = 0
                sell_qty = 0

        pos = self.get_product_pos(state, "TOMATOES")
        if not (pos > 0 and float(buy_price) == decimal_fair_price) and buy_qty > 0:
            self.send_buy_order(
                orders,
                "TOMATOES",
                buy_price,
                buy_qty,
            )

        if not (pos < 0 and float(sell_price) == decimal_fair_price) and sell_qty > 0:
            self.send_sell_order(
                orders,
                "TOMATOES",
                sell_price,
                -sell_qty,
            )

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        self.emerald_position = self.get_product_pos(state, "EMERALDS")
        self.emerald_buy_orders = 0
        self.emerald_sell_orders = 0

        self.tomato_position = self.get_product_pos(state, "TOMATOES")
        self.tomato_buy_orders = 0
        self.tomato_sell_orders = 0

        for product in state.order_depths:
            result[product] = []

        if "EMERALDS" in state.order_depths:
            self.trade_emeralds(state, result["EMERALDS"])

        if "TOMATOES" in state.order_depths:
            self.trade_tomatoes(state, result["TOMATOES"])

        traderData = ""
        conversions = 0
        return result, conversions, traderData