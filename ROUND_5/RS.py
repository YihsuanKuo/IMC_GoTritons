from datamodel import Order, TradingState
from typing import Dict, List
import json
import statistics


class Trader:
    def __init__(self):
        self.RASPBERRY = "SNACKPACK_RASPBERRY"
        self.STRAWBERRY = "SNACKPACK_STRAWBERRY"

        self.POSITION_LIMIT = 10

        # Relationship assumption:
        # RASPBERRY + 0.80 * STRAWBERRY is relatively stable.
        #
        # This is intentionally not rolling beta.
        # Raspberry/Strawberry beta changes across days, so 0.80 is a stable compromise.
        self.STRAWBERRY_WEIGHT = 0.80

        # Longer window because this pair reverts slower than Chocolate/Vanilla.
        self.WINDOW = 700
        self.MIN_HISTORY = 150

        # Wider edges because residual volatility is larger.
        self.TAKE_EDGE = 34
        self.PASSIVE_EDGE = 20
        self.CLOSE_EDGE = 5

        # Conservative sizing.
        self.TAKE_SIZE = 1
        self.PASSIVE_SIZE = 1
        self.CLOSE_SIZE = 2

        # Stronger inventory control than Chocolate/Vanilla.
        self.INVENTORY_SKEW = 2.0
        self.SOFT_LIMIT = 6

    def get_best_bid_ask(self, state: TradingState, product: str):
        order_depth = state.order_depths.get(product, None)

        if order_depth is None:
            return None, None

        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return None, None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        return best_bid, best_ask

    def get_mid_price(self, state: TradingState, product: str):
        best_bid, best_ask = self.get_best_bid_ask(state, product)

        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2

    def add_buy_order(
        self,
        orders: List[Order],
        product: str,
        price: int,
        quantity: int,
        current_position: int,
    ):
        max_buy = self.POSITION_LIMIT - current_position
        quantity = min(quantity, max_buy)

        if quantity > 0:
            orders.append(Order(product, int(price), quantity))

    def add_sell_order(
        self,
        orders: List[Order],
        product: str,
        price: int,
        quantity: int,
        current_position: int,
    ):
        max_sell = self.POSITION_LIMIT + current_position
        quantity = min(quantity, max_sell)

        if quantity > 0:
            orders.append(Order(product, int(price), -quantity))

    def reduce_position_if_edge_gone(
        self,
        orders: List[Order],
        product: str,
        position: int,
        best_bid: int,
        best_ask: int,
        adjusted_fair: float,
    ):
        """
        If position exists and price has moved back near fair,
        reduce inventory.
        """
        if position > 0:
            # Long position: sell if bid is close enough to fair.
            if best_bid >= adjusted_fair - self.CLOSE_EDGE:
                self.add_sell_order(
                    orders,
                    product,
                    best_bid,
                    min(self.CLOSE_SIZE, position),
                    position,
                )

        elif position < 0:
            # Short position: buy back if ask is close enough to fair.
            if best_ask <= adjusted_fair + self.CLOSE_EDGE:
                self.add_buy_order(
                    orders,
                    product,
                    best_ask,
                    min(self.CLOSE_SIZE, -position),
                    position,
                )

    def trade_against_fair(
        self,
        state: TradingState,
        product: str,
        fair_value: float,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(state, product)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)

        # Inventory skew:
        # If long, lower fair to encourage selling.
        # If short, raise fair to encourage buying.
        adjusted_fair = fair_value - self.INVENTORY_SKEW * position

        # Extra inventory protection near soft limit.
        if position >= self.SOFT_LIMIT:
            adjusted_fair -= 8
        elif position <= -self.SOFT_LIMIT:
            adjusted_fair += 8

        buy_edge = adjusted_fair - best_ask
        sell_edge = best_bid - adjusted_fair

        # 1. First, if current position is no longer supported, reduce.
        if position > 0 and best_bid >= adjusted_fair - self.CLOSE_EDGE:
            self.add_sell_order(
                orders,
                product,
                best_bid,
                min(self.CLOSE_SIZE, position),
                position,
            )
            return orders

        if position < 0 and best_ask <= adjusted_fair + self.CLOSE_EDGE:
            self.add_buy_order(
                orders,
                product,
                best_ask,
                min(self.CLOSE_SIZE, -position),
                position,
            )
            return orders

        # 2. Strong edge: take liquidity, but only 1 lot.
        if buy_edge > self.TAKE_EDGE:
            self.add_buy_order(
                orders,
                product,
                best_ask,
                self.TAKE_SIZE,
                position,
            )
            return orders

        if sell_edge > self.TAKE_EDGE:
            self.add_sell_order(
                orders,
                product,
                best_bid,
                self.TAKE_SIZE,
                position,
            )
            return orders

        # 3. Medium edge: place passive order.
        passive_buy_price = best_bid + 1
        if passive_buy_price >= best_ask:
            passive_buy_price = best_bid

        passive_sell_price = best_ask - 1
        if passive_sell_price <= best_bid:
            passive_sell_price = best_ask

        passive_buy_edge = adjusted_fair - passive_buy_price
        passive_sell_edge = passive_sell_price - adjusted_fair

        if passive_buy_edge > self.PASSIVE_EDGE:
            self.add_buy_order(
                orders,
                product,
                passive_buy_price,
                self.PASSIVE_SIZE,
                position,
            )

        elif passive_sell_edge > self.PASSIVE_EDGE:
            self.add_sell_order(
                orders,
                product,
                passive_sell_price,
                self.PASSIVE_SIZE,
                position,
            )

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {
            self.RASPBERRY: [],
            self.STRAWBERRY: [],
        }

        conversions = 0

        # Load traderData
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        if "rs_basket_history" not in data:
            data["rs_basket_history"] = []

        raspberry_mid = self.get_mid_price(state, self.RASPBERRY)
        strawberry_mid = self.get_mid_price(state, self.STRAWBERRY)

        if raspberry_mid is None or strawberry_mid is None:
            traderData = json.dumps(data)
            return result, conversions, traderData

        # Negative basket relationship
        basket = raspberry_mid + self.STRAWBERRY_WEIGHT * strawberry_mid

        data["rs_basket_history"].append(basket)

        if len(data["rs_basket_history"]) > self.WINDOW:
            data["rs_basket_history"] = data["rs_basket_history"][-self.WINDOW:]

        history = data["rs_basket_history"]

        if len(history) < self.MIN_HISTORY:
            traderData = json.dumps(data)
            return result, conversions, traderData

        basket_mean = statistics.mean(history)

        # Implied fair values
        raspberry_fair = basket_mean - self.STRAWBERRY_WEIGHT * strawberry_mid
        strawberry_fair = (basket_mean - raspberry_mid) / self.STRAWBERRY_WEIGHT

        result[self.RASPBERRY] += self.trade_against_fair(
            state,
            self.RASPBERRY,
            raspberry_fair,
        )

        result[self.STRAWBERRY] += self.trade_against_fair(
            state,
            self.STRAWBERRY,
            strawberry_fair,
        )

        data["last_rs_basket"] = basket
        data["last_rs_basket_mean"] = basket_mean
        data["last_raspberry_fair"] = raspberry_fair
        data["last_strawberry_fair"] = strawberry_fair

        traderData = json.dumps(data)

        return result, conversions, traderData