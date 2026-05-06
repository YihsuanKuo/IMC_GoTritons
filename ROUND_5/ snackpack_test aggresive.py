from datamodel import Order, TradingState
from typing import Dict, List
import json
import statistics


PROFILE = "aggressive"
# PROFILE = "conservative"


class Trader:
    def __init__(self):
        self.CHOCOLATE = "SNACKPACK_CHOCOLATE"
        self.VANILLA = "SNACKPACK_VANILLA"

        self.POSITION_LIMIT = 10
        self.VANILLA_WEIGHT = 0.87

        if PROFILE == "aggressive":
            # This is close to the version that gave you around +221.
            self.WINDOW = 500
            self.MIN_HISTORY = 120

            self.TAKE_EDGE = 22
            self.PASSIVE_EDGE = 14
            self.CLOSE_EDGE = 2

            self.INVENTORY_SKEW = 1.5

            self.TAKE_SIZE = 2
            self.PASSIVE_SIZE = 1

            self.USE_REDUCE_ONLY = False
            self.REDUCE_ONLY_POS = 8

            self.USE_REGIME_FILTER = False
            self.REGIME_Z_LIMIT = 4.0

        else:
            # More defensive version: fewer aggressive fills, stronger inventory control.
            self.WINDOW = 700
            self.MIN_HISTORY = 180

            self.TAKE_EDGE = 30
            self.PASSIVE_EDGE = 16
            self.CLOSE_EDGE = 3

            self.INVENTORY_SKEW = 2.0

            self.TAKE_SIZE = 1
            self.PASSIVE_SIZE = 1

            self.USE_REDUCE_ONLY = True
            self.REDUCE_ONLY_POS = 8

            self.USE_REGIME_FILTER = True
            self.REGIME_Z_LIMIT = 3.8

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

    def close_position_if_edge_disappeared(
        self,
        orders: List[Order],
        product: str,
        position: int,
        best_bid: int,
        best_ask: int,
        adjusted_fair: float,
    ):
        """
        If we already have position and edge disappears, reduce position.
        Long position: sell if bid is close enough to fair.
        Short position: buy if ask is close enough to fair.
        """
        if position > 0:
            if best_bid >= adjusted_fair - self.CLOSE_EDGE:
                self.add_sell_order(
                    orders,
                    product,
                    best_bid,
                    min(2, position),
                    position,
                )

        elif position < 0:
            if best_ask <= adjusted_fair + self.CLOSE_EDGE:
                self.add_buy_order(
                    orders,
                    product,
                    best_ask,
                    min(2, -position),
                    position,
                )

    def trade_against_fair(
        self,
        state: TradingState,
        product: str,
        fair_value: float,
        basket_z: float,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(state, product)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)

        # Inventory skew:
        # long position -> lower fair -> encourage selling
        # short position -> raise fair -> encourage buying
        adjusted_fair = fair_value - self.INVENTORY_SKEW * position

        buy_edge = adjusted_fair - best_ask
        sell_edge = best_bid - adjusted_fair

        # Conservative safety:
        # If position is already large, stop adding in the same direction.
        reduce_only = False
        if self.USE_REDUCE_ONLY and abs(position) >= self.REDUCE_ONLY_POS:
            reduce_only = True

        # Conservative safety:
        # If basket is in extreme regime, relationship may be unstable.
        # In that case, do not open new positions; only reduce existing ones.
        if self.USE_REGIME_FILTER and abs(basket_z) > self.REGIME_Z_LIMIT:
            reduce_only = True

        if reduce_only:
            self.close_position_if_edge_disappeared(
                orders,
                product,
                position,
                best_bid,
                best_ask,
                adjusted_fair,
            )
            return orders

        # 1. Strong executable edge: take liquidity
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

        # 2. Passive market making with directional edge
        passive_buy_price = best_bid + 1
        passive_sell_price = best_ask - 1

        # Avoid crossing the market
        if passive_buy_price >= best_ask:
            passive_buy_price = best_bid

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

        # 3. If no fresh edge, gently close existing position
        else:
            self.close_position_if_edge_disappeared(
                orders,
                product,
                position,
                best_bid,
                best_ask,
                adjusted_fair,
            )

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {
            self.CHOCOLATE: [],
            self.VANILLA: [],
        }

        conversions = 0

        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        if "basket_history" not in data:
            data["basket_history"] = []

        chocolate_mid = self.get_mid_price(state, self.CHOCOLATE)
        vanilla_mid = self.get_mid_price(state, self.VANILLA)

        if chocolate_mid is None or vanilla_mid is None:
            traderData = json.dumps(data)
            return result, conversions, traderData

        # Negative relation:
        # CHOCOLATE + 0.87 * VANILLA is relatively stable
        basket = chocolate_mid + self.VANILLA_WEIGHT * vanilla_mid

        data["basket_history"].append(basket)

        if len(data["basket_history"]) > self.WINDOW:
            data["basket_history"] = data["basket_history"][-self.WINDOW:]

        history = data["basket_history"]

        if len(history) < self.MIN_HISTORY:
            traderData = json.dumps(data)
            return result, conversions, traderData

        basket_mean = statistics.mean(history)
        basket_std = statistics.pstdev(history)

        if basket_std == 0:
            traderData = json.dumps(data)
            return result, conversions, traderData

        basket_z = (basket - basket_mean) / basket_std

        # Implied fair values from basket relation
        chocolate_fair = basket_mean - self.VANILLA_WEIGHT * vanilla_mid
        vanilla_fair = (basket_mean - chocolate_mid) / self.VANILLA_WEIGHT

        result[self.CHOCOLATE] += self.trade_against_fair(
            state,
            self.CHOCOLATE,
            chocolate_fair,
            basket_z,
        )

        result[self.VANILLA] += self.trade_against_fair(
            state,
            self.VANILLA,
            vanilla_fair,
            basket_z,
        )

        data["last_basket"] = basket
        data["last_basket_mean"] = basket_mean
        data["last_basket_std"] = basket_std
        data["last_basket_z"] = basket_z
        data["last_chocolate_fair"] = chocolate_fair
        data["last_vanilla_fair"] = vanilla_fair
        data["profile"] = PROFILE

        traderData = json.dumps(data)

        return result, conversions, traderData