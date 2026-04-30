from datamodel import Order, TradingState
from typing import Dict, List
import json
import statistics


class Trader:
    def __init__(self):
        self.CHOCOLATE = "SNACKPACK_CHOCOLATE"
        self.VANILLA = "SNACKPACK_VANILLA"

        self.POSITION_LIMIT = 10

        # Core relation:
        # Chocolate + 0.87 * Vanilla is relatively stable.
        # Vanilla is signal / anchor.
        # Chocolate is trading / monetization leg.
        self.VANILLA_WEIGHT = 0.87

        # Basket fair-value window
        self.WINDOW = 500
        self.MIN_HISTORY = 120

        # Lead-lag signal from Vanilla
        self.LEAD_WINDOW = 5
        self.MIN_VANILLA_SIGNAL = 4.0

        # Trading edges
        self.TAKE_EDGE = 22
        self.PASSIVE_EDGE = 15
        self.EXTREME_EDGE = 34

        # Exit logic
        self.CLOSE_EDGE = 5
        self.HOLD_EDGE = 12
        self.MAX_HOLD_TICKS = 90

        # Size control
        self.TAKE_SIZE = 2
        self.PASSIVE_SIZE = 1
        self.CLOSE_SIZE = 2

        # Inventory control
        self.INVENTORY_SKEW = 1.5
        self.SOFT_LIMIT = 8
        self.SOFT_LIMIT_EXTRA_SKEW = 5

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
            return quantity

        return 0

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
            return quantity

        return 0

    def get_vanilla_lead_signal(self, data: dict):
        """
        Negative relation:
        - Vanilla down recently => Chocolate expected up
        - Vanilla up recently   => Chocolate expected down
        """
        vanilla_history = data.get("vanilla_mid_history", [])

        if len(vanilla_history) < 2:
            return 0.0, False, False

        vanilla_move = vanilla_history[-1] - vanilla_history[0]

        bullish_chocolate = vanilla_move <= -self.MIN_VANILLA_SIGNAL
        bearish_chocolate = vanilla_move >= self.MIN_VANILLA_SIGNAL

        return vanilla_move, bullish_chocolate, bearish_chocolate

    def reduce_position(
        self,
        state: TradingState,
        chocolate_fair: float,
        force_reduce: bool = False,
    ) -> List[Order]:

        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(state, self.CHOCOLATE)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(self.CHOCOLATE, 0)

        if position == 0:
            return orders

        adjusted_fair = chocolate_fair - self.INVENTORY_SKEW * position

        if force_reduce:
            if position > 0:
                self.add_sell_order(
                    orders,
                    self.CHOCOLATE,
                    best_bid,
                    min(self.CLOSE_SIZE, position),
                    position,
                )
            elif position < 0:
                self.add_buy_order(
                    orders,
                    self.CHOCOLATE,
                    best_ask,
                    min(self.CLOSE_SIZE, -position),
                    position,
                )
            return orders

        if position > 0:
            if best_bid >= adjusted_fair - self.CLOSE_EDGE:
                self.add_sell_order(
                    orders,
                    self.CHOCOLATE,
                    best_bid,
                    min(self.CLOSE_SIZE, position),
                    position,
                )

        elif position < 0:
            if best_ask <= adjusted_fair + self.CLOSE_EDGE:
                self.add_buy_order(
                    orders,
                    self.CHOCOLATE,
                    best_ask,
                    min(self.CLOSE_SIZE, -position),
                    position,
                )

        return orders

    def trade_chocolate(
        self,
        state: TradingState,
        chocolate_fair: float,
        data: dict,
    ) -> List[Order]:

        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(state, self.CHOCOLATE)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(self.CHOCOLATE, 0)

        if "position_age" not in data:
            data["position_age"] = 0

        if position == 0:
            data["position_age"] = 0
        else:
            data["position_age"] += 1

        vanilla_move, bullish_chocolate, bearish_chocolate = self.get_vanilla_lead_signal(data)

        adjusted_fair = chocolate_fair - self.INVENTORY_SKEW * position

        if position >= self.SOFT_LIMIT:
            adjusted_fair -= self.SOFT_LIMIT_EXTRA_SKEW
        elif position <= -self.SOFT_LIMIT:
            adjusted_fair += self.SOFT_LIMIT_EXTRA_SKEW

        buy_edge = adjusted_fair - best_ask
        sell_edge = best_bid - adjusted_fair

        # ------------------------------------------------------------
        # 1. Exit first.
        # Core insight from gradual information diffusion:
        # If the signal has not worked after some time and edge is weak,
        # do not keep trusting stale signal forever.
        # ------------------------------------------------------------

        if position > 0:
            # If long Chocolate, but edge has disappeared, sell.
            if best_bid >= adjusted_fair - self.CLOSE_EDGE:
                self.add_sell_order(
                    orders,
                    self.CHOCOLATE,
                    best_bid,
                    min(self.CLOSE_SIZE, position),
                    position,
                )
                return orders

            # If long too long and buy edge is no longer strong, reduce.
            if data["position_age"] > self.MAX_HOLD_TICKS and buy_edge < self.HOLD_EDGE:
                self.add_sell_order(
                    orders,
                    self.CHOCOLATE,
                    best_bid,
                    min(self.CLOSE_SIZE, position),
                    position,
                )
                return orders

            # If Vanilla now gives opposite bearish signal, reduce.
            if bearish_chocolate and buy_edge < self.TAKE_EDGE:
                self.add_sell_order(
                    orders,
                    self.CHOCOLATE,
                    best_bid,
                    min(self.CLOSE_SIZE, position),
                    position,
                )
                return orders

        elif position < 0:
            # If short Chocolate, but edge has disappeared, buy back.
            if best_ask <= adjusted_fair + self.CLOSE_EDGE:
                self.add_buy_order(
                    orders,
                    self.CHOCOLATE,
                    best_ask,
                    min(self.CLOSE_SIZE, -position),
                    position,
                )
                return orders

            # If short too long and sell edge is no longer strong, reduce.
            if data["position_age"] > self.MAX_HOLD_TICKS and sell_edge < self.HOLD_EDGE:
                self.add_buy_order(
                    orders,
                    self.CHOCOLATE,
                    best_ask,
                    min(self.CLOSE_SIZE, -position),
                    position,
                )
                return orders

            # If Vanilla now gives opposite bullish signal, reduce.
            if bullish_chocolate and sell_edge < self.TAKE_EDGE:
                self.add_buy_order(
                    orders,
                    self.CHOCOLATE,
                    best_ask,
                    min(self.CLOSE_SIZE, -position),
                    position,
                )
                return orders

        # ------------------------------------------------------------
        # 2. Entry.
        # We do not trade every Vanilla signal.
        # We require:
        #   A. Chocolate is mispriced against implied fair
        #   B. Vanilla lead signal agrees
        # OR:
        #   C. Chocolate edge is extreme enough by itself
        # ------------------------------------------------------------

        allow_buy = bullish_chocolate or buy_edge > self.EXTREME_EDGE
        allow_sell = bearish_chocolate or sell_edge > self.EXTREME_EDGE

        # Strong active buy
        if buy_edge > self.TAKE_EDGE and allow_buy:
            self.add_buy_order(
                orders,
                self.CHOCOLATE,
                best_ask,
                self.TAKE_SIZE,
                position,
            )
            return orders

        # Strong active sell
        if sell_edge > self.TAKE_EDGE and allow_sell:
            self.add_sell_order(
                orders,
                self.CHOCOLATE,
                best_bid,
                self.TAKE_SIZE,
                position,
            )
            return orders

        # ------------------------------------------------------------
        # 3. Passive order.
        # Only place passive order if the Vanilla signal is not against us.
        # This avoids paying spread too often while still allowing fills.
        # ------------------------------------------------------------

        passive_buy_price = best_bid + 1
        if passive_buy_price >= best_ask:
            passive_buy_price = best_bid

        passive_sell_price = best_ask - 1
        if passive_sell_price <= best_bid:
            passive_sell_price = best_ask

        passive_buy_edge = adjusted_fair - passive_buy_price
        passive_sell_edge = passive_sell_price - adjusted_fair

        if passive_buy_edge > self.PASSIVE_EDGE and not bearish_chocolate:
            self.add_buy_order(
                orders,
                self.CHOCOLATE,
                passive_buy_price,
                self.PASSIVE_SIZE,
                position,
            )

        elif passive_sell_edge > self.PASSIVE_EDGE and not bullish_chocolate:
            self.add_sell_order(
                orders,
                self.CHOCOLATE,
                passive_sell_price,
                self.PASSIVE_SIZE,
                position,
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

        if "vanilla_mid_history" not in data:
            data["vanilla_mid_history"] = []

        chocolate_mid = self.get_mid_price(state, self.CHOCOLATE)
        vanilla_mid = self.get_mid_price(state, self.VANILLA)

        if chocolate_mid is None or vanilla_mid is None:
            return result, conversions, json.dumps(data)

        # Update Vanilla lead window
        data["vanilla_mid_history"].append(vanilla_mid)
        if len(data["vanilla_mid_history"]) > self.LEAD_WINDOW:
            data["vanilla_mid_history"] = data["vanilla_mid_history"][-self.LEAD_WINDOW:]

        # Core basket relation
        basket = chocolate_mid + self.VANILLA_WEIGHT * vanilla_mid

        data["basket_history"].append(basket)

        if len(data["basket_history"]) > self.WINDOW:
            data["basket_history"] = data["basket_history"][-self.WINDOW:]

        history = data["basket_history"]

        if len(history) < self.MIN_HISTORY:
            return result, conversions, json.dumps(data)

        basket_mean = statistics.mean(history)

        chocolate_fair = basket_mean - self.VANILLA_WEIGHT * vanilla_mid

        result[self.CHOCOLATE] += self.trade_chocolate(
            state,
            chocolate_fair,
            data,
        )

        data["last_basket"] = basket
        data["last_basket_mean"] = basket_mean
        data["last_chocolate_fair"] = chocolate_fair
        data["last_chocolate_position"] = state.position.get(self.CHOCOLATE, 0)
        data["last_vanilla_mid"] = vanilla_mid
        data["last_chocolate_mid"] = chocolate_mid

        return result, conversions, json.dumps(data)