from datamodel import Order, TradingState
from typing import Dict, List
import json
import statistics


class Trader:
    def __init__(self):
        self.CHOCOLATE = "SNACKPACK_CHOCOLATE"
        self.VANILLA = "SNACKPACK_VANILLA"

        self.POSITION_LIMIT = 10

        # Rolling regression parameters
        self.WINDOW = 500
        self.MIN_HISTORY = 120

        # Relationship filter
        self.MIN_BETA_ABS = 0.60
        self.MAX_BETA_ABS = 1.35

        # Residual z-score thresholds
        self.ENTRY_Z_1 = 0.90
        self.ENTRY_Z_2 = 1.35
        self.ENTRY_Z_3 = 1.85
        self.ENTRY_Z_4 = 2.40

        self.EXIT_Z = 0.35

        # If residual is too extreme, avoid adding more.
        self.DANGER_Z = 4.00

        # Execution size per timestamp
        self.MAX_TRADE_SIZE = 2

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

    def rolling_regression(self, x_list, y_list):
        """
        Fit:
            y = alpha + beta * x

        x = Vanilla mid
        y = Chocolate mid
        """
        if len(x_list) < self.MIN_HISTORY:
            return None, None, None

        x_mean = statistics.mean(x_list)
        y_mean = statistics.mean(y_list)

        var_x = 0.0
        cov_xy = 0.0

        for x, y in zip(x_list, y_list):
            dx = x - x_mean
            dy = y - y_mean
            var_x += dx * dx
            cov_xy += dx * dy

        if var_x == 0:
            return None, None, None

        beta = cov_xy / var_x
        alpha = y_mean - beta * x_mean

        residuals = []
        for x, y in zip(x_list, y_list):
            fair_y = alpha + beta * x
            residuals.append(y - fair_y)

        residual_std = statistics.pstdev(residuals)

        if residual_std == 0:
            return None, None, None

        return alpha, beta, residual_std

    def add_order_toward_target(
        self,
        state: TradingState,
        product: str,
        target_position: int,
        aggressive: bool,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(state, product)

        if best_bid is None or best_ask is None:
            return orders

        current_position = state.position.get(product, 0)

        target_position = max(
            -self.POSITION_LIMIT,
            min(self.POSITION_LIMIT, target_position),
        )

        delta = target_position - current_position

        if delta == 0:
            return orders

        # Do not move too much in one timestamp.
        if delta > 0:
            quantity = min(delta, self.MAX_TRADE_SIZE)

            if aggressive:
                price = best_ask
            else:
                price = best_bid + 1
                if price >= best_ask:
                    price = best_bid

            orders.append(Order(product, int(price), quantity))

        elif delta < 0:
            quantity = max(delta, -self.MAX_TRADE_SIZE)

            if aggressive:
                price = best_bid
            else:
                price = best_ask - 1
                if price <= best_bid:
                    price = best_ask

            orders.append(Order(product, int(price), quantity))

        return orders

    def get_base_target_from_z(self, z: float) -> int:
        """
        Convert residual z-score into target absolute position.
        Larger residual = larger target.
        """
        abs_z = abs(z)

        if abs_z < self.EXIT_Z:
            return 0

        if abs_z < self.ENTRY_Z_1:
            return 0

        if abs_z < self.ENTRY_Z_2:
            return 3

        if abs_z < self.ENTRY_Z_3:
            return 5

        if abs_z < self.ENTRY_Z_4:
            return 7

        return 10

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

        if "chocolate_history" not in data:
            data["chocolate_history"] = []

        if "vanilla_history" not in data:
            data["vanilla_history"] = []

        chocolate_mid = self.get_mid_price(state, self.CHOCOLATE)
        vanilla_mid = self.get_mid_price(state, self.VANILLA)

        if chocolate_mid is None or vanilla_mid is None:
            traderData = json.dumps(data)
            return result, conversions, traderData

        data["chocolate_history"].append(chocolate_mid)
        data["vanilla_history"].append(vanilla_mid)

        if len(data["chocolate_history"]) > self.WINDOW:
            data["chocolate_history"] = data["chocolate_history"][-self.WINDOW:]

        if len(data["vanilla_history"]) > self.WINDOW:
            data["vanilla_history"] = data["vanilla_history"][-self.WINDOW:]

        chocolate_hist = data["chocolate_history"]
        vanilla_hist = data["vanilla_history"]

        if len(chocolate_hist) < self.MIN_HISTORY:
            traderData = json.dumps(data)
            return result, conversions, traderData

        alpha, beta, residual_std = self.rolling_regression(
            vanilla_hist,
            chocolate_hist,
        )

        if alpha is None or beta is None or residual_std is None:
            traderData = json.dumps(data)
            return result, conversions, traderData

        # Only use this strategy when Chocolate/Vanilla relationship is clearly negative.
        if beta >= 0:
            traderData = json.dumps(data)
            return result, conversions, traderData

        beta_abs = abs(beta)

        if beta_abs < self.MIN_BETA_ABS or beta_abs > self.MAX_BETA_ABS:
            traderData = json.dumps(data)
            return result, conversions, traderData

        chocolate_fair = alpha + beta * vanilla_mid

        residual = chocolate_mid - chocolate_fair
        z = residual / residual_std

        base_target = self.get_base_target_from_z(z)

        chocolate_target = state.position.get(self.CHOCOLATE, 0)
        vanilla_target = state.position.get(self.VANILLA, 0)

        # residual = Chocolate - alpha - beta * Vanilla
        # beta < 0, so residual behaves like Chocolate + abs(beta) * Vanilla.
        #
        # residual high:
        #     basket too expensive -> sell both
        #
        # residual low:
        #     basket too cheap -> buy both
        if abs(z) < self.EXIT_Z:
            chocolate_target = 0
            vanilla_target = 0

        elif abs(z) > self.DANGER_Z:
            # Extreme residual can mean regime shift.
            # Do not add more. Only reduce if close to exit later.
            chocolate_target = state.position.get(self.CHOCOLATE, 0)
            vanilla_target = state.position.get(self.VANILLA, 0)

        elif z > 0:
            # Basket too expensive
            chocolate_target = -base_target
            vanilla_target = -round(base_target * beta_abs)

        elif z < 0:
            # Basket too cheap
            chocolate_target = base_target
            vanilla_target = round(base_target * beta_abs)

        aggressive = abs(z) > self.ENTRY_Z_4 or abs(z) < self.EXIT_Z

        result[self.CHOCOLATE] += self.add_order_toward_target(
            state,
            self.CHOCOLATE,
            chocolate_target,
            aggressive=aggressive,
        )

        result[self.VANILLA] += self.add_order_toward_target(
            state,
            self.VANILLA,
            vanilla_target,
            aggressive=aggressive,
        )

        data["last_alpha"] = alpha
        data["last_beta"] = beta
        data["last_beta_abs"] = beta_abs
        data["last_residual"] = residual
        data["last_residual_std"] = residual_std
        data["last_z"] = z
        data["last_chocolate_fair"] = chocolate_fair
        data["last_chocolate_target"] = chocolate_target
        data["last_vanilla_target"] = vanilla_target

        traderData = json.dumps(data)

        return result, conversions, traderData