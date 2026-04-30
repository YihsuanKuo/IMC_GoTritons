from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


class Trader:
    # Round 5: all products have position limit 10
    POSITION_LIMIT = 10

    # 50 products
    PRODUCTS = [
        # Galaxy Sounds Recorders
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",

        # Vertical Sleeping Pods
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",

        # Organic Microchips
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_TRIANGLE",

        # Purification Pebbles
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",

        # Domestic Robots
        "ROBOT_VACUUMING",
        "ROBOT_MOPPING",
        "ROBOT_DISHES",
        "ROBOT_LAUNDRY",
        "ROBOT_IRONING",

        # UV-Visors
        "UV_VISOR_YELLOW",
        "UV_VISOR_AMBER",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_MAGENTA",

        # Instant Translators
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",

        # Construction Panels
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",

        # Liquid Breath Oxygen Shakes
        "OXYGEN_SHAKE_MORNING_BREATH",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_GARLIC",

        # Protein Snack Packs
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ]

    GROUPS = {
        "GALAXY": [
            "GALAXY_SOUNDS_DARK_MATTER",
            "GALAXY_SOUNDS_BLACK_HOLES",
            "GALAXY_SOUNDS_PLANETARY_RINGS",
            "GALAXY_SOUNDS_SOLAR_WINDS",
            "GALAXY_SOUNDS_SOLAR_FLAMES",
        ],
        "SLEEP": [
            "SLEEP_POD_SUEDE",
            "SLEEP_POD_LAMB_WOOL",
            "SLEEP_POD_POLYESTER",
            "SLEEP_POD_NYLON",
            "SLEEP_POD_COTTON",
        ],
        "MICROCHIP": [
            "MICROCHIP_CIRCLE",
            "MICROCHIP_OVAL",
            "MICROCHIP_SQUARE",
            "MICROCHIP_RECTANGLE",
            "MICROCHIP_TRIANGLE",
        ],
        "PEBBLES": [
            "PEBBLES_XS",
            "PEBBLES_S",
            "PEBBLES_M",
            "PEBBLES_L",
            "PEBBLES_XL",
        ],
        "ROBOT": [
            "ROBOT_VACUUMING",
            "ROBOT_MOPPING",
            "ROBOT_DISHES",
            "ROBOT_LAUNDRY",
            "ROBOT_IRONING",
        ],
        "UV": [
            "UV_VISOR_YELLOW",
            "UV_VISOR_AMBER",
            "UV_VISOR_ORANGE",
            "UV_VISOR_RED",
            "UV_VISOR_MAGENTA",
        ],
        "TRANSLATOR": [
            "TRANSLATOR_SPACE_GRAY",
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_VOID_BLUE",
        ],
        "PANEL": [
            "PANEL_1X2",
            "PANEL_2X2",
            "PANEL_1X4",
            "PANEL_2X4",
            "PANEL_4X4",
        ],
        "OXYGEN": [
            "OXYGEN_SHAKE_MORNING_BREATH",
            "OXYGEN_SHAKE_EVENING_BREATH",
            "OXYGEN_SHAKE_MINT",
            "OXYGEN_SHAKE_CHOCOLATE",
            "OXYGEN_SHAKE_GARLIC",
        ],
        "SNACKPACK": [
            "SNACKPACK_CHOCOLATE",
            "SNACKPACK_VANILLA",
            "SNACKPACK_PISTACHIO",
            "SNACKPACK_STRAWBERRY",
            "SNACKPACK_RASPBERRY",
        ],
    }

    def __init__(self):
        self.product_group = {}
        for group, products in self.GROUPS.items():
            for product in products:
                self.product_group[product] = group

        # Group-specific parameters.
        # Idea:
        # - ROBOT / TRANSLATOR / MICROCHIP have lower spread, can be more active.
        # - SNACKPACK has wider spread, use higher edge and smaller size.
        # - PEBBLES / GALAXY / OXYGEN are noisier, use stricter z.
        self.group_params = {
            "ROBOT": {
                "history": 30,
                "min_history": 10,
                "entry_z": 1.05,
                "exit_z": 0.35,
                "take_edge": 8,
                "passive_edge": 4,
                "quote_offset": 2,
                "inventory_skew": 0.8,
                "take_size": 2,
                "passive_size": 1,
            },
            "TRANSLATOR": {
                "history": 35,
                "min_history": 12,
                "entry_z": 1.10,
                "exit_z": 0.35,
                "take_edge": 10,
                "passive_edge": 5,
                "quote_offset": 2,
                "inventory_skew": 0.9,
                "take_size": 2,
                "passive_size": 1,
            },
            "MICROCHIP": {
                "history": 35,
                "min_history": 12,
                "entry_z": 1.15,
                "exit_z": 0.40,
                "take_edge": 10,
                "passive_edge": 5,
                "quote_offset": 2,
                "inventory_skew": 0.9,
                "take_size": 2,
                "passive_size": 1,
            },
            "PANEL": {
                "history": 40,
                "min_history": 15,
                "entry_z": 1.20,
                "exit_z": 0.40,
                "take_edge": 12,
                "passive_edge": 6,
                "quote_offset": 3,
                "inventory_skew": 1.0,
                "take_size": 1,
                "passive_size": 1,
            },
            "SLEEP": {
                "history": 40,
                "min_history": 15,
                "entry_z": 1.20,
                "exit_z": 0.40,
                "take_edge": 12,
                "passive_edge": 6,
                "quote_offset": 3,
                "inventory_skew": 1.0,
                "take_size": 1,
                "passive_size": 1,
            },
            "UV": {
                "history": 45,
                "min_history": 16,
                "entry_z": 1.25,
                "exit_z": 0.45,
                "take_edge": 15,
                "passive_edge": 8,
                "quote_offset": 4,
                "inventory_skew": 1.1,
                "take_size": 1,
                "passive_size": 1,
            },
            "PEBBLES": {
                "history": 50,
                "min_history": 18,
                "entry_z": 1.30,
                "exit_z": 0.45,
                "take_edge": 18,
                "passive_edge": 10,
                "quote_offset": 5,
                "inventory_skew": 1.2,
                "take_size": 1,
                "passive_size": 1,
            },
            "GALAXY": {
                "history": 45,
                "min_history": 16,
                "entry_z": 1.30,
                "exit_z": 0.45,
                "take_edge": 18,
                "passive_edge": 10,
                "quote_offset": 5,
                "inventory_skew": 1.2,
                "take_size": 1,
                "passive_size": 1,
            },
            "OXYGEN": {
                "history": 45,
                "min_history": 16,
                "entry_z": 1.30,
                "exit_z": 0.45,
                "take_edge": 18,
                "passive_edge": 10,
                "quote_offset": 5,
                "inventory_skew": 1.2,
                "take_size": 1,
                "passive_size": 1,
            },
            "SNACKPACK": {
                "history": 60,
                "min_history": 20,
                "entry_z": 1.35,
                "exit_z": 0.45,
                "take_edge": 24,
                "passive_edge": 14,
                "quote_offset": 7,
                "inventory_skew": 1.5,
                "take_size": 1,
                "passive_size": 1,
            },
        }

    def get_params(self, product):
        group = self.product_group.get(product, "PANEL")
        return self.group_params[group]

    def get_best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    def mean_std(self, values):
        if len(values) == 0:
            return None, None
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, math.sqrt(var)

    def add_buy_order(self, orders, product, price, qty, position):
        max_buy = self.POSITION_LIMIT - position
        qty = min(qty, max_buy)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))
            return qty
        return 0

    def add_sell_order(self, orders, product, price, qty, position):
        max_sell = self.POSITION_LIMIT + position
        qty = min(qty, max_sell)
        if qty > 0:
            orders.append(Order(product, int(price), -int(qty)))
            return qty
        return 0

    def sweep_buy(self, product, order_depth, fair, position, max_qty):
        orders = []
        remaining = min(max_qty, self.POSITION_LIMIT - position)
        temp_position = position

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price > fair:
                break

            available = abs(order_depth.sell_orders[ask_price])
            qty = min(available, remaining)
            if qty <= 0:
                continue

            filled = self.add_buy_order(
                orders,
                product,
                ask_price,
                qty,
                temp_position,
            )
            temp_position += filled
            remaining -= filled

            if remaining <= 0:
                break

        return orders

    def sweep_sell(self, product, order_depth, fair, position, max_qty):
        orders = []
        remaining = min(max_qty, self.POSITION_LIMIT + position)
        temp_position = position

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price < fair:
                break

            available = order_depth.buy_orders[bid_price]
            qty = min(available, remaining)
            if qty <= 0:
                continue

            filled = self.add_sell_order(
                orders,
                product,
                bid_price,
                qty,
                temp_position,
            )
            temp_position -= filled
            remaining -= filled

            if remaining <= 0:
                break

        return orders

    def close_if_reverted(self, product, order_depth, fair, z, position, params):
        orders = []
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_bid is None or best_ask is None:
            return orders

        exit_z = params["exit_z"]
        close_size = 2

        if abs(z) > exit_z:
            return orders

        if position > 0:
            self.add_sell_order(
                orders,
                product,
                best_bid,
                min(close_size, position),
                position,
            )

        elif position < 0:
            self.add_buy_order(
                orders,
                product,
                best_ask,
                min(close_size, -position),
                position,
            )

        return orders

    def trade_product(self, state, product, order_depth, history):
        orders = []

        params = self.get_params(product)
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return orders, history

        history.append(mid)
        max_len = params["history"]
        if len(history) > max_len:
            history = history[-max_len:]

        if len(history) < params["min_history"]:
            return orders, history

        mean, std = self.mean_std(history)
        if mean is None or std is None or std < 1e-6:
            return orders, history

        z = (mid - mean) / std

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders, history

        position = state.position.get(product, 0)

        inventory_skew = params["inventory_skew"]
        adjusted_fair = mean - inventory_skew * position

        # Extra skew near limits
        if position >= 8:
            adjusted_fair -= 4
        elif position <= -8:
            adjusted_fair += 4

        buy_edge = adjusted_fair - best_ask
        sell_edge = best_bid - adjusted_fair

        entry_z = params["entry_z"]
        take_edge = params["take_edge"]
        passive_edge = params["passive_edge"]

        # 1. If price has reverted, reduce inventory first.
        close_orders = self.close_if_reverted(
            product,
            order_depth,
            adjusted_fair,
            z,
            position,
            params,
        )
        if close_orders:
            return close_orders, history

        # 2. Mean reversion active taking
        if z < -entry_z and buy_edge > take_edge:
            orders += self.sweep_buy(
                product,
                order_depth,
                adjusted_fair,
                position,
                params["take_size"],
            )
            if orders:
                return orders, history

        if z > entry_z and sell_edge > take_edge:
            orders += self.sweep_sell(
                product,
                order_depth,
                adjusted_fair,
                position,
                params["take_size"],
            )
            if orders:
                return orders, history

        # 3. Passive directional market making around adjusted fair
        passive_buy_price = best_bid + 1
        if passive_buy_price >= best_ask:
            passive_buy_price = best_bid

        passive_sell_price = best_ask - 1
        if passive_sell_price <= best_bid:
            passive_sell_price = best_ask

        passive_buy_edge = adjusted_fair - passive_buy_price
        passive_sell_edge = passive_sell_price - adjusted_fair

        # Directional passive quotes:
        # If current mid is below mean, we prefer buying.
        # If current mid is above mean, we prefer selling.
        if z < -0.3 and passive_buy_edge > passive_edge:
            self.add_buy_order(
                orders,
                product,
                passive_buy_price,
                params["passive_size"],
                position,
            )

        elif z > 0.3 and passive_sell_edge > passive_edge:
            self.add_sell_order(
                orders,
                product,
                passive_sell_price,
                params["passive_size"],
                position,
            )

        return orders, history

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product in self.PRODUCTS:
            result[product] = []

        # Load state
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        if "histories" not in data:
            data["histories"] = {}

        histories = data["histories"]

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            if product not in histories:
                histories[product] = []

            orders, new_history = self.trade_product(
                state,
                product,
                state.order_depths[product],
                histories[product],
            )

            result[product] = orders
            histories[product] = new_history

        # Save only necessary history
        data["histories"] = histories

        traderData = json.dumps(data)

        return result, 0, traderData