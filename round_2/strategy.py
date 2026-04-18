from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    # =========================
    # product configs
    # =========================
    PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # ---------- ASH_COATED_OSMIUM params ----------
    ASH_HISTORY_LENGTH = 3
    ASH_ENTRY_Z = 0.6
    ASH_EXIT_Z = 0.5
    ASH_BASE_QUOTE_OFFSET = 1
    ASH_PASSIVE_ORDER_SIZE = 30
    ASH_MAX_TAKE_SIZE = 30
    ASH_INVENTORY_SKEW = 1.5

    # ---------- INTARIAN_PEPPER_ROOT params ----------
    PEPPER_CORE_HOLD = 76
    PEPPER_MIN_SPREAD = 18
    PEPPER_TRADE_CLIP = 2
    PEPPER_QUOTE_IMPROVEMENT = 1

    def __init__(self):
        # runtime state reset every tick
        self.positions: Dict[str, int] = {}
        self.buy_orders_sent: Dict[str, int] = {}
        self.sell_orders_sent: Dict[str, int] = {}

        # persistent state
        self.ash_mid_history: List[float] = []

    # =========================
    # generic helpers
    # =========================
    def get_position_limit(self, product: str) -> int:
        return self.POSITION_LIMITS.get(product, 20)

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def remaining_buy_capacity(self, product: str) -> int:
        limit = self.get_position_limit(product)
        pos = self.positions.get(product, 0)
        sent = self.buy_orders_sent.get(product, 0)
        return max(0, limit - pos - sent)

    def remaining_sell_capacity(self, product: str) -> int:
        limit = self.get_position_limit(product)
        pos = self.positions.get(product, 0)
        sent = self.sell_orders_sent.get(product, 0)
        return max(0, limit + pos - sent)

    def send_buy_order(
        self, orders: List[Order], product: str, price: int, amount: int
    ) -> None:
        size = min(amount, self.remaining_buy_capacity(product))
        if size > 0:
            orders.append(Order(product, int(price), int(size)))
            self.buy_orders_sent[product] += int(size)

    def send_sell_order(
        self, orders: List[Order], product: str, price: int, amount: int
    ) -> None:
        size = min(amount, self.remaining_sell_capacity(product))
        if size > 0:
            orders.append(Order(product, int(price), -int(size)))
            self.sell_orders_sent[product] += int(size)

    def get_best_bid_ask(
        self, order_depth: OrderDepth
    ) -> Tuple[Optional[int], Optional[int]]:
        best_bid = (
            max(order_depth.buy_orders.keys())
            if order_depth.buy_orders
            else None
        )
        best_ask = (
            min(order_depth.sell_orders.keys())
            if order_depth.sell_orders
            else None
        )
        return best_bid, best_ask

    # =========================
    # persistence
    # =========================
    def load_state(self, state: TradingState) -> None:
        self.ash_mid_history = []

        if not state.traderData:
            return

        try:
            saved = json.loads(state.traderData)
            self.ash_mid_history = saved.get("ash_mid_history", [])
        except Exception:
            self.ash_mid_history = []

    def save_state(self) -> str:
        data = {
            "ash_mid_history": self.ash_mid_history[-self.ASH_HISTORY_LENGTH :]
        }
        return json.dumps(data)

    # =========================
    # ASH_COATED_OSMIUM logic
    # =========================
    def get_ash_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2

        if self.ash_mid_history:
            return self.ash_mid_history[-1]

        return None

    def update_ash_mid_history(self, mid: float) -> None:
        self.ash_mid_history.append(mid)
        if len(self.ash_mid_history) > self.ASH_HISTORY_LENGTH:
            self.ash_mid_history.pop(0)

    def get_ash_mean_std(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.ash_mid_history:
            return None, None

        mean = sum(self.ash_mid_history) / len(self.ash_mid_history)
        var = sum((x - mean) ** 2 for x in self.ash_mid_history) / len(
            self.ash_mid_history
        )
        return mean, math.sqrt(max(var, 0.0))

    def get_ash_zscore(self, mid: float) -> Tuple[float, float]:
        mean, std = self.get_ash_mean_std()
        if mean is None or std is None or std < 1e-6:
            return 0.0, mid
        return (mid - mean) / std, mean

    def ash_take_exit_orders(
        self,
        order_depth: OrderDepth,
        orders: List[Order],
        mean: float,
        z: float,
    ) -> None:
        product = "ASH_COATED_OSMIUM"
        position = self.positions.get(product, 0)

        if position > 0 and z >= -self.ASH_EXIT_Z:
            for bid, vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if bid >= mean and self.remaining_sell_capacity(product) > 0:
                    size = min(
                        vol,
                        position,
                        self.remaining_sell_capacity(product),
                        self.ASH_MAX_TAKE_SIZE,
                    )
                    if size > 0:
                        self.send_sell_order(orders, product, bid, size)
                    break

        elif position < 0 and z <= self.ASH_EXIT_Z:
            for ask, vol in sorted(order_depth.sell_orders.items()):
                ask_vol = -vol
                if ask <= mean and self.remaining_buy_capacity(product) > 0:
                    size = min(
                        ask_vol,
                        -position,
                        self.remaining_buy_capacity(product),
                        self.ASH_MAX_TAKE_SIZE,
                    )
                    if size > 0:
                        self.send_buy_order(orders, product, ask, size)
                    break

    def ash_take_mean_reversion_entries(
        self,
        order_depth: OrderDepth,
        orders: List[Order],
        mean: float,
        z: float,
    ) -> None:
        product = "ASH_COATED_OSMIUM"

        if z < -self.ASH_ENTRY_Z:
            taken = 0
            for ask, vol in sorted(order_depth.sell_orders.items()):
                ask_vol = -vol
                if (
                    ask <= mean
                    and self.remaining_buy_capacity(product) > 0
                    and taken < self.ASH_MAX_TAKE_SIZE
                ):
                    size = min(
                        ask_vol,
                        self.remaining_buy_capacity(product),
                        self.ASH_MAX_TAKE_SIZE - taken,
                    )
                    if size > 0:
                        self.send_buy_order(orders, product, ask, size)
                        taken += size

        elif z > self.ASH_ENTRY_Z:
            taken = 0
            for bid, vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if (
                    bid >= mean
                    and self.remaining_sell_capacity(product) > 0
                    and taken < self.ASH_MAX_TAKE_SIZE
                ):
                    size = min(
                        vol,
                        self.remaining_sell_capacity(product),
                        self.ASH_MAX_TAKE_SIZE - taken,
                    )
                    if size > 0:
                        self.send_sell_order(orders, product, bid, size)
                        taken += size

    def ash_place_passive_quotes(
        self,
        order_depth: OrderDepth,
        orders: List[Order],
        mean: float,
        z: float,
    ) -> None:
        product = "ASH_COATED_OSMIUM"
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_bid is None and best_ask is None:
            return

        fair = mean
        position = self.positions.get(product, 0)
        pos_ratio = position / self.get_position_limit(product)
        skew = self.ASH_INVENTORY_SKEW * pos_ratio

        bid_offset = self.ASH_BASE_QUOTE_OFFSET
        ask_offset = self.ASH_BASE_QUOTE_OFFSET

        if z > self.ASH_EXIT_Z:
            bid_offset += 1
        elif z < -self.ASH_EXIT_Z:
            ask_offset += 1

        bid_price = int(fair - bid_offset - skew)
        ask_price = int(fair + ask_offset - skew)

        if best_bid is not None:
            bid_price = min(best_bid + 1, bid_price)
        if best_ask is not None:
            ask_price = max(best_ask - 1, ask_price)

        if best_ask is not None:
            bid_price = min(bid_price, best_ask - 1)
        if best_bid is not None:
            ask_price = max(ask_price, best_bid + 1)

        if bid_price < ask_price:
            buy_size = min(
                self.ASH_PASSIVE_ORDER_SIZE,
                self.remaining_buy_capacity(product),
            )
            sell_size = min(
                self.ASH_PASSIVE_ORDER_SIZE,
                self.remaining_sell_capacity(product),
            )

            if position > 40:
                buy_size = min(buy_size, 6)
                sell_size = min(self.remaining_sell_capacity(product), 12)
            elif position < -40:
                sell_size = min(sell_size, 6)
                buy_size = min(self.remaining_buy_capacity(product), 12)

            if buy_size > 0:
                self.send_buy_order(orders, product, bid_price, buy_size)
            if sell_size > 0:
                self.send_sell_order(orders, product, ask_price, sell_size)

    def trade_ash(self, state: TradingState, orders: List[Order]) -> None:
        product = "ASH_COATED_OSMIUM"
        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        mid = self.get_ash_mid_price(order_depth)
        if mid is None:
            return

        self.update_ash_mid_history(mid)
        z, mean = self.get_ash_zscore(mid)

        self.ash_take_exit_orders(order_depth, orders, mean, z)
        self.ash_take_mean_reversion_entries(order_depth, orders, mean, z)
        self.ash_place_passive_quotes(order_depth, orders, mean, z)

    # =========================
    # INTARIAN_PEPPER_ROOT logic
    # =========================
    def pepper_add_wide_spread_passive_quotes(
        self,
        product: str,
        order_depth: OrderDepth,
        orders: List[Order],
        current_position: int,
        limit: int,
    ) -> None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread = best_ask - best_bid

        if spread < self.PEPPER_MIN_SPREAD:
            return

        pending_buys = sum(
            order.quantity for order in orders if order.quantity > 0
        )
        pending_sells = -sum(
            order.quantity for order in orders if order.quantity < 0
        )

        working_position = current_position + pending_buys - pending_sells
        core_target = min(limit, self.PEPPER_CORE_HOLD)

        reserve_inventory = max(0, working_position - core_target)
        reserve_capacity = max(0, limit - working_position)

        if reserve_inventory > 0:
            sell_size = min(self.PEPPER_TRADE_CLIP, reserve_inventory)
            sell_price = best_ask - self.PEPPER_QUOTE_IMPROVEMENT
            if sell_size > 0 and sell_price > best_bid:
                self.send_sell_order(orders, product, sell_price, sell_size)

        elif reserve_capacity > 0:
            buy_size = min(self.PEPPER_TRADE_CLIP, reserve_capacity)
            buy_price = best_bid + self.PEPPER_QUOTE_IMPROVEMENT
            if buy_size > 0 and buy_price < best_ask:
                self.send_buy_order(orders, product, buy_price, buy_size)

    def trade_pepper(self, state: TradingState, orders: List[Order]) -> None:
        product = "INTARIAN_PEPPER_ROOT"
        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        current_position = self.positions.get(product, 0)
        limit = self.get_position_limit(product)
        max_buy = self.remaining_buy_capacity(product)

        # Aggressive + passive accumulation toward limit
        if current_position < limit and max_buy > 0:
            remaining_to_buy = max_buy

            # use integer prices only
            acceptable_ask = best_ask

            for ask_price in sorted(order_depth.sell_orders.keys()):
                if remaining_to_buy <= 0:
                    break
                if ask_price > acceptable_ask:
                    break

                available = -order_depth.sell_orders[ask_price]
                vol = min(remaining_to_buy, available)
                if vol > 0:
                    self.send_buy_order(orders, product, ask_price, vol)
                    remaining_to_buy -= vol

            if remaining_to_buy > 0:
                passive_bid_price = best_bid + 1
                if passive_bid_price < best_ask:
                    self.send_buy_order(
                        orders, product, passive_bid_price, remaining_to_buy
                    )

        self.pepper_add_wide_spread_passive_quotes(
            product,
            order_depth,
            orders,
            current_position,
            limit,
        )

    # =========================
    # main
    # =========================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {
            product: [] for product in self.PRODUCTS
        }

        # reset per-tick counters
        self.positions = {
            product: self.get_product_pos(state, product)
            for product in self.PRODUCTS
        }
        self.buy_orders_sent = {product: 0 for product in self.PRODUCTS}
        self.sell_orders_sent = {product: 0 for product in self.PRODUCTS}

        # load persistent data
        self.load_state(state)

        # run each strategy independently
        self.trade_ash(state, result["ASH_COATED_OSMIUM"])
        self.trade_pepper(state, result["INTARIAN_PEPPER_ROOT"])

        conversions = 0
        traderData = self.save_state()
        return result, conversions, traderData
