from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    # =========================
    # PEPPER PARAMETERS
    # =========================
    PEPPER_CORE_HOLD = 76
    PEPPER_MIN_SPREAD = 18
    PEPPER_TRADE_CLIP = 2
    PEPPER_QUOTE_IMPROVEMENT = 1

    # =========================
    # OSMIUM PARAMETERS
    # =========================
    OSMIUM_PRODUCT = "ASH_COATED_OSMIUM"
    OSMIUM_POSITION_LIMIT = 80

    # --- signal ---
    HISTORY_LENGTH = 3
    ENTRY_Z = 0.6
    EXIT_Z = 0.5

    # --- quoting / execution ---
    BASE_QUOTE_OFFSET = 1
    PASSIVE_ORDER_SIZE = 30
    MAX_TAKE_SIZE = 30
    INVENTORY_SKEW = 1.5  # max skew in ticks at full position

    def __init__(self):
        # Osmium state
        self.position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.mid_history = []

    def bid(self):
        return 15

    # =========================
    # PEPPER LOGIC
    # =========================
    def add_wide_spread_passive_quotes(
        self,
        product: str,
        order_depth: OrderDepth,
        orders: List[Order],
        current_position: int,
        limit: int,
    ) -> None:
        if product != "INTARIAN_PEPPER_ROOT":
            return
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

        # Quote only one side at a time and only inside a genuinely wide spread
        if reserve_inventory > 0:
            sell_size = min(self.PEPPER_TRADE_CLIP, reserve_inventory)
            sell_price = best_ask - self.PEPPER_QUOTE_IMPROVEMENT
            if sell_size > 0 and sell_price > best_bid:
                orders.append(Order(product, sell_price, -sell_size))
        elif reserve_capacity > 0:
            buy_size = min(self.PEPPER_TRADE_CLIP, reserve_capacity)
            buy_price = best_bid + self.PEPPER_QUOTE_IMPROVEMENT
            if buy_size > 0 and buy_price < best_ask:
                orders.append(Order(product, buy_price, buy_size))

    # =========================
    # OSMIUM HELPERS
    # =========================
    def send_sell_order(
        self, orders: List[Order], product: str, price: int, amount: int
    ):
        size = min(amount, self.remaining_sell_capacity())
        if size > 0:
            orders.append(Order(product, int(price), -int(size)))
            self.sell_orders_sent += int(size)

    def send_buy_order(
        self, orders: List[Order], product: str, price: int, amount: int
    ):
        size = min(amount, self.remaining_buy_capacity())
        if size > 0:
            orders.append(Order(product, int(price), int(size)))
            self.buy_orders_sent += int(size)

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def remaining_buy_capacity(self) -> int:
        return max(
            0,
            self.OSMIUM_POSITION_LIMIT - self.position - self.buy_orders_sent,
        )

    def remaining_sell_capacity(self) -> int:
        return max(
            0,
            self.OSMIUM_POSITION_LIMIT + self.position - self.sell_orders_sent,
        )

    # =========================
    # OSMIUM PERSISTENCE
    # =========================
    def load_history(self, state: TradingState):
        if self.mid_history:
            return
        if not state.traderData:
            return
        try:
            saved = json.loads(state.traderData)

            # Preferred merged format
            if isinstance(saved, dict) and "osmium_mid_history" in saved:
                self.mid_history = saved.get("osmium_mid_history", [])
            # Backward-compatible with standalone osmium file
            elif isinstance(saved, dict) and "mid_history" in saved:
                self.mid_history = saved.get("mid_history", [])
            else:
                self.mid_history = []
        except Exception:
            self.mid_history = []

    def save_state(self, pepper_data: Dict[str, float]):
        payload = {
            "pepper_mid_prices": pepper_data,
            "osmium_mid_history": self.mid_history[-self.HISTORY_LENGTH :],
        }
        return json.dumps(payload)

    # =========================
    # OSMIUM PRICE / SIGNAL
    # =========================
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

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2

        if self.mid_history:
            return self.mid_history[-1]

        return None

    def update_mid_history(self, mid: float):
        self.mid_history.append(mid)
        if len(self.mid_history) > self.HISTORY_LENGTH:
            self.mid_history.pop(0)

    def get_mean_std(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.mid_history:
            return None, None
        mean = sum(self.mid_history) / len(self.mid_history)
        var = sum((x - mean) ** 2 for x in self.mid_history) / len(
            self.mid_history
        )
        return mean, math.sqrt(max(var, 0.0))

    def get_zscore(self, mid: float) -> Tuple[float, float]:
        mean, std = self.get_mean_std()
        if mean is None or std is None or std < 1e-6:
            return 0.0, mid
        return (mid - mean) / std, mean

    # =========================
    # OSMIUM EXECUTION LOGIC
    # =========================
    def take_mean_reversion_entries(
        self,
        order_depth: OrderDepth,
        orders: List[Order],
        mean: float,
        z: float,
    ):
        if z < -self.ENTRY_Z:
            taken = 0
            for ask, vol in sorted(order_depth.sell_orders.items()):
                ask_vol = -vol
                if (
                    ask <= mean
                    and self.remaining_buy_capacity() > 0
                    and taken < self.MAX_TAKE_SIZE
                ):
                    size = min(
                        ask_vol,
                        self.remaining_buy_capacity(),
                        self.MAX_TAKE_SIZE - taken,
                    )
                    if size > 0:
                        self.send_buy_order(
                            orders, self.OSMIUM_PRODUCT, ask, size
                        )
                        taken += size

        elif z > self.ENTRY_Z:
            taken = 0
            for bid, vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                bid_vol = vol
                if (
                    bid >= mean
                    and self.remaining_sell_capacity() > 0
                    and taken < self.MAX_TAKE_SIZE
                ):
                    size = min(
                        bid_vol,
                        self.remaining_sell_capacity(),
                        self.MAX_TAKE_SIZE - taken,
                    )
                    if size > 0:
                        self.send_sell_order(
                            orders, self.OSMIUM_PRODUCT, bid, size
                        )
                        taken += size

    def take_exit_orders(
        self,
        order_depth: OrderDepth,
        orders: List[Order],
        mean: float,
        z: float,
    ):
        if self.position > 0 and z >= -self.EXIT_Z:
            for bid, vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if bid >= mean and self.remaining_sell_capacity() > 0:
                    size = min(
                        vol,
                        self.position,
                        self.remaining_sell_capacity(),
                        self.MAX_TAKE_SIZE,
                    )
                    if size > 0:
                        self.send_sell_order(
                            orders, self.OSMIUM_PRODUCT, bid, size
                        )
                    break

        elif self.position < 0 and z <= self.EXIT_Z:
            for ask, vol in sorted(order_depth.sell_orders.items()):
                ask_vol = -vol
                if ask <= mean and self.remaining_buy_capacity() > 0:
                    size = min(
                        ask_vol,
                        -self.position,
                        self.remaining_buy_capacity(),
                        self.MAX_TAKE_SIZE,
                    )
                    if size > 0:
                        self.send_buy_order(
                            orders, self.OSMIUM_PRODUCT, ask, size
                        )
                    break

    def place_passive_quotes(
        self,
        order_depth: OrderDepth,
        orders: List[Order],
        mean: float,
        z: float,
    ):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_bid is None and best_ask is None:
            return

        fair = mean

        pos_ratio = self.position / self.OSMIUM_POSITION_LIMIT
        skew = self.INVENTORY_SKEW * pos_ratio

        bid_offset = self.BASE_QUOTE_OFFSET
        ask_offset = self.BASE_QUOTE_OFFSET

        if z > self.EXIT_Z:
            bid_offset += 1
        elif z < -self.EXIT_Z:
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
                self.PASSIVE_ORDER_SIZE, self.remaining_buy_capacity()
            )
            sell_size = min(
                self.PASSIVE_ORDER_SIZE, self.remaining_sell_capacity()
            )

            if self.position > 40:
                buy_size = min(buy_size, 6)
                sell_size = min(self.remaining_sell_capacity(), 12)
            elif self.position < -40:
                sell_size = min(sell_size, 6)
                buy_size = min(self.remaining_buy_capacity(), 12)

            if buy_size > 0:
                self.send_buy_order(
                    orders, self.OSMIUM_PRODUCT, bid_price, buy_size
                )
            if sell_size > 0:
                self.send_sell_order(
                    orders, self.OSMIUM_PRODUCT, ask_price, sell_size
                )

    def trade_osmium(self, state: TradingState, orders: List[Order]):
        od = state.order_depths[self.OSMIUM_PRODUCT]
        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_mid_history(mid)
        z, mean = self.get_zscore(mid)

        self.take_exit_orders(od, orders, mean, z)
        self.take_mean_reversion_entries(od, orders, mean, z)
        self.place_passive_quotes(od, orders, mean, z)

    # =========================
    # MAIN
    # =========================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        pepper_data: Dict[str, float] = {}

        # Reset osmium order counters each run
        self.position = self.get_product_pos(state, self.OSMIUM_PRODUCT)
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.load_history(state)

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            if (
                len(order_depth.buy_orders) == 0
                or len(order_depth.sell_orders) == 0
            ):
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            pepper_data[product] = mid_price

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            max_buy = limit - current_position

            # ---------------- INTARIAN_PEPPER_ROOT ----------------
            if product == "INTARIAN_PEPPER_ROOT":
                if current_position < limit and max_buy > 0:
                    remaining_to_buy = max_buy

                    # 1. AGGRESSIVE BUYING (with a price cap)
                    acceptable_ask = best_ask + 0.75

                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if remaining_to_buy <= 0:
                            break
                        if ask_price > acceptable_ask:
                            break

                        vol = min(
                            remaining_to_buy,
                            -order_depth.sell_orders[ask_price],
                        )
                        orders.append(Order(product, ask_price, vol))
                        remaining_to_buy -= vol

                    # 2. PASSIVE BUYING
                    if remaining_to_buy > 0:
                        passive_bid_price = best_ask - 0.75
                        orders.append(
                            Order(product, passive_bid_price, remaining_to_buy)
                        )

                self.add_wide_spread_passive_quotes(
                    product,
                    order_depth,
                    orders,
                    current_position,
                    limit,
                )

            # ---------------- ASH_COATED_OSMIUM ----------------
            elif product == self.OSMIUM_PRODUCT:
                self.trade_osmium(state, orders)

            result[product] = orders

        traderData = self.save_state(pepper_data)
        conversions = 0
        return result, conversions, traderData
