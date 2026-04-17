from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    ROOT_PRODUCT = "INTARIAN_PEPPER_ROOT"
    OSMIUM_PRODUCT = "ASH_COATED_OSMIUM"

    POSITION_LIMITS = {
        ROOT_PRODUCT: 80,
        OSMIUM_PRODUCT: 80,
    }

    HISTORY_LENGTH = 30
    ENTRY_Z = 0.8
    EXIT_Z = 0.4

    BASE_QUOTE_OFFSET = 1
    INVENTORY_SKEW = 0.05

    SINGLE_SIDE_ORDER_SIZE = 5
    SINGLE_SIDE_EDGE = 1.0
    QUOTE_SIZE = 15

    def __init__(self):
        self.osmium_position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.mid_history: List[float] = []

    def send_sell_order(
        self, orders: List[Order], product: str, price: int, amount: int
    ) -> None:
        if amount >= 0:
            amount = -abs(amount)
        orders.append(Order(product, int(price), int(amount)))

    def send_buy_order(
        self, orders: List[Order], product: str, price: int, amount: int
    ) -> None:
        if amount <= 0:
            amount = abs(amount)
        orders.append(Order(product, int(price), int(amount)))

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def remaining_buy_capacity(self) -> int:
        limit = self.POSITION_LIMITS[self.OSMIUM_PRODUCT]
        return max(0, limit - self.osmium_position)

    def remaining_sell_capacity(self) -> int:
        limit = self.POSITION_LIMITS[self.OSMIUM_PRODUCT]
        return max(0, limit + self.osmium_position)

    def load_history(self, state: TradingState) -> None:
        if self.mid_history or not state.traderData:
            return

        try:
            saved = json.loads(state.traderData)
            self.mid_history = saved.get("mid_history", [])
        except Exception:
            self.mid_history = []

    def save_trader_data(self, last_mid_prices: Dict[str, float]) -> str:
        return json.dumps(
            {
                "mid_history": self.mid_history[-self.HISTORY_LENGTH :],
                "last_mid_prices": last_mid_prices,
            }
        )

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (
            max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())
        ) / 2

    def update_mid_history(self, mid: float) -> None:
        self.mid_history.append(mid)
        if len(self.mid_history) > self.HISTORY_LENGTH:
            self.mid_history.pop(0)

    def get_mean_std(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.mid_history:
            return None, None
        mean = sum(self.mid_history) / len(self.mid_history)
        var = sum((x - mean) ** 2 for x in self.mid_history) / len(self.mid_history)
        return mean, math.sqrt(var)

    def get_zscore(self, mid: float) -> Tuple[float, float]:
        mean, std = self.get_mean_std()
        if mean is None or std is None or std < 1e-6:
            return 0.0, mid
        return (mid - mean) / std, mean

    def inventory_adjusted_fair(self, fair: float) -> float:
        return fair - self.INVENTORY_SKEW * self.osmium_position

    def handle_single_sided_book(
        self, order_depth: OrderDepth, orders: List[Order]
    ) -> bool:
        has_bids = bool(order_depth.buy_orders)
        has_asks = bool(order_depth.sell_orders)

        if not has_bids and not has_asks:
            return True

        mean, _ = self.get_mean_std()

        if not has_bids and has_asks:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = -order_depth.sell_orders[best_ask]

            if mean is not None and best_ask <= mean - self.SINGLE_SIDE_EDGE:
                size = min(
                    self.SINGLE_SIDE_ORDER_SIZE,
                    best_ask_vol,
                    self.remaining_buy_capacity(),
                )
                if size > 0:
                    self.send_buy_order(
                        orders, self.OSMIUM_PRODUCT, best_ask, size
                    )
            return True

        if has_bids and not has_asks:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]

            if mean is not None and best_bid >= mean + self.SINGLE_SIDE_EDGE:
                size = min(
                    self.SINGLE_SIDE_ORDER_SIZE,
                    best_bid_vol,
                    self.remaining_sell_capacity(),
                )
                if size > 0:
                    self.send_sell_order(
                        orders, self.OSMIUM_PRODUCT, best_bid, -size
                    )
            return True

        return False

    def trade_osmium(self, state: TradingState, orders: List[Order]) -> None:
        order_depth = state.order_depths[self.OSMIUM_PRODUCT]

        if self.handle_single_sided_book(order_depth, orders):
            return

        mid = self.get_mid_price(order_depth)
        if mid is None:
            return

        self.update_mid_history(mid)
        z_score, mean = self.get_zscore(mid)
        fair = self.inventory_adjusted_fair(mean)

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        if z_score < -self.ENTRY_Z:
            buy_capacity = self.remaining_buy_capacity()
            for ask, volume in sorted(order_depth.sell_orders.items()):
                if buy_capacity <= 0:
                    break
                available = -volume
                if ask <= fair:
                    size = min(available, buy_capacity)
                    if size > 0:
                        self.send_buy_order(
                            orders, self.OSMIUM_PRODUCT, ask, size
                        )
                        buy_capacity -= size
        elif z_score > self.ENTRY_Z:
            sell_capacity = self.remaining_sell_capacity()
            for bid, volume in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if sell_capacity <= 0:
                    break
                available = volume
                if bid >= fair:
                    size = min(available, sell_capacity)
                    if size > 0:
                        self.send_sell_order(
                            orders, self.OSMIUM_PRODUCT, bid, -size
                        )
                        sell_capacity -= size

        buy_capacity = self.remaining_buy_capacity()
        sell_capacity = self.remaining_sell_capacity()

        bid_price = min(best_bid + 1, int(fair - self.BASE_QUOTE_OFFSET))
        ask_price = max(best_ask - 1, int(fair + self.BASE_QUOTE_OFFSET))

        if bid_price < ask_price:
            buy_size = min(self.QUOTE_SIZE, buy_capacity)
            sell_size = min(self.QUOTE_SIZE, sell_capacity)

            if buy_size > 0:
                self.send_buy_order(
                    orders, self.OSMIUM_PRODUCT, bid_price, buy_size
                )
            if sell_size > 0:
                self.send_sell_order(
                    orders, self.OSMIUM_PRODUCT, ask_price, -sell_size
                )

    def trade_pepper_root(
        self, state: TradingState, order_depth: OrderDepth, orders: List[Order]
    ) -> None:
        product = self.ROOT_PRODUCT
        current_position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        max_buy = limit - current_position

        if current_position < limit and max_buy > 0 and order_depth.sell_orders:
            remaining = max_buy
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                volume = min(remaining, -order_depth.sell_orders[ask_price])
                orders.append(Order(product, ask_price, volume))
                remaining -= volume

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        last_mid_prices: Dict[str, float] = {}

        self.load_history(state)

        self.osmium_position = self.get_product_pos(state, self.OSMIUM_PRODUCT)
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            mid_price = self.get_mid_price(order_depth)
            if mid_price is not None:
                last_mid_prices[product] = mid_price

            if product == self.ROOT_PRODUCT:
                self.trade_pepper_root(state, order_depth, orders)
            elif product == self.OSMIUM_PRODUCT:
                self.trade_osmium(state, orders)

            result[product] = orders

        conversions = 0
        trader_data = self.save_trader_data(last_mid_prices)
        return result, conversions, trader_data