from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    HISTORY_LENGTH = 20
    ENTRY_Z = 0.8
    EXIT_Z = 0.4

    BASE_QUOTE_OFFSET = 1
    INVENTORY_SKEW = 0.05

    # Single-sided book behavior
    SINGLE_SIDE_ORDER_SIZE = 5
    SINGLE_SIDE_EDGE = 1.0   # only act if single visible price is this far from rolling mean

    # Regular market making size
    QUOTE_SIZE = 15

    def __init__(self):
        self.position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.mid_history: List[float] = []

    def send_sell_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        if amount >= 0:
            amount = -abs(amount)
        orders.append(Order(product, int(price), int(amount)))

    def send_buy_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        if amount <= 0:
            amount = abs(amount)
        orders.append(Order(product, int(price), int(amount)))

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def remaining_buy_capacity(self) -> int:
        return max(0, self.POSITION_LIMIT - self.position)

    def remaining_sell_capacity(self) -> int:
        return max(0, self.POSITION_LIMIT + self.position)

    def load_history(self, state: TradingState) -> None:
        if self.mid_history:
            return
        if not state.traderData:
            return
        try:
            saved = json.loads(state.traderData)
            self.mid_history = saved.get("mid_history", [])
        except Exception:
            self.mid_history = []

    def save_history(self) -> str:
        return json.dumps({"mid_history": self.mid_history[-self.HISTORY_LENGTH:]})

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

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
        return fair - self.INVENTORY_SKEW * self.position

    def handle_single_sided_book(self, od: OrderDepth, orders: List[Order]) -> bool:
        """
        Returns True if a single-sided-book branch was handled and we should stop.
        Returns False if book is normal two-sided and main strategy should continue.
        """
        has_bids = bool(od.buy_orders)
        has_asks = bool(od.sell_orders)

        if not has_bids and not has_asks:
            return True

        mean, _ = self.get_mean_std()

        # Only asks visible: market may be too cheap. Consider buying if ask is attractive.
        if not has_bids and has_asks:
            best_ask = min(od.sell_orders.keys())
            best_ask_vol = -od.sell_orders[best_ask]  # ask vols are usually negative in IMC data

            if mean is not None and best_ask <= mean - self.SINGLE_SIDE_EDGE:
                size = min(self.SINGLE_SIDE_ORDER_SIZE, best_ask_vol, self.remaining_buy_capacity())
                if size > 0:
                    self.send_buy_order(orders, self.PRODUCT, best_ask, size)
            return True

        # Only bids visible: market may be too expensive. Consider selling if bid is attractive.
        if has_bids and not has_asks:
            best_bid = max(od.buy_orders.keys())
            best_bid_vol = od.buy_orders[best_bid]

            if mean is not None and best_bid >= mean + self.SINGLE_SIDE_EDGE:
                size = min(self.SINGLE_SIDE_ORDER_SIZE, best_bid_vol, self.remaining_sell_capacity())
                if size > 0:
                    self.send_sell_order(orders, self.PRODUCT, best_bid, -size)
            return True

        return False

    def trade(self, state: TradingState, orders: List[Order]) -> None:
        od = state.order_depths[self.PRODUCT]

        # Handle single-sided / empty book first
        if self.handle_single_sided_book(od, orders):
            return

        # From this point onward we know both sides exist
        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_mid_history(mid)
        z, mean = self.get_zscore(mid)
        fair = self.inventory_adjusted_fair(mean)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        # Mean reversion taker logic
        if z < -self.ENTRY_Z:
            buy_capacity = self.remaining_buy_capacity()
            for ask, vol in sorted(od.sell_orders.items()):
                if buy_capacity <= 0:
                    break
                available = -vol
                if ask <= fair:
                    size = min(available, buy_capacity)
                    if size > 0:
                        self.send_buy_order(orders, self.PRODUCT, ask, size)
                        buy_capacity -= size

        elif z > self.ENTRY_Z:
            sell_capacity = self.remaining_sell_capacity()
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if sell_capacity <= 0:
                    break
                available = vol
                if bid >= fair:
                    size = min(available, sell_capacity)
                    if size > 0:
                        self.send_sell_order(orders, self.PRODUCT, bid, -size)
                        sell_capacity -= size

        # Passive market making around inventory-adjusted fair
        buy_capacity = self.remaining_buy_capacity()
        sell_capacity = self.remaining_sell_capacity()

        bid_price = min(best_bid + 1, int(fair - self.BASE_QUOTE_OFFSET))
        ask_price = max(best_ask - 1, int(fair + self.BASE_QUOTE_OFFSET))

        # Avoid crossing ourselves accidentally
        if bid_price < ask_price:
            buy_size = min(self.QUOTE_SIZE, buy_capacity)
            sell_size = min(self.QUOTE_SIZE, sell_capacity)

            if buy_size > 0:
                self.send_buy_order(orders, self.PRODUCT, bid_price, buy_size)
            if sell_size > 0:
                self.send_sell_order(orders, self.PRODUCT, ask_price, -sell_size)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {self.PRODUCT: []}

        self.position = self.get_product_pos(state, self.PRODUCT)
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0

        self.load_history(state)

        if self.PRODUCT in state.order_depths:
            self.trade(state, result[self.PRODUCT])

        return result, 0, self.save_history()