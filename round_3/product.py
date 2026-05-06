from datamodel import TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    # ==========================================================================
    # Config
    # ==========================================================================
    PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]

    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
    }

    # ── Mean reversion params ──────────────────────────────────────────────────
    # Rolling window for mean and std estimation (Bollinger Band style)
    WINDOW = 50

    # EWMA decay for fast-tracking fair value used in passive quoting
    EWMA_ALPHA = 0.1   # weight on latest price; 1-EWMA_ALPHA on previous EWMA

    # Entry: open a position when |z| exceeds this
    ENTRY_Z = 2.0
    # Exit: close position when |z| falls back below this
    EXIT_Z = 0.5

    # Passive quoting: always quote this many ticks inside the spread
    QUOTE_OFFSET = 1
    PASSIVE_SIZE = 30
    MAX_TAKE = 30

    # Inventory skew: shifts quotes to discourage accumulating one-sided risk
    SKEW = 1.5

    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.buy_orders_sent: Dict[str, int] = {}
        self.sell_orders_sent: Dict[str, int] = {}

        # Separate state per product
        self.mid_history: Dict[str, List[float]] = {
            "HYDROGEL_PACK": [],
            "VELVETFRUIT_EXTRACT": [],
        }
        self.ewma: Dict[str, Optional[float]] = {
            "HYDROGEL_PACK": None,
            "VELVETFRUIT_EXTRACT": None,
        }

    # ==========================================================================
    # Helpers
    # ==========================================================================
    def get_position_limit(self, product: str) -> int:
        return self.POSITION_LIMITS.get(product, 20)

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

    def send_buy_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        size = min(amount, self.remaining_buy_capacity(product))
        if size > 0:
            orders.append(Order(product, int(price), int(size)))
            self.buy_orders_sent[product] = self.buy_orders_sent.get(product, 0) + int(size)

    def send_sell_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        size = min(amount, self.remaining_sell_capacity(product))
        if size > 0:
            orders.append(Order(product, int(price), -int(size)))
            self.sell_orders_sent[product] = self.sell_orders_sent.get(product, 0) + int(size)

    def get_best_bid_ask(self, od) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    # ==========================================================================
    # Persistence
    # ==========================================================================
    def load_state(self, state: TradingState) -> None:
        for p in self.PRODUCTS:
            self.mid_history[p] = []
            self.ewma[p] = None

        if not state.traderData:
            return
        try:
            saved = json.loads(state.traderData)
            for p in self.PRODUCTS:
                self.mid_history[p] = saved.get(f"{p}_history", [])
                self.ewma[p] = saved.get(f"{p}_ewma", None)
        except Exception:
            pass

    def save_state(self) -> str:
        data = {}
        for p in self.PRODUCTS:
            data[f"{p}_history"] = self.mid_history[p][-self.WINDOW:]
            data[f"{p}_ewma"] = self.ewma[p]
        return json.dumps(data)

    # ==========================================================================
    # Mean reversion core
    # ==========================================================================
    def _rolling_mean_std(self, history: List[float]) -> Tuple[float, float]:
        """Rolling mean and std over the last WINDOW observations."""
        window = history[-self.WINDOW:]
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / len(window)
        return mean, math.sqrt(max(var, 0.0))

    def _update_ewma(self, product: str, mid: float) -> float:
        """Update and return the EWMA fair price."""
        prev = self.ewma.get(product)
        if prev is None:
            self.ewma[product] = mid
        else:
            self.ewma[product] = (1 - self.EWMA_ALPHA) * prev + self.EWMA_ALPHA * mid
        return self.ewma[product]

    def trade_product(self, product: str, state: TradingState, orders: List[Order]) -> None:
        if product not in state.order_depths:
            return
        od = state.order_depths[product]
        best_bid, best_ask = self.get_best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return

        mid = (best_bid + best_ask) / 2.0

        # Update histories
        self.mid_history[product].append(mid)
        if len(self.mid_history[product]) > self.WINDOW:
            self.mid_history[product].pop(0)
        fair = self._update_ewma(product, mid)

        # Need enough history for reliable z-score
        if len(self.mid_history[product]) < 10:
            return

        rolling_mean, rolling_std = self._rolling_mean_std(self.mid_history[product])

        # Z-score: how many std devs is current price from rolling mean
        z = (mid - rolling_mean) / rolling_std if rolling_std > 1e-6 else 0.0

        position = self.positions.get(product, 0)
        limit = self.get_position_limit(product)

        # ── EXIT: close position when price reverts toward mean (|z| < EXIT_Z) ──
        if position > 0 and z >= -self.EXIT_Z:
            # Was long (bought on dip) — sell as price recovers
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if self.remaining_sell_capacity(product) > 0:
                    size = min(vol, position, self.remaining_sell_capacity(product), self.MAX_TAKE)
                    if size > 0:
                        self.send_sell_order(orders, product, bid, size)
                    break

        elif position < 0 and z <= self.EXIT_Z:
            # Was short (sold on spike) — buy back as price falls
            for ask, vol in sorted(od.sell_orders.items()):
                if self.remaining_buy_capacity(product) > 0:
                    size = min(-vol, -position, self.remaining_buy_capacity(product), self.MAX_TAKE)
                    if size > 0:
                        self.send_buy_order(orders, product, ask, size)
                    break

        # ── ENTRY: trade aggressively when price is far from mean (|z| > ENTRY_Z) ──
        if z < -self.ENTRY_Z:
            # Price significantly below mean — buy, expecting reversion up
            taken = 0
            for ask, vol in sorted(od.sell_orders.items()):
                if self.remaining_buy_capacity(product) > 0 and taken < self.MAX_TAKE:
                    size = min(-vol, self.remaining_buy_capacity(product), self.MAX_TAKE - taken)
                    if size > 0:
                        self.send_buy_order(orders, product, ask, size)
                        taken += size

        elif z > self.ENTRY_Z:
            # Price significantly above mean — sell, expecting reversion down
            taken = 0
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if self.remaining_sell_capacity(product) > 0 and taken < self.MAX_TAKE:
                    size = min(vol, self.remaining_sell_capacity(product), self.MAX_TAKE - taken)
                    if size > 0:
                        self.send_sell_order(orders, product, bid, size)
                        taken += size

        # ── PASSIVE QUOTES: always quote around EWMA fair value ──────────────
        # Inventory skew shifts both quotes in the direction that reduces position
        pos_ratio = position / limit
        inv_skew = self.SKEW * pos_ratio

        bid_price = int(fair - self.QUOTE_OFFSET - inv_skew)
        ask_price = int(fair + self.QUOTE_OFFSET - inv_skew)

        # Clamp inside the current spread
        bid_price = min(bid_price, best_ask - 1)
        ask_price = max(ask_price, best_bid + 1)
        # Step inside the spread by 1 tick where possible
        bid_price = min(bid_price, best_bid + 1)
        ask_price = max(ask_price, best_ask - 1)

        if bid_price < ask_price:
            buy_sz  = min(self.PASSIVE_SIZE, self.remaining_buy_capacity(product))
            sell_sz = min(self.PASSIVE_SIZE, self.remaining_sell_capacity(product))

            # Reduce quote size on the side that would worsen inventory
            if position > limit * 0.5:
                buy_sz  = min(buy_sz, 6)
                sell_sz = min(self.remaining_sell_capacity(product), 12)
            elif position < -limit * 0.5:
                sell_sz = min(sell_sz, 6)
                buy_sz  = min(self.remaining_buy_capacity(product), 12)

            if buy_sz > 0:
                self.send_buy_order(orders, product, bid_price, buy_sz)
            if sell_sz > 0:
                self.send_sell_order(orders, product, ask_price, sell_sz)

    # ==========================================================================
    # Main
    # ==========================================================================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {p: [] for p in self.PRODUCTS}

        self.positions        = {p: state.position.get(p, 0) for p in self.PRODUCTS}
        self.buy_orders_sent  = {p: 0 for p in self.PRODUCTS}
        self.sell_orders_sent = {p: 0 for p in self.PRODUCTS}

        self.load_state(state)

        for product in self.PRODUCTS:
            self.trade_product(product, state, result[product])

        return result, 0, self.save_state()
