from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any, Optional, Tuple
import json
try:
    import jsonpickle
except Exception:
    class _JsonPickleFallback:
        @staticmethod
        def encode(obj):
            return json.dumps(obj, separators=(",", ":"))
        @staticmethod
        def decode(data):
            return json.loads(data)
    jsonpickle = _JsonPickleFallback()
import math
import base64
import zlib


class Trader:
    """
    Clean merger for the five independent strategies.

    Important:
    - Each child strategy receives only its own previous traderData string.
    - Outer traderData is a compressed JSON dict of those child strings.
    - No child strategy logic is changed.
    """

    OUTER_PREFIX = "Z:"

    def __init__(self):
        self.rest_of_follow = RestOfFollowTrader()
        self.uv_visor = UVVisorTradeflowTrader()
        self.translator = TranslatorTrader()
        self.robot = RobotTrader()
        self.pebbles = PebblesTrader()

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        memory = self.load_outer_memory(state.traderData)

        strategies = [
            ("rest_of_follow", self.rest_of_follow),
            ("uv_visor_tradeflow", self.uv_visor),
            ("translator", self.translator),
            ("robot", self.robot),
            ("pebbles", self.pebbles),
        ]

        for key, strategy in strategies:
            sub_state = self.with_trader_data(state, memory.get(key, ""))
            sub_result, sub_conversions, sub_trader_data = strategy.run(sub_state)

            conversions += sub_conversions
            memory[key] = sub_trader_data or ""

            for product, orders in sub_result.items():
                if orders:
                    result.setdefault(product, []).extend(orders)

        traderData = self.dump_outer_memory(memory)
        return result, conversions, traderData

    @staticmethod
    def with_trader_data(state: TradingState, trader_data: str) -> TradingState:
        return TradingState(
            trader_data,
            state.timestamp,
            state.listings,
            state.order_depths,
            state.own_trades,
            state.market_trades,
            state.position,
            state.observations,
        )

    @classmethod
    def dump_outer_memory(cls, memory: Dict[str, str]) -> str:
        raw = json.dumps(memory, separators=(",", ":"), ensure_ascii=False)
        compressed = zlib.compress(raw.encode("utf-8"), level=9)
        return cls.OUTER_PREFIX + base64.b64encode(compressed).decode("ascii")

    @classmethod
    def load_outer_memory(cls, traderData: str) -> Dict[str, str]:
        if not traderData:
            return {}

        # Current compact format.
        if traderData.startswith(cls.OUTER_PREFIX):
            try:
                payload = traderData[len(cls.OUTER_PREFIX):]
                raw = zlib.decompress(base64.b64decode(payload.encode("ascii"))).decode("utf-8")
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    return {str(k): (v if isinstance(v, str) else "") for k, v in decoded.items()}
            except Exception:
                return {}

        # Backward-compatible fallback for uncompressed JSON.
        try:
            decoded = json.loads(traderData)
            if isinstance(decoded, dict):
                return {str(k): (v if isinstance(v, str) else "") for k, v in decoded.items()}
        except Exception:
            pass

        # Backward-compatible fallback for the previous jsonpickle outer wrapper.
        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return {str(k): (v if isinstance(v, str) else "") for k, v in decoded.items()}
        except Exception:
            pass

        return {}


class RestOfFollowTrader:
    """
    MICROCHIP + OXYGEN v4 rollback-fix
    """

    POSITION_LIMITS = {
        "MICROCHIP_RECTANGLE": 10,
        "OXYGEN_SHAKE_CHOCOLATE": 10,
    }

    PARAMS = {
        "MICROCHIP_RECTANGLE": {
            "decay": 0.74,
            "flow_to_fair": 0.44,
            "take_edge": 0.70,
            "quote_edge": 0.90,
            "signal_threshold": 0.85,
            "exit_threshold": 0.65,
            "base_size": 2,
            "max_order_size": 4,
            "inventory_skew": 0.24,
            "chase_guard_ticks": 2.5,
            "chase_take_penalty": 0.75,
            "pullback_bonus": 0.22,
            "use_fresh_gate": False,
            "fresh_threshold": 0.0,
        },

        "OXYGEN_SHAKE_CHOCOLATE": {
            "decay": 0.82,
            "flow_to_fair": 0.34,
            "take_edge": 0.95,
            "quote_edge": 1.10,
            "signal_threshold": 1.10,
            "exit_threshold": 0.75,
            "base_size": 2,
            "max_order_size": 4,
            "inventory_skew": 0.22,
            "chase_guard_ticks": 3.5,
            "chase_take_penalty": 0.85,
            "pullback_bonus": 0.18,
            "use_fresh_gate": True,
            "fresh_threshold": 1.0,
        },
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        memory = self.load_memory(state.traderData)
        memory.setdefault("flow_signal", {})
        memory.setdefault("fresh_flow", {})
        memory.setdefault("last_mid", {})
        memory.setdefault("ema_move", {})

        for product in self.POSITION_LIMITS:
            if product not in state.order_depths:
                continue

            orders = self.trade_flow_product(product, state, memory)
            if orders:
                result[product] = orders

        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData

    def trade_flow_product(
        self,
        product: str,
        state: TradingState,
        memory: Dict[str, Any],
    ) -> List[Order]:

        depth: OrderDepth = state.order_depths[product]

        if not depth.buy_orders or not depth.sell_orders:
            return []

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        bid_vol = depth.buy_orders[best_bid]
        ask_vol = -depth.sell_orders[best_ask]

        if bid_vol <= 0 or ask_vol <= 0:
            return []

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        p = self.PARAMS[product]

        # ------------------------------------------------------------
        # 1. Mid movement memory
        # ------------------------------------------------------------
        last_mid = memory["last_mid"].get(product, mid)
        mid_move = mid - last_mid
        memory["last_mid"][product] = mid

        old_ema_move = memory["ema_move"].get(product, 0.0)
        ema_move = 0.70 * old_ema_move + 0.30 * mid_move
        memory["ema_move"][product] = ema_move

        # ------------------------------------------------------------
        # 2. Trade-flow signal
        # Trade above mid = bullish.
        # Trade below mid = bearish.
        # ------------------------------------------------------------
        old_signal = memory["flow_signal"].get(product, 0.0)
        signal = old_signal * p["decay"]

        raw_flow = 0.0

        for trade in state.market_trades.get(product, []):
            if trade.price > mid:
                raw_flow += trade.quantity
            elif trade.price < mid:
                raw_flow -= trade.quantity

        signal += raw_flow
        signal = self.clip(signal, -24.0, 24.0)
        memory["flow_signal"][product] = signal

        # ------------------------------------------------------------
        # 3. Fresh-flow signal
        # Only OXYGEN uses this as aggressive entry confirmation.
        # ------------------------------------------------------------
        old_fresh = memory["fresh_flow"].get(product, 0.0)
        fresh = 0.50 * old_fresh + raw_flow
        fresh = self.clip(fresh, -16.0, 16.0)
        memory["fresh_flow"][product] = fresh

        fresh_agrees = True

        if p["use_fresh_gate"]:
            fresh_agrees = (
                (signal > 0 and fresh >= p["fresh_threshold"])
                or (signal < 0 and fresh <= -p["fresh_threshold"])
            )

        # ------------------------------------------------------------
        # 4. Inventory-adjusted fair value
        # ------------------------------------------------------------
        fair = mid + p["flow_to_fair"] * signal - p["inventory_skew"] * pos

        # ------------------------------------------------------------
        # 5. Dynamic edge
        # Avoid chasing large same-direction moves.
        # Reward pullback entries.
        # ------------------------------------------------------------
        same_direction_move = signal * mid_move > 0
        pullback_entry = signal * mid_move < 0

        take_edge = p["take_edge"]
        quote_edge = p["quote_edge"]

        if same_direction_move and abs(mid_move) >= p["chase_guard_ticks"]:
            take_edge += p["chase_take_penalty"]
            quote_edge += 0.35

        elif pullback_entry:
            take_edge = max(0.35, take_edge - p["pullback_bonus"])
            quote_edge = max(0.55, quote_edge - p["pullback_bonus"] / 2)

        # ------------------------------------------------------------
        # 6. Size
        # ------------------------------------------------------------
        abs_signal = abs(signal)
        size = p["base_size"] + int(abs_signal // 6)
        size = self.clip_int(size, 1, p["max_order_size"])

        if abs(pos) >= 7:
            size = min(size, 2)

        orders: List[Order] = []

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # ------------------------------------------------------------
        # 7. Reversal exit
        # If signal flips against inventory, reduce first.
        # ------------------------------------------------------------
        if pos > 0 and signal <= -p["exit_threshold"] and sell_capacity > 0:
            qty = min(pos, bid_vol, max(1, size))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_capacity -= qty

        elif pos < 0 and signal >= p["exit_threshold"] and buy_capacity > 0:
            qty = min(-pos, ask_vol, max(1, size))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_capacity -= qty

        hypothetical_pos = pos + sum(o.quantity for o in orders)
        buy_capacity = limit - hypothetical_pos
        sell_capacity = limit + hypothetical_pos

        # ------------------------------------------------------------
        # 8. Aggressive taking
        # MICROCHIP can take normally.
        # OXYGEN needs fresh-flow confirmation.
        # ------------------------------------------------------------
        if signal >= p["signal_threshold"] and fresh_agrees and buy_capacity > 0:
            if hypothetical_pos < int(0.85 * limit) and best_ask <= fair - take_edge:
                qty = min(buy_capacity, ask_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    hypothetical_pos += qty
                    buy_capacity -= qty

        elif signal <= -p["signal_threshold"] and fresh_agrees and sell_capacity > 0:
            if hypothetical_pos > -int(0.85 * limit) and best_bid >= fair + take_edge:
                qty = min(sell_capacity, bid_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    hypothetical_pos -= qty
                    sell_capacity -= qty

        # ------------------------------------------------------------
        # 9. Passive one-sided quoting
        # OXYGEN stale signal only quotes tiny size 1.
        # No forced flatten. No hard stop.
        # ------------------------------------------------------------
        passive_size = size

        if p["use_fresh_gate"] and not fresh_agrees:
            passive_size = 1

        if spread >= 2:
            if signal >= p["signal_threshold"] and buy_capacity > 0 and hypothetical_pos < limit:
                quote_price = min(best_bid + 1, math.floor(fair - quote_edge / 2))
                quote_price = min(quote_price, best_ask - 1)

                if best_bid <= quote_price < best_ask:
                    qty = min(buy_capacity, passive_size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, qty))

            elif signal <= -p["signal_threshold"] and sell_capacity > 0 and hypothetical_pos > -limit:
                quote_price = max(best_ask - 1, math.ceil(fair + quote_edge / 2))
                quote_price = max(quote_price, best_bid + 1)

                if best_bid < quote_price <= best_ask:
                    qty = min(sell_capacity, passive_size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, -qty))

        return orders

    @staticmethod
    def clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def clip_int(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData: str) -> Dict[str, Any]:
        if not traderData:
            return {}

        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass

        return {}


class UVVisorTradeflowTrader:
    POSITION_LIMITS = {
        "UV_VISOR_RED": 10,
        "UV_VISOR_YELLOW": 10,
        # "UV_VISOR_MAGENTA": 10,
        # "UV_VISOR_ORANGE": 10,
        # "UV_VISOR_AMBER": 10,
    }

    PARAMS = {
        # Keep RED as the monster version.
        "UV_VISOR_RED": {
            "mode": "protected",
            "decay": 0.76,
            "flow_to_fair": 0.46,
            "take_edge": 0.65,
            "quote_edge": 0.85,
            "signal_threshold": 0.8,
            "exit_threshold": 0.6,
            "base_size": 2,
            "max_order_size": 4,
            "inventory_skew": 0.22,
            "chase_guard_ticks": 3.0,
            "chase_take_penalty": 0.75,
            "pullback_bonus": 0.25,
        },

        # YELLOW goes back toward v1/simple style.
        # Less anti-chase, no forced reversal exit, more willing to hold trend.
        "UV_VISOR_YELLOW": {
            "mode": "simple",
            "decay": 0.70,
            "flow_to_fair": 0.39,
            "take_edge": 0.62,
            "quote_edge": 0.90,
            "signal_threshold": 0.75,
            "base_size": 2,
            "max_order_size": 5,
            "inventory_skew": 0.16,
        },
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        memory = self.load_memory(state.traderData)
        memory.setdefault("flow_signal", {})
        memory.setdefault("last_mid", {})
        memory.setdefault("ema_move", {})

        for product in self.POSITION_LIMITS:
            if product not in state.order_depths:
                continue

            p = self.PARAMS[product]

            if p["mode"] == "protected":
                orders = self.trade_protected(product, state, memory)
            else:
                orders = self.trade_simple(product, state, memory)

            if orders:
                result[product] = orders

        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData

    # ================================================================
    # RED LOGIC: protected v2/v4 style
    # ================================================================
    def trade_protected(self, product: str, state: TradingState, memory: Dict[str, Any]) -> List[Order]:
        depth: OrderDepth = state.order_depths[product]
        if not depth.buy_orders or not depth.sell_orders:
            return []

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_vol = depth.buy_orders[best_bid]
        ask_vol = -depth.sell_orders[best_ask]

        if bid_vol <= 0 or ask_vol <= 0:
            return []

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        p = self.PARAMS[product]

        last_mid = memory["last_mid"].get(product, mid)
        mid_move = mid - last_mid
        memory["last_mid"][product] = mid

        old_ema_move = memory["ema_move"].get(product, 0.0)
        ema_move = 0.65 * old_ema_move + 0.35 * mid_move
        memory["ema_move"][product] = ema_move

        old_signal = memory["flow_signal"].get(product, 0.0)
        signal = old_signal * p["decay"]

        raw_flow = 0.0
        for trade in state.market_trades.get(product, []):
            if trade.price > mid:
                raw_flow += trade.quantity
            elif trade.price < mid:
                raw_flow -= trade.quantity

        signal += raw_flow
        signal = self.clip(signal, -24.0, 24.0)
        memory["flow_signal"][product] = signal

        fair = mid + p["flow_to_fair"] * signal - p["inventory_skew"] * pos

        same_direction_move = signal * mid_move > 0
        pullback_entry = signal * mid_move < 0

        take_edge = p["take_edge"]
        quote_edge = p["quote_edge"]

        if same_direction_move and abs(mid_move) >= p["chase_guard_ticks"]:
            take_edge += p["chase_take_penalty"]
            quote_edge += 0.35
        elif pullback_entry:
            take_edge = max(0.30, take_edge - p["pullback_bonus"])
            quote_edge = max(0.45, quote_edge - p["pullback_bonus"] / 2)

        abs_signal = abs(signal)
        size = p["base_size"] + int(abs_signal // 6)
        size = self.clip_int(size, 1, p["max_order_size"])

        if abs(pos) >= 7:
            size = min(size, 2)

        orders: List[Order] = []

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Reversal exit: keep for RED because it helped.
        if pos > 0 and signal <= -p["exit_threshold"] and sell_capacity > 0:
            qty = min(pos, bid_vol, max(1, size))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_capacity -= qty

        elif pos < 0 and signal >= p["exit_threshold"] and buy_capacity > 0:
            qty = min(-pos, ask_vol, max(1, size))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_capacity -= qty

        hypothetical_pos = pos + sum(o.quantity for o in orders)
        buy_capacity = limit - hypothetical_pos
        sell_capacity = limit + hypothetical_pos

        # Aggressive taking.
        if signal >= p["signal_threshold"] and buy_capacity > 0:
            if hypothetical_pos < int(0.85 * limit) and best_ask <= fair - take_edge:
                qty = min(buy_capacity, ask_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    hypothetical_pos += qty
                    buy_capacity -= qty

        elif signal <= -p["signal_threshold"] and sell_capacity > 0:
            if hypothetical_pos > -int(0.85 * limit) and best_bid >= fair + take_edge:
                qty = min(sell_capacity, bid_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    hypothetical_pos -= qty
                    sell_capacity -= qty

        # Passive one-sided quote.
        if spread >= 2:
            if signal >= p["signal_threshold"] and buy_capacity > 0 and hypothetical_pos < limit:
                quote_price = min(best_bid + 1, math.floor(fair - quote_edge / 2))
                quote_price = min(quote_price, best_ask - 1)

                if best_bid <= quote_price < best_ask:
                    qty = min(buy_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, qty))

            elif signal <= -p["signal_threshold"] and sell_capacity > 0 and hypothetical_pos > -limit:
                quote_price = max(best_ask - 1, math.ceil(fair + quote_edge / 2))
                quote_price = max(quote_price, best_bid + 1)

                if best_bid < quote_price <= best_ask:
                    qty = min(sell_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, -qty))

        return orders

    # ================================================================
    # YELLOW LOGIC: simpler v1-style trend follower
    # ================================================================
    def trade_simple(self, product: str, state: TradingState, memory: Dict[str, Any]) -> List[Order]:
        depth: OrderDepth = state.order_depths[product]
        if not depth.buy_orders or not depth.sell_orders:
            return []

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_vol = depth.buy_orders[best_bid]
        ask_vol = -depth.sell_orders[best_ask]

        if bid_vol <= 0 or ask_vol <= 0:
            return []

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        p = self.PARAMS[product]

        old_signal = memory["flow_signal"].get(product, 0.0)
        signal = old_signal * p["decay"]

        raw_flow = 0.0
        for trade in state.market_trades.get(product, []):
            if trade.price > mid:
                raw_flow += trade.quantity
            elif trade.price < mid:
                raw_flow -= trade.quantity

        signal += raw_flow
        signal = self.clip(signal, -24.0, 24.0)
        memory["flow_signal"][product] = signal

        fair = mid + p["flow_to_fair"] * signal - p["inventory_skew"] * pos

        abs_signal = abs(signal)
        size = p["base_size"] + int(abs_signal // 5)
        size = self.clip_int(size, 1, p["max_order_size"])

        # Let YELLOW press harder than RED, but avoid stupid full-limit slamming.
        if abs(pos) >= 8:
            size = min(size, 2)

        orders: List[Order] = []

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # Aggressive take in signal direction.
        if signal >= p["signal_threshold"] and buy_capacity > 0:
            if best_ask <= fair - p["take_edge"]:
                qty = min(buy_capacity, ask_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_capacity -= qty

        elif signal <= -p["signal_threshold"] and sell_capacity > 0:
            if best_bid >= fair + p["take_edge"]:
                qty = min(sell_capacity, bid_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_capacity -= qty

        hypothetical_pos = pos + sum(o.quantity for o in orders)
        buy_capacity = limit - hypothetical_pos
        sell_capacity = limit + hypothetical_pos

        # Passive one-sided quote.
        if spread >= 2:
            if signal >= p["signal_threshold"] and buy_capacity > 0:
                quote_price = min(best_bid + 1, math.floor(fair - p["quote_edge"] / 2))
                quote_price = min(quote_price, best_ask - 1)

                if best_bid <= quote_price < best_ask:
                    qty = min(buy_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, qty))

            elif signal <= -p["signal_threshold"] and sell_capacity > 0:
                quote_price = max(best_ask - 1, math.ceil(fair + p["quote_edge"] / 2))
                quote_price = max(quote_price, best_bid + 1)

                if best_bid < quote_price <= best_ask:
                    qty = min(sell_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, -qty))

        return orders

    @staticmethod
    def clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def clip_int(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData: str) -> Dict[str, Any]:
        if not traderData:
            return {}
        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
        return {}


class TranslatorTrader:
    POSITION_LIMIT = 10

    PRODUCTS = [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ]

    LOOKBACK = 250
    VOL_LOOKBACK = 120
    REBALANCE_EVERY = 25
    MAX_TRADE_SIZE = 3

    # Dynamic ranking sizes
    TOP_SIZE = 10
    SECOND_SIZE = 5
    BOTTOM_SIZE = -10
    SECOND_BOTTOM_SIZE = -5

    # If scores are too close, don't force a trade.
    MIN_SCORE_GAP = 0.20

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.cached_targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

    def bid(self):
        return 15

    # =====================================================
    # Helpers
    # =====================================================

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if (
            len(order_depth.buy_orders) == 0
            or len(order_depth.sell_orders) == 0
        ):
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def mean(self, xs: List[float]) -> float:
        return sum(xs) / len(xs)

    def std(self, xs: List[float]) -> float:
        if len(xs) <= 1:
            return 0.0
        m = self.mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def clamp_position(self, x: int) -> int:
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    # =====================================================
    # History
    # =====================================================

    def update_history(self, state: TradingState):
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue

            self.mid_history[product].append(mid)

            if len(self.mid_history[product]) > 1500:
                self.mid_history[product] = self.mid_history[product][-1500:]

    # =====================================================
    # Dynamic cross-sectional signal
    # =====================================================

    def product_score(self, product: str) -> Optional[float]:
        hist = self.mid_history[product]

        if len(hist) < max(self.LOOKBACK, self.VOL_LOOKBACK) + 1:
            return None

        now = hist[-1]
        old = hist[-self.LOOKBACK]

        if old <= 0:
            return None

        # Recent return
        ret = (now - old) / old

        # Volatility of percentage changes
        recent = hist[-self.VOL_LOOKBACK :]
        pct_changes = []

        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                pct_changes.append((recent[i] - recent[i - 1]) / recent[i - 1])

        vol = self.std(pct_changes)

        if vol <= 1e-9:
            return None

        return ret / (vol * math.sqrt(self.LOOKBACK))

    def build_targets(self) -> Dict[str, int]:
        targets = {p: 0 for p in self.PRODUCTS}

        scores = {}

        for product in self.PRODUCTS:
            score = self.product_score(product)
            if score is not None:
                scores[product] = score

        if len(scores) < len(self.PRODUCTS):
            return targets

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_product, best_score = ranked[0]
        second_product, second_score = ranked[1]
        middle_product, middle_score = ranked[2]
        second_worst_product, second_worst_score = ranked[3]
        worst_product, worst_score = ranked[4]

        # Avoid trading if the whole ranking is too compressed.
        if best_score - worst_score < self.MIN_SCORE_GAP:
            return targets

        targets[best_product] = self.TOP_SIZE
        targets[second_product] = self.SECOND_SIZE
        targets[middle_product] = 0
        targets[second_worst_product] = self.SECOND_BOTTOM_SIZE
        targets[worst_product] = self.BOTTOM_SIZE

        for product in targets:
            targets[product] = self.clamp_position(targets[product])

        return targets

    def get_targets(self) -> Dict[str, int]:
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets()

        return self.cached_targets

    # =====================================================
    # Execution
    # =====================================================

    def move_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ) -> List[Order]:

        orders: List[Order] = []

        target_pos = self.clamp_position(target_pos)
        needed = target_pos - current_pos

        if needed == 0:
            return orders

        if needed > 0:
            remaining = min(
                needed,
                self.POSITION_LIMIT - current_pos,
                self.MAX_TRADE_SIZE,
            )

            for ask_price in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break

                ask_volume = -order_depth.sell_orders[ask_price]
                if ask_volume <= 0:
                    continue

                qty = min(remaining, ask_volume)

                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    remaining -= qty

        else:
            remaining = min(
                -needed,
                self.POSITION_LIMIT + current_pos,
                self.MAX_TRADE_SIZE,
            )

            for bid_price in sorted(
                order_depth.buy_orders.keys(), reverse=True
            ):
                if remaining <= 0:
                    break

                bid_volume = order_depth.buy_orders[bid_price]
                if bid_volume <= 0:
                    continue

                qty = min(remaining, bid_volume)

                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    remaining -= qty

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {p: [] for p in self.PRODUCTS}
        conversions = 0

        if state.traderData:
            try:
                memory = jsonpickle.decode(state.traderData)
                if isinstance(memory, dict):
                    self.tick = int(memory.get("tick", self.tick))
                    self.mid_history = memory.get("mid_history", self.mid_history)
                    self.cached_targets = memory.get("cached_targets", self.cached_targets)
            except Exception:
                pass

        self.tick += 1
        self.update_history(state)

        targets = self.get_targets()

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            current_pos = state.position.get(product, 0)
            target_pos = targets.get(product, 0)

            result[product] = self.move_to_target(
                product=product,
                order_depth=state.order_depths[product],
                current_pos=current_pos,
                target_pos=target_pos,
            )

        traderData = jsonpickle.encode({
            "tick": self.tick,
            "mid_history": self.mid_history,
            "cached_targets": self.cached_targets,
        })
        return result, conversions, traderData


class RobotTrader:
    def __init__(self):
        # Use all robots for the fair-value basket
        self.ALL_ROBOTS = [
            "ROBOT_DISHES",
            "ROBOT_IRONING",
            "ROBOT_LAUNDRY",
            "ROBOT_MOPPING",
            "ROBOT_VACUUMING",
        ]

        # Only trade the products that made money
        self.TRADE_PRODUCTS = [
            "ROBOT_DISHES",
            "ROBOT_LAUNDRY",
            "ROBOT_VACUUMING",
        ]

        self.DISHES = "ROBOT_DISHES"

        self.LIMIT = {
            "ROBOT_DISHES": 10,
            "ROBOT_IRONING": 10,
            "ROBOT_LAUNDRY": 10,
            "ROBOT_MOPPING": 10,
            "ROBOT_VACUUMING": 10,
        }

        # Base parameters
        self.window = 90
        self.min_history = 35
        self.exit_z = 0.35

        # Product-specific thresholds
        # Laundry stays aggressive because it is the main winner.
        # Dishes is made more conservative because it had larger drawdowns.
        self.entry_z_by_product = {
            "ROBOT_LAUNDRY": 1.85,
            "ROBOT_VACUUMING": 1.90,
            "ROBOT_DISHES": 2.10,
        }

        self.strong_z_by_product = {
            "ROBOT_LAUNDRY": 2.65,
            "ROBOT_VACUUMING": 2.75,
            "ROBOT_DISHES": 3.00,
        }

        # Prioritize Laundry when only 2 trades can be sent
        self.PRODUCT_PRIORITY = {
            "ROBOT_LAUNDRY": 1.20,
            "ROBOT_VACUUMING": 1.00,
            "ROBOT_DISHES": 0.85,
        }

        # Dishes flow signal
        self.flow_decay = 0.70
        self.flow_entry = 5.0
        self.flow_strong = 10.0

        # Size control
        self.passive_size = 2
        self.active_size = 1
        self.max_orders_per_tick = 2

    # ---------------- basic helpers ----------------

    def pos(self, state: TradingState, product: str):
        return state.position.get(product, 0)

    def best_bid(self, depth: OrderDepth):
        if not depth.buy_orders:
            return None
        return max(depth.buy_orders.keys())

    def best_ask(self, depth: OrderDepth):
        if not depth.sell_orders:
            return None
        return min(depth.sell_orders.keys())

    def mid(self, depth: OrderDepth):
        bid = self.best_bid(depth)
        ask = self.best_ask(depth)

        if bid is None or ask is None:
            return None

        return (bid + ask) / 2

    def mean_std(self, arr):
        if len(arr) == 0:
            return 0, 1

        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)

        return mean, max(math.sqrt(var), 1e-9)

    # ---------------- capacity helpers ----------------

    def net_ordered(self, orders: List[Order]):
        return sum(o.quantity for o in orders)

    def buy_cap(self, state: TradingState, product: str, orders: List[Order]):
        current = self.pos(state, product)
        already = self.net_ordered(orders)
        projected = current + already

        return self.LIMIT[product] - projected

    def sell_cap(self, state: TradingState, product: str, orders: List[Order]):
        current = self.pos(state, product)
        already = self.net_ordered(orders)
        projected = current + already

        return self.LIMIT[product] + projected

    # ---------------- order helpers ----------------

    def passive_buy(self, orders, state, product, depth, size):
        bid = self.best_bid(depth)
        ask = self.best_ask(depth)

        if bid is None or ask is None:
            return

        qty = min(size, self.buy_cap(state, product, orders))

        if qty <= 0:
            return

        price = min(bid + 1, ask - 1)

        if price > bid and price < ask:
            orders.append(Order(product, price, qty))
        else:
            orders.append(Order(product, bid, qty))

    def passive_sell(self, orders, state, product, depth, size):
        bid = self.best_bid(depth)
        ask = self.best_ask(depth)

        if bid is None or ask is None:
            return

        qty = min(size, self.sell_cap(state, product, orders))

        if qty <= 0:
            return

        price = max(ask - 1, bid + 1)

        if price > bid and price < ask:
            orders.append(Order(product, price, -qty))
        else:
            orders.append(Order(product, ask, -qty))

    def active_buy(self, orders, state, product, depth, size):
        ask = self.best_ask(depth)

        if ask is None:
            return

        qty = min(size, self.buy_cap(state, product, orders))

        if qty > 0:
            orders.append(Order(product, ask, qty))

    def active_sell(self, orders, state, product, depth, size):
        bid = self.best_bid(depth)

        if bid is None:
            return

        qty = min(size, self.sell_cap(state, product, orders))

        if qty > 0:
            orders.append(Order(product, bid, -qty))

    def reduce_position_passive(self, orders, state, product, depth, max_qty=1):
        p = self.pos(state, product)

        if p > 0:
            self.passive_sell(orders, state, product, depth, min(max_qty, p))
        elif p < 0:
            self.passive_buy(orders, state, product, depth, min(max_qty, -p))

    # ---------------- flow signal ----------------

    def update_dishes_flow(self, state, data, current_mid):
        old_flow = data.get("dishes_flow", 0.0)
        signed = 0.0

        trades = state.market_trades.get(self.DISHES, [])

        for t in trades:
            qty = abs(t.quantity)

            if t.price > current_mid:
                signed += qty
            elif t.price < current_mid:
                signed -= qty

        new_flow = self.flow_decay * old_flow + signed
        data["dishes_flow"] = new_flow

        return new_flow

    def calc_z(self, hist):
        lookback = hist[-self.window:] if len(hist) >= self.window else hist
        mean, std = self.mean_std(lookback)

        return (hist[-1] - mean) / std

    # ---------------- main ----------------

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        # Return empty order lists for all robots.
        # Ironing and Mopping stay empty, so they are not traded.
        for p in self.ALL_ROBOTS:
            result[p] = []

        # Need all 5 robots because the fair-value basket uses all 5
        for p in self.ALL_ROBOTS:
            if p not in state.order_depths:
                return result, 0, json.dumps(data)

        mids = {}

        for p in self.ALL_ROBOTS:
            m = self.mid(state.order_depths[p])

            if m is None:
                return result, 0, json.dumps(data)

            mids[p] = m

        # Full robot basket average
        robot_avg = sum(mids[p] for p in self.ALL_ROBOTS) / len(self.ALL_ROBOTS)

        dev_hist = data.get("robot_dev_hist", {})

        for p in self.ALL_ROBOTS:
            if p not in dev_hist:
                dev_hist[p] = []

            dev = mids[p] - robot_avg
            dev_hist[p].append(dev)

            if len(dev_hist[p]) > 250:
                dev_hist[p] = dev_hist[p][-250:]

        data["robot_dev_hist"] = dev_hist

        dishes_flow = self.update_dishes_flow(state, data, mids[self.DISHES])

        if len(dev_hist[self.DISHES]) < self.min_history:
            return result, 0, json.dumps(data)

        candidates = []

        # ---------------- build candidates ----------------

        for p in self.TRADE_PRODUCTS:
            hist = dev_hist[p]
            z = self.calc_z(hist)

            entry_z = self.entry_z_by_product[p]

            signal = 0.0

            # Expensive -> sell
            if z > entry_z:
                signal = -abs(z)

            # Cheap -> buy
            elif z < -entry_z:
                signal = abs(z)

            # Dishes flow confirmation only adjusts an existing signal
            if p == self.DISHES and signal != 0:
                if dishes_flow > self.flow_entry:
                    signal += 0.75
                elif dishes_flow < -self.flow_entry:
                    signal -= 0.75

            # Exit when close to fair
            if abs(z) < self.exit_z:
                candidates.append(("exit", 0.0, p, z))

            elif signal != 0:
                strength = abs(signal) * self.PRODUCT_PRIORITY.get(p, 1.0)
                candidates.append(("trade", strength, p, z))

        trade_candidates = [c for c in candidates if c[0] == "trade"]
        exit_candidates = [c for c in candidates if c[0] == "exit"]

        trade_candidates.sort(key=lambda x: x[1], reverse=True)

        trades_sent = 0

        # ---------------- execute strongest trades ----------------

        for _, strength, p, z in trade_candidates:
            if trades_sent >= self.max_orders_per_tick:
                break

            depth = state.order_depths[p]
            orders = result[p]

            entry_z = self.entry_z_by_product[p]
            strong_z = self.strong_z_by_product[p]

            # Expensive -> sell
            if z > entry_z:
                self.passive_sell(orders, state, p, depth, self.passive_size)

                if z > strong_z:
                    self.active_sell(orders, state, p, depth, self.active_size)

                trades_sent += 1

            # Cheap -> buy
            elif z < -entry_z:
                self.passive_buy(orders, state, p, depth, self.passive_size)

                if z < -strong_z:
                    self.active_buy(orders, state, p, depth, self.active_size)

                trades_sent += 1

        # ---------------- safer extra Dishes flow trade ----------------
        # Only follow Dishes flow if relative-value signal agrees.
        # This avoids adding random Dishes exposure during bad drawdowns.

        if trades_sent < self.max_orders_per_tick:
            p = self.DISHES
            depth = state.order_depths[p]
            orders = result[p]

            z = self.calc_z(dev_hist[p])
            entry_z = self.entry_z_by_product[p]

            # Dishes cheap + aggressive buying flow -> buy
            if z < -entry_z and dishes_flow > self.flow_strong:
                self.passive_buy(orders, state, p, depth, 1)
                trades_sent += 1

            # Dishes expensive + aggressive selling flow -> sell
            elif z > entry_z and dishes_flow < -self.flow_strong:
                self.passive_sell(orders, state, p, depth, 1)
                trades_sent += 1

        # ---------------- reduce risk when fair ----------------

        if trades_sent < self.max_orders_per_tick:
            for _, _, p, z in exit_candidates:
                if trades_sent >= self.max_orders_per_tick:
                    break

                if self.pos(state, p) != 0:
                    self.reduce_position_passive(
                        result[p],
                        state,
                        p,
                        state.order_depths[p],
                        max_qty=1,
                    )
                    trades_sent += 1

        return result, 0, json.dumps(data)


class PebblesTrader:
    """
    Hybrid Pebbles Trader - Version 4: XS/L priority + slightly more active side scanner.

    Trades only Pebbles.

    Idea:
    1. Use the XS/L specialist signal as the main attacking module when it is active.
       This preserves the high-upside behavior of our v11 XS/L strategy.
    2. When XS/L specialist is active, allow a small non-conflicting scanner only on XL/S.
       This tries to add the teammate scanner alpha while XS/L owns XS and L.
    3. When XS/L specialist is inactive, use the normal quality-filtered relative-z scanner.
       PEBBLES_M is tracked and flattened if needed, but excluded from new scanner pairs.
    4. All targets are centralized, so the two Pebbles strategies never fight on the same product.

    No PnL lock.
    No timestamp cutoff.
    No fixed static target basket.
    """

    PRODUCTS = [
        "PEBBLES_XL",
        "PEBBLES_L",
        "PEBBLES_M",
        "PEBBLES_S",
        "PEBBLES_XS",
    ]

    XS = "PEBBLES_XS"
    L = "PEBBLES_L"

    POSITION_LIMIT = 10

    # ---------- XS/L specialist parameters, inherited from v11 ----------
    ALPHA = 13768.699562
    BETA = -0.625515

    XL_WINDOW = 300
    SHORT_VOL_WINDOW = 80
    ACCEL_SHORT = 20
    ACCEL_LONG = 80

    SHORT_ENTRY_Z = 1.95
    LONG_ENTRY_Z = 2.15
    EXIT_Z = 0.40

    LONG_MIN_VOL_RATIO = 1.03
    SHORT_MAX_ACCEL_Z = 2.90

    MAX_HOLD_BARS = 420

    RAW_TRAIL_ACTIVATE_STD = 0.55
    RAW_TRAIL_RETRACE_STD = 0.40
    RESET_Z = 0.85

    XS_L_TARGET_SIZE = 10
    XS_L_ORDER_SIZE = 3

    # ---------- Dynamic scanner parameters, based on teammate version ----------
    SCAN_WINDOW = 90
    SCAN_MIN_HISTORY = 15
    SCAN_PRODUCTS = [
        "PEBBLES_XL",
        "PEBBLES_L",
        "PEBBLES_S",
        "PEBBLES_XS",
    ]

    SCAN_ENTRY_Z = 2.15
    SCAN_EXIT_Z = 0.95
    SCAN_FLIP_BUFFER = 0.75
    SCAN_PASSIVE_SIZE = 2
    SCAN_NORMAL_IMPROVE = 1
    SCAN_ENTRY_IMPROVE = 2

    # Side scanner used only while XS/L specialist is active.
    # It only trades XL/S, so it does not fight the specialist's XS/L inventory.
    SIDE_SCAN_PRODUCTS = [
        "PEBBLES_XL",
        "PEBBLES_S",
    ]
    # v4: lower than v3 to let XL/S side scanner pick up more non-conflicting opportunities.
    SIDE_ENTRY_Z = 2.10
    SIDE_EXIT_Z = 0.90
    SIDE_FLIP_BUFFER = 0.80
    SIDE_PASSIVE_SIZE = 1
    SIDE_MAX_TARGET = 4
    SIDE_NORMAL_IMPROVE = 1
    SIDE_ENTRY_IMPROVE = 2

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {p: [] for p in self.PRODUCTS}
        conversions = 0

        memory = self.load_memory(state.traderData)
        memory.setdefault("xl", {})
        memory.setdefault("scanner", {})
        memory.setdefault("side_scanner", {})
        memory.setdefault("owner", "scanner")

        # Need all Pebbles order books available.
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                return result, conversions, jsonpickle.encode(memory)

        mids: Dict[str, float] = {}
        for product in self.PRODUCTS:
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                return result, conversions, jsonpickle.encode(memory)
            mids[product] = mid

        # Update both signal modules every tick, even if one is not currently trading.
        xl_mode, xl_ready = self.update_xs_l_specialist(memory["xl"], mids)
        scanner_targets, scanner_just_opened = self.update_scanner(memory["scanner"], mids)
        side_targets, side_just_opened = self.update_side_scanner(memory["side_scanner"], mids)

        positions = {p: state.position.get(p, 0) for p in self.PRODUCTS}
        targets = {p: 0 for p in self.PRODUCTS}
        use_specialist_execution = False

        xs_l_inventory = abs(positions[self.XS]) > 0 or abs(positions[self.L]) > 0

        # Priority rule:
        # If XS/L specialist has an active mode, it owns the Pebbles book.
        # If it just exited but still has inventory, it keeps ownership until flat.
        if xl_mode != 0:
            memory["owner"] = "xs_l"
            memory["scanner"]["active_rich"] = None
            memory["scanner"]["active_cheap"] = None

        elif memory.get("owner") == "xs_l" and xs_l_inventory:
            # Continue flattening specialist inventory before allowing scanner back in.
            pass

        else:
            memory["owner"] = "scanner"

        if memory.get("owner") == "xs_l":
            use_specialist_execution = True
            if xl_mode == -1:
                # short spread: sell XS, buy L
                targets[self.XS] = -self.XS_L_TARGET_SIZE
                targets[self.L] = self.XS_L_TARGET_SIZE
                # Add only non-conflicting XL/S side-scanner targets.
                targets["PEBBLES_XL"] = side_targets.get("PEBBLES_XL", 0)
                targets["PEBBLES_S"] = side_targets.get("PEBBLES_S", 0)
            elif xl_mode == 1:
                # long spread: buy XS, sell L
                targets[self.XS] = self.XS_L_TARGET_SIZE
                targets[self.L] = -self.XS_L_TARGET_SIZE
                targets["PEBBLES_XL"] = side_targets.get("PEBBLES_XL", 0)
                targets["PEBBLES_S"] = side_targets.get("PEBBLES_S", 0)
            else:
                # Specialist flat mode: flatten all Pebbles before handing control back.
                targets = {p: 0 for p in self.PRODUCTS}
        else:
            # Main scanner owns the book; side scanner must not run separately.
            targets = scanner_targets

        # Execute centralized targets.
        if use_specialist_execution:
            # Use taking-style execution for XS/L to preserve v11 behavior.
            for product in self.PRODUCTS:
                current_pos = positions[product]
                target_pos = self.clamp(targets[product], -self.POSITION_LIMIT, self.POSITION_LIMIT)

                if product in (self.XS, self.L):
                    result[product] += self.take_toward_target(
                        product,
                        state.order_depths[product],
                        current_pos,
                        target_pos,
                        self.XS_L_ORDER_SIZE,
                    )
                else:
                    # Non-conflicting side scanner on XL/S while specialist owns XS/L.
                    # Other Pebbles are quietly flattened.
                    if product in self.SIDE_SCAN_PRODUCTS:
                        side_improve = self.SIDE_ENTRY_IMPROVE if side_just_opened else self.SIDE_NORMAL_IMPROVE
                        side_size = self.SIDE_PASSIVE_SIZE
                    else:
                        side_improve = self.SCAN_NORMAL_IMPROVE
                        side_size = self.SCAN_PASSIVE_SIZE
                    result[product] += self.passive_toward_target(
                        product,
                        state.order_depths[product],
                        current_pos,
                        target_pos,
                        side_size,
                        side_improve,
                    )
        else:
            improve = self.SCAN_ENTRY_IMPROVE if scanner_just_opened else self.SCAN_NORMAL_IMPROVE
            for product in self.PRODUCTS:
                result[product] += self.passive_toward_target(
                    product,
                    state.order_depths[product],
                    positions[product],
                    self.clamp(targets[product], -self.POSITION_LIMIT, self.POSITION_LIMIT),
                    self.SCAN_PASSIVE_SIZE,
                    improve,
                )

        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData

    # =====================================================================
    # XS/L specialist signal. Updates memory and returns current mode.
    # mode = -1 short XS/long L, +1 long XS/short L, 0 flat.
    # =====================================================================
    def update_xs_l_specialist(self, data: Dict[str, Any], mids: Dict[str, float]) -> Tuple[int, bool]:
        data.setdefault("spread_hist", [])
        data.setdefault("raw_hist", [])
        data.setdefault("mode", 0)
        data.setdefault("hold_bars", 0)
        data.setdefault("reset_ready", True)

        mid_xs = mids[self.XS]
        mid_l = mids[self.L]

        spread = mid_xs - (self.ALPHA + self.BETA * mid_l)
        raw_diff = mid_xs - mid_l

        data["spread_hist"].append(spread)
        data["raw_hist"].append(raw_diff)

        max_hist = max(self.XL_WINDOW, self.ACCEL_LONG, self.SHORT_VOL_WINDOW) + 5
        data["spread_hist"] = data["spread_hist"][-max_hist:]
        data["raw_hist"] = data["raw_hist"][-max_hist:]

        hist = data["spread_hist"]
        raw_hist = data["raw_hist"]

        if len(hist) < self.XL_WINDOW or len(raw_hist) < self.XL_WINDOW:
            data["last_ready"] = False
            return int(data.get("mode", 0)), False

        long_hist = hist[-self.XL_WINDOW:]
        mean = self.mean(long_hist)
        std = self.std(long_hist)
        if std < 1e-9:
            data["last_ready"] = False
            return int(data.get("mode", 0)), False

        z = (spread - mean) / std
        raw_std = self.std(raw_hist[-self.XL_WINDOW:])
        if raw_std < 1e-9:
            raw_std = 1.0

        short_vol = self.std(hist[-self.SHORT_VOL_WINDOW:])
        vol_ratio = short_vol / std if std > 1e-9 else 1.0

        accel_z = 0.0
        if len(hist) > self.ACCEL_LONG:
            recent_change = hist[-1] - hist[-1 - self.ACCEL_SHORT]
            older_change = hist[-1 - self.ACCEL_SHORT] - hist[-1 - self.ACCEL_LONG]
            accel_z = (recent_change - older_change) / std

        mode = int(data.get("mode", 0))

        if mode == 0:
            data["hold_bars"] = 0

            if not data.get("reset_ready", True):
                if abs(z) < self.RESET_Z:
                    data["reset_ready"] = True

            if data.get("reset_ready", True):
                if z > self.SHORT_ENTRY_Z and accel_z <= self.SHORT_MAX_ACCEL_Z:
                    mode = -1
                    self.start_xl_trade(data, raw_diff, raw_std)
                elif z < -self.LONG_ENTRY_Z and vol_ratio >= self.LONG_MIN_VOL_RATIO:
                    mode = 1
                    self.start_xl_trade(data, raw_diff, raw_std)

        else:
            data["hold_bars"] = data.get("hold_bars", 0) + 1
            entry_raw = data.get("entry_raw_diff", raw_diff)
            best_raw = data.get("best_raw_diff", raw_diff)
            entry_raw_std = data.get("entry_raw_std", raw_std)
            trail_active = data.get("raw_trail_active", False)
            exit_reason = ""

            if mode == -1:
                if raw_diff < best_raw:
                    best_raw = raw_diff
                favorable_move = entry_raw - best_raw
                retrace = raw_diff - best_raw
                if favorable_move >= self.RAW_TRAIL_ACTIVATE_STD * entry_raw_std:
                    trail_active = True
                if trail_active and retrace >= self.RAW_TRAIL_RETRACE_STD * raw_std:
                    mode = 0
                    exit_reason = "raw_trail"

            elif mode == 1:
                if raw_diff > best_raw:
                    best_raw = raw_diff
                favorable_move = best_raw - entry_raw
                retrace = best_raw - raw_diff
                if favorable_move >= self.RAW_TRAIL_ACTIVATE_STD * entry_raw_std:
                    trail_active = True
                if trail_active and retrace >= self.RAW_TRAIL_RETRACE_STD * raw_std:
                    mode = 0
                    exit_reason = "raw_trail"

            data["best_raw_diff"] = best_raw
            data["raw_trail_active"] = trail_active

            if mode != 0 and abs(z) < self.EXIT_Z:
                mode = 0
                exit_reason = "z_exit"
            elif mode != 0 and data["hold_bars"] >= self.MAX_HOLD_BARS:
                mode = 0
                exit_reason = "max_hold"

            if mode == 0:
                data["reset_ready"] = True if exit_reason == "z_exit" else False

        data["mode"] = mode
        data["last_ready"] = True
        data["last_spread"] = spread
        data["last_raw_diff"] = raw_diff
        data["last_z"] = z
        data["last_vol_ratio"] = vol_ratio
        data["last_accel_z"] = accel_z
        data["last_raw_std"] = raw_std
        return mode, True

    @staticmethod
    def start_xl_trade(data: Dict[str, Any], raw_diff: float, raw_std: float) -> None:
        data["hold_bars"] = 0
        data["entry_raw_diff"] = raw_diff
        data["best_raw_diff"] = raw_diff
        data["entry_raw_std"] = raw_std
        data["raw_trail_active"] = False

    # =====================================================================
    # Non-conflicting side scanner used only when XS/L specialist is active.
    # It trades only XL/S and uses smaller size than the main scanner.
    # =====================================================================
    def update_side_scanner(self, data: Dict[str, Any], mids: Dict[str, float]) -> Tuple[Dict[str, int], bool]:
        data.setdefault("mid_history", {})
        data.setdefault("active_rich", None)
        data.setdefault("active_cheap", None)
        data.setdefault("just_opened_pair", False)
        data["just_opened_pair"] = False

        for product in self.SIDE_SCAN_PRODUCTS:
            data["mid_history"].setdefault(product, [])
            data["mid_history"][product].append(mids[product])
            data["mid_history"][product] = data["mid_history"][product][-self.SCAN_WINDOW:]

        targets = {p: 0 for p in self.PRODUCTS}
        min_hist_len = min(len(data["mid_history"][p]) for p in self.SIDE_SCAN_PRODUCTS)
        if min_hist_len < self.SCAN_MIN_HISTORY:
            return targets, False

        z_scores: Dict[str, float] = {}
        for product in self.SIDE_SCAN_PRODUCTS:
            hist = data["mid_history"][product]
            z_scores[product] = (mids[product] - self.mean(hist)) / self.std_sample(hist)

        avg_z = self.mean(list(z_scores.values()))
        rel_z = {p: z_scores[p] - avg_z for p in self.SIDE_SCAN_PRODUCTS}

        rich_product = max(self.SIDE_SCAN_PRODUCTS, key=lambda p: rel_z[p])
        cheap_product = min(self.SIDE_SCAN_PRODUCTS, key=lambda p: rel_z[p])
        best_z_spread = rel_z[rich_product] - rel_z[cheap_product]

        active_rich = data.get("active_rich")
        active_cheap = data.get("active_cheap")
        have_active_pair = active_rich in self.SIDE_SCAN_PRODUCTS and active_cheap in self.SIDE_SCAN_PRODUCTS
        active_pair_spread = rel_z[active_rich] - rel_z[active_cheap] if have_active_pair else 0.0

        data["last_rich_product"] = rich_product
        data["last_cheap_product"] = cheap_product
        data["last_best_z_spread"] = best_z_spread
        data["last_active_pair_spread"] = active_pair_spread
        data["last_rel_z"] = rel_z

        should_open = False
        should_close = False
        should_flip = False

        if not have_active_pair:
            if best_z_spread >= self.SIDE_ENTRY_Z:
                should_open = True
        else:
            if active_pair_spread <= self.SIDE_EXIT_Z:
                should_close = True
            elif (
                best_z_spread >= self.SIDE_ENTRY_Z
                and (rich_product != active_rich or cheap_product != active_cheap)
                and best_z_spread > active_pair_spread + self.SIDE_FLIP_BUFFER
            ):
                should_flip = True

        if should_close:
            data["active_rich"] = None
            data["active_cheap"] = None
            return targets, False

        if should_open or should_flip:
            active_rich = rich_product
            active_cheap = cheap_product
            data["active_rich"] = active_rich
            data["active_cheap"] = active_cheap
            data["just_opened_pair"] = True
            size = self.side_scanner_size(best_z_spread)
            targets[active_rich] = -size
            targets[active_cheap] = size
            return targets, True

        if have_active_pair:
            size = self.side_scanner_size(max(active_pair_spread, self.SIDE_ENTRY_Z))
            targets[active_rich] = -size
            targets[active_cheap] = size

        return targets, bool(data.get("just_opened_pair", False))

    @classmethod
    def side_scanner_size(cls, z_spread: float) -> int:
        if z_spread < 2.90:
            return 2
        if z_spread < 3.70:
            return 3
        return cls.SIDE_MAX_TARGET

    # =====================================================================
    # 5-Pebble relative-z scanner.
    # =====================================================================
    def update_scanner(self, data: Dict[str, Any], mids: Dict[str, float]) -> Tuple[Dict[str, int], bool]:
        data.setdefault("mid_history", {})
        data.setdefault("active_rich", None)
        data.setdefault("active_cheap", None)
        data.setdefault("just_opened_pair", False)
        data["just_opened_pair"] = False

        for product in self.PRODUCTS:
            data["mid_history"].setdefault(product, [])
            data["mid_history"][product].append(mids[product])
            data["mid_history"][product] = data["mid_history"][product][-self.SCAN_WINDOW:]

        targets = {p: 0 for p in self.PRODUCTS}
        min_hist_len = min(len(data["mid_history"][p]) for p in self.SCAN_PRODUCTS)
        if min_hist_len < self.SCAN_MIN_HISTORY:
            return targets, False

        z_scores: Dict[str, float] = {}
        for product in self.SCAN_PRODUCTS:
            hist = data["mid_history"][product]
            z_scores[product] = (mids[product] - self.mean(hist)) / self.std_sample(hist)

        avg_z = self.mean(list(z_scores.values()))
        rel_z = {p: z_scores[p] - avg_z for p in self.SCAN_PRODUCTS}

        rich_product = max(self.SCAN_PRODUCTS, key=lambda p: rel_z[p])
        cheap_product = min(self.SCAN_PRODUCTS, key=lambda p: rel_z[p])
        best_z_spread = rel_z[rich_product] - rel_z[cheap_product]

        active_rich = data.get("active_rich")
        active_cheap = data.get("active_cheap")
        have_active_pair = active_rich in self.SCAN_PRODUCTS and active_cheap in self.SCAN_PRODUCTS
        active_pair_spread = rel_z[active_rich] - rel_z[active_cheap] if have_active_pair else 0.0

        data["last_rich_product"] = rich_product
        data["last_cheap_product"] = cheap_product
        data["last_best_z_spread"] = best_z_spread
        data["last_active_pair_spread"] = active_pair_spread
        data["last_rel_z"] = rel_z

        should_open = False
        should_close = False
        should_flip = False

        if not have_active_pair:
            if best_z_spread >= self.SCAN_ENTRY_Z:
                should_open = True
        else:
            if active_pair_spread <= self.SCAN_EXIT_Z:
                should_close = True
            elif (
                best_z_spread >= self.SCAN_ENTRY_Z
                and (rich_product != active_rich or cheap_product != active_cheap)
                and best_z_spread > active_pair_spread + self.SCAN_FLIP_BUFFER
            ):
                should_flip = True

        if should_close:
            data["active_rich"] = None
            data["active_cheap"] = None
            return targets, False

        if should_open or should_flip:
            active_rich = rich_product
            active_cheap = cheap_product
            data["active_rich"] = active_rich
            data["active_cheap"] = active_cheap
            data["just_opened_pair"] = True
            size = self.scanner_size(best_z_spread)
            targets[active_rich] = -size
            targets[active_cheap] = size
            return targets, True

        if have_active_pair:
            size = self.scanner_size(max(active_pair_spread, self.SCAN_ENTRY_Z))
            targets[active_rich] = -size
            targets[active_cheap] = size

        return targets, bool(data.get("just_opened_pair", False))

    @staticmethod
    def scanner_size(z_spread: float) -> int:
        if z_spread < 2.75:
            return 3
        if z_spread < 3.55:
            return 6
        return 10

    # =====================================================================
    # Execution helpers.
    # =====================================================================
    def take_toward_target(self, product: str, od: OrderDepth, current_pos: int, target_pos: int, max_size: int) -> List[Order]:
        orders: List[Order] = []
        diff = target_pos - current_pos
        if diff > 0:
            if not od.sell_orders:
                return orders
            best_ask = min(od.sell_orders.keys())
            ask_volume = -od.sell_orders[best_ask]
            qty = min(diff, ask_volume, max_size, self.POSITION_LIMIT - current_pos)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        elif diff < 0:
            if not od.buy_orders:
                return orders
            best_bid = max(od.buy_orders.keys())
            bid_volume = od.buy_orders[best_bid]
            qty = min(-diff, bid_volume, max_size, self.POSITION_LIMIT + current_pos)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        return orders

    def passive_toward_target(self, product: str, od: OrderDepth, current_pos: int, target_pos: int, max_size: int, improve: int) -> List[Order]:
        orders: List[Order] = []
        diff = target_pos - current_pos
        if not od.buy_orders or not od.sell_orders:
            return orders
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        if diff > 0:
            qty = min(diff, max_size, self.POSITION_LIMIT - current_pos)
            if qty <= 0:
                return orders
            price = best_bid + improve
            if price >= best_ask:
                price = best_bid
            orders.append(Order(product, price, qty))
        elif diff < 0:
            qty = min(-diff, max_size, self.POSITION_LIMIT + current_pos)
            if qty <= 0:
                return orders
            price = best_ask - improve
            if price <= best_bid:
                price = best_ask
            orders.append(Order(product, price, -qty))
        return orders

    @staticmethod
    def get_mid(od: OrderDepth) -> Optional[float]:
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def std(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

    @staticmethod
    def std_sample(values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        m = sum(values) / len(values)
        return math.sqrt(max(sum((x - m) ** 2 for x in values) / (len(values) - 1), 1e-6))

    @staticmethod
    def clamp(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData: str) -> Dict[str, Any]:
        if not traderData:
            return {}
        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
        return {}
