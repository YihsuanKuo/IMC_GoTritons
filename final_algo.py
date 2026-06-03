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
import statistics as _statistics


class Trader:
    OUTER_PREFIX = "Z:"

    def __init__(self):
        self.rest_of_follow = RestOfFollowTrader()
        self.uv_visor = UVVisorTradeflowTrader()
        self.translator = TranslatorTrader()
        self.robot = RobotTrader()
        self.pebbles = PebblesTrader()
        self.galaxy = GalaxyTrader()
        self.panel = PanelTrader()
        self.sleep = SleepTrader()
        self.snackpack = SnackpackTrader()

    def run(self, state):
        result = {}
        conversions = 0
        memory = self.load_outer_memory(state.traderData)
        strategies = [
            ("rest_of_follow", self.rest_of_follow),
            ("uv_visor_tradeflow", self.uv_visor),
            ("translator", self.translator),
            ("robot", self.robot),
            ("pebbles", self.pebbles),
            ("galaxy", self.galaxy),
            ("panel", self.panel),
            ("sleep", self.sleep),
            ("snackpack", self.snackpack),
        ]
        for key, strategy in strategies:
            sub_state = self.with_trader_data(state, memory.get(key, ""))
            sub_result, sub_conversions, sub_trader_data = strategy.run(
                sub_state
            )
            conversions += sub_conversions
            memory[key] = sub_trader_data or ""
            for product, orders in sub_result.items():
                if orders:
                    result.setdefault(product, []).extend(orders)
        traderData = self.dump_outer_memory(memory)
        return result, conversions, traderData

    @staticmethod
    def with_trader_data(state, trader_data):
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
    def dump_outer_memory(cls, memory):
        raw = json.dumps(memory, separators=(",", ":"), ensure_ascii=False)
        compressed = zlib.compress(raw.encode("utf-8"), level=9)
        return cls.OUTER_PREFIX + base64.b64encode(compressed).decode("ascii")

    @classmethod
    def load_outer_memory(cls, traderData):
        if not traderData:
            return {}
        if traderData.startswith(cls.OUTER_PREFIX):
            try:
                payload = traderData[len(cls.OUTER_PREFIX) :]
                raw = zlib.decompress(
                    base64.b64decode(payload.encode("ascii"))
                ).decode("utf-8")
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    return {
                        str(k): (v if isinstance(v, str) else "")
                        for k, v in decoded.items()
                    }
            except Exception:
                return {}
        try:
            decoded = json.loads(traderData)
            if isinstance(decoded, dict):
                return {
                    str(k): (v if isinstance(v, str) else "")
                    for k, v in decoded.items()
                }
        except Exception:
            pass
        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return {
                    str(k): (v if isinstance(v, str) else "")
                    for k, v in decoded.items()
                }
        except Exception:
            pass
        return {}


class RestOfFollowTrader:
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

    def run(self, state):
        result = {}
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
    ):
        depth = state.order_depths[product]
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
        ema_move = 0.70 * old_ema_move + 0.30 * mid_move
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
        old_fresh = memory["fresh_flow"].get(product, 0.0)
        fresh = 0.50 * old_fresh + raw_flow
        fresh = self.clip(fresh, -16.0, 16.0)
        memory["fresh_flow"][product] = fresh
        fresh_agrees = True
        if p["use_fresh_gate"]:
            fresh_agrees = (signal > 0 and fresh >= p["fresh_threshold"]) or (
                signal < 0 and fresh <= -p["fresh_threshold"]
            )
        fair = mid + p["flow_to_fair"] * signal - p["inventory_skew"] * pos
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
        abs_signal = abs(signal)
        size = p["base_size"] + int(abs_signal // 6)
        size = self.clip_int(size, 1, p["max_order_size"])
        if abs(pos) >= 7:
            size = min(size, 2)
        orders = []
        buy_capacity = limit - pos
        sell_capacity = limit + pos
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
        if (
            signal >= p["signal_threshold"]
            and fresh_agrees
            and buy_capacity > 0
        ):
            if (
                hypothetical_pos < int(0.85 * limit)
                and best_ask <= fair - take_edge
            ):
                qty = min(buy_capacity, ask_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    hypothetical_pos += qty
                    buy_capacity -= qty
        elif (
            signal <= -p["signal_threshold"]
            and fresh_agrees
            and sell_capacity > 0
        ):
            if (
                hypothetical_pos > -int(0.85 * limit)
                and best_bid >= fair + take_edge
            ):
                qty = min(sell_capacity, bid_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    hypothetical_pos -= qty
                    sell_capacity -= qty
        passive_size = size
        if p["use_fresh_gate"] and not fresh_agrees:
            passive_size = 1
        if spread >= 2:
            if (
                signal >= p["signal_threshold"]
                and buy_capacity > 0
                and hypothetical_pos < limit
            ):
                quote_price = min(
                    best_bid + 1, math.floor(fair - quote_edge / 2)
                )
                quote_price = min(quote_price, best_ask - 1)
                if best_bid <= quote_price < best_ask:
                    qty = min(buy_capacity, passive_size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, qty))
            elif (
                signal <= -p["signal_threshold"]
                and sell_capacity > 0
                and hypothetical_pos > -limit
            ):
                quote_price = max(
                    best_ask - 1, math.ceil(fair + quote_edge / 2)
                )
                quote_price = max(quote_price, best_bid + 1)
                if best_bid < quote_price <= best_ask:
                    qty = min(sell_capacity, passive_size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, -qty))
        return orders

    @staticmethod
    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def clip_int(x, lo, hi):
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData):
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
    }
    PARAMS = {
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

    def run(self, state):
        result = {}
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

    def trade_protected(self, product, state, memory):
        depth = state.order_depths[product]
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
        orders = []
        buy_capacity = limit - pos
        sell_capacity = limit + pos
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
        if signal >= p["signal_threshold"] and buy_capacity > 0:
            if (
                hypothetical_pos < int(0.85 * limit)
                and best_ask <= fair - take_edge
            ):
                qty = min(buy_capacity, ask_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    hypothetical_pos += qty
                    buy_capacity -= qty
        elif signal <= -p["signal_threshold"] and sell_capacity > 0:
            if (
                hypothetical_pos > -int(0.85 * limit)
                and best_bid >= fair + take_edge
            ):
                qty = min(sell_capacity, bid_vol, size)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    hypothetical_pos -= qty
                    sell_capacity -= qty
        if spread >= 2:
            if (
                signal >= p["signal_threshold"]
                and buy_capacity > 0
                and hypothetical_pos < limit
            ):
                quote_price = min(
                    best_bid + 1, math.floor(fair - quote_edge / 2)
                )
                quote_price = min(quote_price, best_ask - 1)
                if best_bid <= quote_price < best_ask:
                    qty = min(buy_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, qty))
            elif (
                signal <= -p["signal_threshold"]
                and sell_capacity > 0
                and hypothetical_pos > -limit
            ):
                quote_price = max(
                    best_ask - 1, math.ceil(fair + quote_edge / 2)
                )
                quote_price = max(quote_price, best_bid + 1)
                if best_bid < quote_price <= best_ask:
                    qty = min(sell_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, -qty))
        return orders

    def trade_simple(self, product, state, memory):
        depth = state.order_depths[product]
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
        if abs(pos) >= 8:
            size = min(size, 2)
        orders = []
        buy_capacity = limit - pos
        sell_capacity = limit + pos
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
        if spread >= 2:
            if signal >= p["signal_threshold"] and buy_capacity > 0:
                quote_price = min(
                    best_bid + 1, math.floor(fair - p["quote_edge"] / 2)
                )
                quote_price = min(quote_price, best_ask - 1)
                if best_bid <= quote_price < best_ask:
                    qty = min(buy_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, qty))
            elif signal <= -p["signal_threshold"] and sell_capacity > 0:
                quote_price = max(
                    best_ask - 1, math.ceil(fair + p["quote_edge"] / 2)
                )
                quote_price = max(quote_price, best_bid + 1)
                if best_bid < quote_price <= best_ask:
                    qty = min(sell_capacity, size)
                    if qty > 0:
                        orders.append(Order(product, quote_price, -qty))
        return orders

    @staticmethod
    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def clip_int(x, lo, hi):
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData):
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
    TOP_SIZE = 10
    SECOND_SIZE = 5
    BOTTOM_SIZE = -10
    SECOND_BOTTOM_SIZE = -5
    MIN_SCORE_GAP = 0.20

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.cached_targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

    def bid(self):
        return 15

    def get_mid(self, order_depth):
        if (
            len(order_depth.buy_orders) == 0
            or len(order_depth.sell_orders) == 0
        ):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def mean(self, xs):
        return sum(xs) / len(xs)

    def std(self, xs):
        if len(xs) <= 1:
            return 0.0
        m = self.mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def clamp_position(self, x):
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    def update_history(self, state):
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue
            self.mid_history[product].append(mid)
            if len(self.mid_history[product]) > 1500:
                self.mid_history[product] = self.mid_history[product][-1500:]

    def product_score(self, product):
        hist = self.mid_history[product]
        if len(hist) < max(self.LOOKBACK, self.VOL_LOOKBACK) + 1:
            return None
        now = hist[-1]
        old = hist[-self.LOOKBACK]
        if old <= 0:
            return None
        ret = (now - old) / old
        recent = hist[-self.VOL_LOOKBACK :]
        pct_changes = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                pct_changes.append((recent[i] - recent[i - 1]) / recent[i - 1])
        vol = self.std(pct_changes)
        if vol <= 1e-9:
            return None
        return ret / (vol * math.sqrt(self.LOOKBACK))

    def build_targets(self):
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

    def get_targets(self):
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets()
        return self.cached_targets

    def move_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ):
        orders = []
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

    def run(self, state):
        result = {p: [] for p in self.PRODUCTS}
        conversions = 0
        traderData = ""
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
        return result, conversions, traderData


class RobotTrader:
    def __init__(self):
        self.ALL_ROBOTS = [
            "ROBOT_DISHES",
            "ROBOT_IRONING",
            "ROBOT_LAUNDRY",
            "ROBOT_MOPPING",
            "ROBOT_VACUUMING",
        ]
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
        self.window = 90
        self.min_history = 35
        self.exit_z = 0.35
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
        self.PRODUCT_PRIORITY = {
            "ROBOT_LAUNDRY": 1.20,
            "ROBOT_VACUUMING": 1.00,
            "ROBOT_DISHES": 0.85,
        }
        self.flow_decay = 0.70
        self.flow_entry = 5.0
        self.flow_strong = 10.0
        self.passive_size = 2
        self.active_size = 1
        self.max_orders_per_tick = 2

    def pos(self, state, product):
        return state.position.get(product, 0)

    def best_bid(self, depth):
        if not depth.buy_orders:
            return None
        return max(depth.buy_orders.keys())

    def best_ask(self, depth):
        if not depth.sell_orders:
            return None
        return min(depth.sell_orders.keys())

    def mid(self, depth):
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

    def net_ordered(self, orders):
        return sum(o.quantity for o in orders)

    def buy_cap(self, state, product, orders):
        current = self.pos(state, product)
        already = self.net_ordered(orders)
        projected = current + already
        return self.LIMIT[product] - projected

    def sell_cap(self, state, product, orders):
        current = self.pos(state, product)
        already = self.net_ordered(orders)
        projected = current + already
        return self.LIMIT[product] + projected

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

    def reduce_position_passive(
        self, orders, state, product, depth, max_qty=1
    ):
        p = self.pos(state, product)
        if p > 0:
            self.passive_sell(orders, state, product, depth, min(max_qty, p))
        elif p < 0:
            self.passive_buy(orders, state, product, depth, min(max_qty, -p))

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
        lookback = hist[-self.window :] if len(hist) >= self.window else hist
        mean, std = self.mean_std(lookback)
        return (hist[-1] - mean) / std

    def run(self, state):
        result = {}
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        for p in self.ALL_ROBOTS:
            result[p] = []
        for p in self.ALL_ROBOTS:
            if p not in state.order_depths:
                return result, 0, json.dumps(data)
        mids = {}
        for p in self.ALL_ROBOTS:
            m = self.mid(state.order_depths[p])
            if m is None:
                return result, 0, json.dumps(data)
            mids[p] = m
        robot_avg = sum(mids[p] for p in self.ALL_ROBOTS) / len(
            self.ALL_ROBOTS
        )
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
        for p in self.TRADE_PRODUCTS:
            hist = dev_hist[p]
            z = self.calc_z(hist)
            entry_z = self.entry_z_by_product[p]
            signal = 0.0
            if z > entry_z:
                signal = -abs(z)
            elif z < -entry_z:
                signal = abs(z)
            if p == self.DISHES and signal != 0:
                if dishes_flow > self.flow_entry:
                    signal += 0.75
                elif dishes_flow < -self.flow_entry:
                    signal -= 0.75
            if abs(z) < self.exit_z:
                candidates.append(("exit", 0.0, p, z))
            elif signal != 0:
                strength = abs(signal) * self.PRODUCT_PRIORITY.get(p, 1.0)
                candidates.append(("trade", strength, p, z))
        trade_candidates = [c for c in candidates if c[0] == "trade"]
        exit_candidates = [c for c in candidates if c[0] == "exit"]
        trade_candidates.sort(key=lambda x: x[1], reverse=True)
        trades_sent = 0
        for _, strength, p, z in trade_candidates:
            if trades_sent >= self.max_orders_per_tick:
                break
            depth = state.order_depths[p]
            orders = result[p]
            entry_z = self.entry_z_by_product[p]
            strong_z = self.strong_z_by_product[p]
            if z > entry_z:
                self.passive_sell(orders, state, p, depth, self.passive_size)
                if z > strong_z:
                    self.active_sell(orders, state, p, depth, self.active_size)
                trades_sent += 1
            elif z < -entry_z:
                self.passive_buy(orders, state, p, depth, self.passive_size)
                if z < -strong_z:
                    self.active_buy(orders, state, p, depth, self.active_size)
                trades_sent += 1
        if trades_sent < self.max_orders_per_tick:
            p = self.DISHES
            depth = state.order_depths[p]
            orders = result[p]
            z = self.calc_z(dev_hist[p])
            entry_z = self.entry_z_by_product[p]
            if z < -entry_z and dishes_flow > self.flow_strong:
                self.passive_buy(orders, state, p, depth, 1)
                trades_sent += 1
            elif z > entry_z and dishes_flow < -self.flow_strong:
                self.passive_sell(orders, state, p, depth, 1)
                trades_sent += 1
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
    SIDE_SCAN_PRODUCTS = [
        "PEBBLES_XL",
        "PEBBLES_S",
    ]
    SIDE_ENTRY_Z = 2.10
    SIDE_EXIT_Z = 0.90
    SIDE_FLIP_BUFFER = 0.80
    SIDE_PASSIVE_SIZE = 1
    SIDE_MAX_TARGET = 4
    SIDE_NORMAL_IMPROVE = 1
    SIDE_ENTRY_IMPROVE = 2

    def run(self, state):
        result = {p: [] for p in self.PRODUCTS}
        conversions = 0
        memory = self.load_memory(state.traderData)
        memory.setdefault("xl", {})
        memory.setdefault("scanner", {})
        memory.setdefault("side_scanner", {})
        memory.setdefault("owner", "scanner")
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                return result, conversions, jsonpickle.encode(memory)
        mids = {}
        for product in self.PRODUCTS:
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                return result, conversions, jsonpickle.encode(memory)
            mids[product] = mid
        xl_mode, xl_ready = self.update_xs_l_specialist(memory["xl"], mids)
        scanner_targets, scanner_just_opened = self.update_scanner(
            memory["scanner"], mids
        )
        side_targets, side_just_opened = self.update_side_scanner(
            memory["side_scanner"], mids
        )
        positions = {p: state.position.get(p, 0) for p in self.PRODUCTS}
        targets = {p: 0 for p in self.PRODUCTS}
        use_specialist_execution = False
        xs_l_inventory = (
            abs(positions[self.XS]) > 0 or abs(positions[self.L]) > 0
        )
        if xl_mode != 0:
            memory["owner"] = "xs_l"
            memory["scanner"]["active_rich"] = None
            memory["scanner"]["active_cheap"] = None
        elif memory.get("owner") == "xs_l" and xs_l_inventory:
            pass
        else:
            memory["owner"] = "scanner"
        if memory.get("owner") == "xs_l":
            use_specialist_execution = True
            if xl_mode == -1:
                targets[self.XS] = -self.XS_L_TARGET_SIZE
                targets[self.L] = self.XS_L_TARGET_SIZE
                targets["PEBBLES_XL"] = side_targets.get("PEBBLES_XL", 0)
                targets["PEBBLES_S"] = side_targets.get("PEBBLES_S", 0)
            elif xl_mode == 1:
                targets[self.XS] = self.XS_L_TARGET_SIZE
                targets[self.L] = -self.XS_L_TARGET_SIZE
                targets["PEBBLES_XL"] = side_targets.get("PEBBLES_XL", 0)
                targets["PEBBLES_S"] = side_targets.get("PEBBLES_S", 0)
            else:
                targets = {p: 0 for p in self.PRODUCTS}
        else:
            targets = scanner_targets
        if use_specialist_execution:
            for product in self.PRODUCTS:
                current_pos = positions[product]
                target_pos = self.clamp(
                    targets[product], -self.POSITION_LIMIT, self.POSITION_LIMIT
                )
                if product in (self.XS, self.L):
                    result[product] += self.take_toward_target(
                        product,
                        state.order_depths[product],
                        current_pos,
                        target_pos,
                        self.XS_L_ORDER_SIZE,
                    )
                else:
                    if product in self.SIDE_SCAN_PRODUCTS:
                        side_improve = (
                            self.SIDE_ENTRY_IMPROVE
                            if side_just_opened
                            else self.SIDE_NORMAL_IMPROVE
                        )
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
            improve = (
                self.SCAN_ENTRY_IMPROVE
                if scanner_just_opened
                else self.SCAN_NORMAL_IMPROVE
            )
            for product in self.PRODUCTS:
                result[product] += self.passive_toward_target(
                    product,
                    state.order_depths[product],
                    positions[product],
                    self.clamp(
                        targets[product],
                        -self.POSITION_LIMIT,
                        self.POSITION_LIMIT,
                    ),
                    self.SCAN_PASSIVE_SIZE,
                    improve,
                )
        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData

    def update_xs_l_specialist(self, data, mids):
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
        max_hist = (
            max(self.XL_WINDOW, self.ACCEL_LONG, self.SHORT_VOL_WINDOW) + 5
        )
        data["spread_hist"] = data["spread_hist"][-max_hist:]
        data["raw_hist"] = data["raw_hist"][-max_hist:]
        hist = data["spread_hist"]
        raw_hist = data["raw_hist"]
        if len(hist) < self.XL_WINDOW or len(raw_hist) < self.XL_WINDOW:
            data["last_ready"] = False
            return int(data.get("mode", 0)), False
        long_hist = hist[-self.XL_WINDOW :]
        mean = self.mean(long_hist)
        std = self.std(long_hist)
        if std < 1e-9:
            data["last_ready"] = False
            return int(data.get("mode", 0)), False
        z = (spread - mean) / std
        raw_std = self.std(raw_hist[-self.XL_WINDOW :])
        if raw_std < 1e-9:
            raw_std = 1.0
        short_vol = self.std(hist[-self.SHORT_VOL_WINDOW :])
        vol_ratio = short_vol / std if std > 1e-9 else 1.0
        accel_z = 0.0
        if len(hist) > self.ACCEL_LONG:
            recent_change = hist[-1] - hist[-1 - self.ACCEL_SHORT]
            older_change = (
                hist[-1 - self.ACCEL_SHORT] - hist[-1 - self.ACCEL_LONG]
            )
            accel_z = (recent_change - older_change) / std
        mode = int(data.get("mode", 0))
        if mode == 0:
            data["hold_bars"] = 0
            if not data.get("reset_ready", True):
                if abs(z) < self.RESET_Z:
                    data["reset_ready"] = True
            if data.get("reset_ready", True):
                if (
                    z > self.SHORT_ENTRY_Z
                    and accel_z <= self.SHORT_MAX_ACCEL_Z
                ):
                    mode = -1
                    self.start_xl_trade(data, raw_diff, raw_std)
                elif (
                    z < -self.LONG_ENTRY_Z
                    and vol_ratio >= self.LONG_MIN_VOL_RATIO
                ):
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
                if (
                    favorable_move
                    >= self.RAW_TRAIL_ACTIVATE_STD * entry_raw_std
                ):
                    trail_active = True
                if (
                    trail_active
                    and retrace >= self.RAW_TRAIL_RETRACE_STD * raw_std
                ):
                    mode = 0
                    exit_reason = "raw_trail"
            elif mode == 1:
                if raw_diff > best_raw:
                    best_raw = raw_diff
                favorable_move = best_raw - entry_raw
                retrace = best_raw - raw_diff
                if (
                    favorable_move
                    >= self.RAW_TRAIL_ACTIVATE_STD * entry_raw_std
                ):
                    trail_active = True
                if (
                    trail_active
                    and retrace >= self.RAW_TRAIL_RETRACE_STD * raw_std
                ):
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
                data["reset_ready"] = (
                    True if exit_reason == "z_exit" else False
                )
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
    def start_xl_trade(data, raw_diff, raw_std):
        data["hold_bars"] = 0
        data["entry_raw_diff"] = raw_diff
        data["best_raw_diff"] = raw_diff
        data["entry_raw_std"] = raw_std
        data["raw_trail_active"] = False

    def update_side_scanner(self, data, mids):
        data.setdefault("mid_history", {})
        data.setdefault("active_rich", None)
        data.setdefault("active_cheap", None)
        data.setdefault("just_opened_pair", False)
        data["just_opened_pair"] = False
        for product in self.SIDE_SCAN_PRODUCTS:
            data["mid_history"].setdefault(product, [])
            data["mid_history"][product].append(mids[product])
            data["mid_history"][product] = data["mid_history"][product][
                -self.SCAN_WINDOW :
            ]
        targets = {p: 0 for p in self.PRODUCTS}
        min_hist_len = min(
            len(data["mid_history"][p]) for p in self.SIDE_SCAN_PRODUCTS
        )
        if min_hist_len < self.SCAN_MIN_HISTORY:
            return targets, False
        z_scores = {}
        for product in self.SIDE_SCAN_PRODUCTS:
            hist = data["mid_history"][product]
            z_scores[product] = (
                mids[product] - self.mean(hist)
            ) / self.std_sample(hist)
        avg_z = self.mean(list(z_scores.values()))
        rel_z = {p: z_scores[p] - avg_z for p in self.SIDE_SCAN_PRODUCTS}
        rich_product = max(self.SIDE_SCAN_PRODUCTS, key=lambda p: rel_z[p])
        cheap_product = min(self.SIDE_SCAN_PRODUCTS, key=lambda p: rel_z[p])
        best_z_spread = rel_z[rich_product] - rel_z[cheap_product]
        active_rich = data.get("active_rich")
        active_cheap = data.get("active_cheap")
        have_active_pair = (
            active_rich in self.SIDE_SCAN_PRODUCTS
            and active_cheap in self.SIDE_SCAN_PRODUCTS
        )
        active_pair_spread = (
            rel_z[active_rich] - rel_z[active_cheap]
            if have_active_pair
            else 0.0
        )
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
                and (
                    rich_product != active_rich
                    or cheap_product != active_cheap
                )
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
            size = self.side_scanner_size(
                max(active_pair_spread, self.SIDE_ENTRY_Z)
            )
            targets[active_rich] = -size
            targets[active_cheap] = size
        return targets, bool(data.get("just_opened_pair", False))

    @classmethod
    def side_scanner_size(cls, z_spread):
        if z_spread < 2.90:
            return 2
        if z_spread < 3.70:
            return 3
        return cls.SIDE_MAX_TARGET

    def update_scanner(self, data, mids):
        data.setdefault("mid_history", {})
        data.setdefault("active_rich", None)
        data.setdefault("active_cheap", None)
        data.setdefault("just_opened_pair", False)
        data["just_opened_pair"] = False
        for product in self.PRODUCTS:
            data["mid_history"].setdefault(product, [])
            data["mid_history"][product].append(mids[product])
            data["mid_history"][product] = data["mid_history"][product][
                -self.SCAN_WINDOW :
            ]
        targets = {p: 0 for p in self.PRODUCTS}
        min_hist_len = min(
            len(data["mid_history"][p]) for p in self.SCAN_PRODUCTS
        )
        if min_hist_len < self.SCAN_MIN_HISTORY:
            return targets, False
        z_scores = {}
        for product in self.SCAN_PRODUCTS:
            hist = data["mid_history"][product]
            z_scores[product] = (
                mids[product] - self.mean(hist)
            ) / self.std_sample(hist)
        avg_z = self.mean(list(z_scores.values()))
        rel_z = {p: z_scores[p] - avg_z for p in self.SCAN_PRODUCTS}
        rich_product = max(self.SCAN_PRODUCTS, key=lambda p: rel_z[p])
        cheap_product = min(self.SCAN_PRODUCTS, key=lambda p: rel_z[p])
        best_z_spread = rel_z[rich_product] - rel_z[cheap_product]
        active_rich = data.get("active_rich")
        active_cheap = data.get("active_cheap")
        have_active_pair = (
            active_rich in self.SCAN_PRODUCTS
            and active_cheap in self.SCAN_PRODUCTS
        )
        active_pair_spread = (
            rel_z[active_rich] - rel_z[active_cheap]
            if have_active_pair
            else 0.0
        )
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
                and (
                    rich_product != active_rich
                    or cheap_product != active_cheap
                )
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
            size = self.scanner_size(
                max(active_pair_spread, self.SCAN_ENTRY_Z)
            )
            targets[active_rich] = -size
            targets[active_cheap] = size
        return targets, bool(data.get("just_opened_pair", False))

    @staticmethod
    def scanner_size(z_spread):
        if z_spread < 2.75:
            return 3
        if z_spread < 3.55:
            return 6
        return 10

    def take_toward_target(
        self, product, od, current_pos, target_pos, max_size
    ):
        orders = []
        diff = target_pos - current_pos
        if diff > 0:
            if not od.sell_orders:
                return orders
            best_ask = min(od.sell_orders.keys())
            ask_volume = -od.sell_orders[best_ask]
            qty = min(
                diff, ask_volume, max_size, self.POSITION_LIMIT - current_pos
            )
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        elif diff < 0:
            if not od.buy_orders:
                return orders
            best_bid = max(od.buy_orders.keys())
            bid_volume = od.buy_orders[best_bid]
            qty = min(
                -diff, bid_volume, max_size, self.POSITION_LIMIT + current_pos
            )
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        return orders

    def passive_toward_target(
        self, product, od, current_pos, target_pos, max_size, improve
    ):
        orders = []
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
    def get_mid(od):
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def std(values):
        if len(values) <= 1:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

    @staticmethod
    def std_sample(values):
        if len(values) < 2:
            return 1.0
        m = sum(values) / len(values)
        return math.sqrt(
            max(sum((x - m) ** 2 for x in values) / (len(values) - 1), 1e-6)
        )

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData):
        if not traderData:
            return {}
        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
        return {}


class GalaxyTrader:
    POSITION_LIMIT = 10
    PRODUCTS = [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ]
    RINGS = "GALAXY_SOUNDS_PLANETARY_RINGS"
    SCOUT_PRODUCTS = [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ]
    RINGS_BASE_SHORT = -8
    RINGS_STRONG_SHORT = -10
    RINGS_REDUCED_SHORT = -4
    RINGS_FLAT = 0
    RINGS_SCORE_LOOKBACK = 160
    RINGS_VOL_LOOKBACK = 90
    RINGS_STRONG_SHORT_SCORE = -0.60
    RINGS_REDUCE_SHORT_SCORE = 0.90
    RINGS_FLAT_SCORE = 1.50
    ENABLE_SCOUTS = True
    FAST_LOOKBACK = 80
    SLOW_LOOKBACK = 240
    VOL_LOOKBACK = 100
    REBALANCE_EVERY = 25
    SCOUT_ENTRY_SCORE = 1.20
    SCOUT_STRONG_SCORE = 1.80
    SCOUT_EXIT_SCORE = 0.35
    MIN_SCORE_SPREAD = 0.90
    CONFIRM_BARS = 3
    SCOUT_SMALL_SIZE = 1
    SCOUT_STRONG_SIZE = 2
    MAX_SCOUT_PRODUCTS = 1
    MAX_GROSS_EXPOSURE = 14
    MAX_TRADE_SIZE = 3
    SCOUT_MAX_TRADE_SIZE = 1

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.score_history: Dict[str, List[float]] = {
            p: [] for p in self.SCOUT_PRODUCTS
        }
        self.cached_targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

    def bid(self):
        return 15

    def get_mid(self, order_depth):
        if (
            len(order_depth.buy_orders) == 0
            or len(order_depth.sell_orders) == 0
        ):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def mean(self, xs):
        return sum(xs) / len(xs)

    def std(self, xs):
        if len(xs) <= 1:
            return 0.0
        m = self.mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def clamp_position(self, x):
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    def update_history(self, state):
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue
            self.mid_history[product].append(mid)
            if len(self.mid_history[product]) > 2000:
                self.mid_history[product] = self.mid_history[product][-2000:]

    def z_momentum_score(self, product: str, lookback: int, vol_lookback: int):
        hist = self.mid_history[product]
        need = max(lookback, vol_lookback) + 1
        if len(hist) < need:
            return None
        now = hist[-1]
        old = hist[-lookback]
        if old <= 0:
            return None
        ret = (now - old) / old
        recent = hist[-vol_lookback:]
        pct_changes = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                pct_changes.append((recent[i] - recent[i - 1]) / recent[i - 1])
        vol = self.std(pct_changes)
        if vol <= 1e-12:
            return None
        return ret / (vol * math.sqrt(lookback))

    def combined_score(self, product):
        fast = self.z_momentum_score(
            product, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        slow = self.z_momentum_score(
            product, self.SLOW_LOOKBACK, self.VOL_LOOKBACK
        )
        if fast is None and slow is None:
            return None
        if slow is None:
            return fast
        if fast is None:
            return slow
        return 0.70 * slow + 0.30 * fast

    def rings_target(self):
        score = self.z_momentum_score(
            self.RINGS,
            self.RINGS_SCORE_LOOKBACK,
            self.RINGS_VOL_LOOKBACK,
        )
        if score is None:
            return self.RINGS_BASE_SHORT
        if score >= self.RINGS_FLAT_SCORE:
            return self.RINGS_FLAT
        if score >= self.RINGS_REDUCE_SHORT_SCORE:
            return self.RINGS_REDUCED_SHORT
        if score <= self.RINGS_STRONG_SHORT_SCORE:
            return self.RINGS_STRONG_SHORT
        return self.RINGS_BASE_SHORT

    def update_score_history(self, scores):
        for product, score in scores.items():
            self.score_history[product].append(score)
            if len(self.score_history[product]) > 20:
                self.score_history[product] = self.score_history[product][-20:]

    def confirmed_direction(self, product, direction):
        hist = self.score_history.get(product, [])
        if len(hist) < self.CONFIRM_BARS:
            return False
        recent = hist[-self.CONFIRM_BARS :]
        if direction > 0:
            return all(x > 0 for x in recent)
        return all(x < 0 for x in recent)

    def build_targets(self, state):
        targets = {p: 0 for p in self.PRODUCTS}
        targets[self.RINGS] = self.rings_target()
        if not self.ENABLE_SCOUTS:
            return targets
        scores = {}
        for product in self.SCOUT_PRODUCTS:
            s = self.combined_score(product)
            if s is not None:
                scores[product] = s
        if len(scores) < 3:
            return targets
        self.update_score_history(scores)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_product, best_score = ranked[0]
        worst_product, worst_score = ranked[-1]
        if best_score - worst_score < self.MIN_SCORE_SPREAD:
            return targets
        candidates = []
        if best_score >= self.SCOUT_ENTRY_SCORE and self.confirmed_direction(
            best_product, +1
        ):
            size = (
                self.SCOUT_STRONG_SIZE
                if best_score >= self.SCOUT_STRONG_SCORE
                else self.SCOUT_SMALL_SIZE
            )
            candidates.append((abs(best_score), best_product, size))
        if (
            worst_score <= -self.SCOUT_ENTRY_SCORE
            and self.confirmed_direction(worst_product, -1)
        ):
            size = (
                -self.SCOUT_STRONG_SIZE
                if worst_score <= -self.SCOUT_STRONG_SCORE
                else -self.SCOUT_SMALL_SIZE
            )
            candidates.append((abs(worst_score), worst_product, size))
        candidates.sort(reverse=True)
        for _, product, size in candidates[: self.MAX_SCOUT_PRODUCTS]:
            targets[product] = size
        for product in self.SCOUT_PRODUCTS:
            current_pos = state.position.get(product, 0)
            if current_pos != 0 and product not in [
                c[1] for c in candidates[: self.MAX_SCOUT_PRODUCTS]
            ]:
                s = scores.get(product, 0.0)
                if abs(s) > self.SCOUT_EXIT_SCORE:
                    if current_pos > 0 and s > 0:
                        targets[product] = min(
                            current_pos, self.SCOUT_SMALL_SIZE
                        )
                    elif current_pos < 0 and s < 0:
                        targets[product] = max(
                            current_pos, -self.SCOUT_SMALL_SIZE
                        )
                    else:
                        targets[product] = 0
                else:
                    targets[product] = 0
        for product in self.PRODUCTS:
            targets[product] = self.clamp_position(targets[product])
        gross = sum(abs(x) for x in targets.values())
        if gross > self.MAX_GROSS_EXPOSURE:
            excess = gross - self.MAX_GROSS_EXPOSURE
            non_rings = sorted(
                [p for p in self.SCOUT_PRODUCTS if targets[p] != 0],
                key=lambda p: abs(targets[p]),
                reverse=True,
            )
            for product in non_rings:
                if excess <= 0:
                    break
                reduce_by = min(abs(targets[product]), excess)
                if targets[product] > 0:
                    targets[product] -= reduce_by
                else:
                    targets[product] += reduce_by
                excess -= reduce_by
        return targets

    def get_targets(self, state):
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets(state)
        return self.cached_targets

    def product_trade_cap(self, product):
        if product == self.RINGS:
            return self.MAX_TRADE_SIZE
        return self.SCOUT_MAX_TRADE_SIZE

    def move_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ):
        orders = []
        target_pos = self.clamp_position(target_pos)
        needed = target_pos - current_pos
        if needed == 0:
            return orders
        trade_cap = self.product_trade_cap(product)
        if needed > 0:
            remaining = min(
                needed, self.POSITION_LIMIT - current_pos, trade_cap
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
                -needed, self.POSITION_LIMIT + current_pos, trade_cap
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

    def run(self, state):
        result = {p: [] for p in self.PRODUCTS}
        conversions = 0
        traderData = ""
        self.tick += 1
        self.update_history(state)
        targets = self.get_targets(state)
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
        return result, conversions, traderData


class PanelTrader:
    POSITION_LIMIT = 10
    PRODUCTS = [
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",
    ]
    PRIOR_WEIGHTS = {
        "PANEL_2X4": 1,
        "PANEL_2X2": -1,
        "PANEL_4X4": -1,
    }
    PRIOR_BASE_SIZE = 7
    LOOKBACK = 250
    VOL_LOOKBACK = 120
    REBALANCE_EVERY = 25
    MIN_SCORE_SPREAD = 0.25
    DYN_TOP_SIZE = 4
    DYN_SECOND_SIZE = 2
    DYN_BOTTOM_SIZE = -4
    DYN_SECOND_BOTTOM_SIZE = -2
    MAX_TARGET_ABS = 10
    BASKET_LOOKBACK = 300
    LEG_LOOKBACK = 250
    REGIME_VOL_LOOKBACK = 120
    BASKET_WEAK_SCORE = -1.25
    BASKET_BAD_SCORE = -2.25
    LEG_WEAK_SCORE = -1.50
    LEG_BAD_SCORE = -2.50
    MAX_TRADE_SIZE = 3

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.basket_history: List[float] = []
        self.cached_targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

    def bid(self):
        return 15

    def get_mid(self, order_depth):
        if (
            len(order_depth.buy_orders) == 0
            or len(order_depth.sell_orders) == 0
        ):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def mean(self, xs):
        return sum(xs) / len(xs)

    def std(self, xs):
        if len(xs) <= 1:
            return 0.0
        m = self.mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def clamp_position(self, x):
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    def update_history(self, state):
        mids = {}
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue
            mids[product] = mid
            self.mid_history[product].append(mid)
            if len(self.mid_history[product]) > 2000:
                self.mid_history[product] = self.mid_history[product][-2000:]
        if all(p in mids for p in self.PRIOR_WEIGHTS):
            val = 0.0
            for product, weight in self.PRIOR_WEIGHTS.items():
                val += weight * mids[product]
            self.basket_history.append(val)
            if len(self.basket_history) > 2000:
                self.basket_history = self.basket_history[-2000:]

    def risk_adjusted_momentum(self, product):
        hist = self.mid_history[product]
        if len(hist) < max(self.LOOKBACK, self.VOL_LOOKBACK) + 1:
            return None
        now = hist[-1]
        old = hist[-self.LOOKBACK]
        if old <= 0:
            return None
        ret = (now - old) / old
        recent = hist[-self.VOL_LOOKBACK :]
        pct_changes = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            if prev > 0:
                pct_changes.append((recent[i] - prev) / prev)
        vol = self.std(pct_changes)
        if vol <= 1e-9:
            return None
        return ret / (vol * math.sqrt(self.LOOKBACK))

    def basket_health_score(self):
        hist = self.basket_history
        if len(hist) < max(self.BASKET_LOOKBACK, self.REGIME_VOL_LOOKBACK) + 1:
            return 0.0
        move = hist[-1] - hist[-self.BASKET_LOOKBACK]
        recent = hist[-self.REGIME_VOL_LOOKBACK :]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        vol = self.std(changes)
        if vol <= 1e-9:
            return 0.0
        return move / (vol * math.sqrt(self.BASKET_LOOKBACK))

    def leg_support_score(self, product: str, intended_direction: int):
        hist = self.mid_history[product]
        if len(hist) < max(self.LEG_LOOKBACK, self.REGIME_VOL_LOOKBACK) + 1:
            return 0.0
        raw_move = hist[-1] - hist[-self.LEG_LOOKBACK]
        favorable_move = intended_direction * raw_move
        recent = hist[-self.REGIME_VOL_LOOKBACK :]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        vol = self.std(changes)
        if vol <= 1e-9:
            return 0.0
        return favorable_move / (vol * math.sqrt(self.LEG_LOOKBACK))

    def static_scale(self):
        score = self.basket_health_score()
        if score <= self.BASKET_BAD_SCORE:
            return 0.40
        if score <= self.BASKET_WEAK_SCORE:
            return 0.70
        return 1.00

    def leg_scale(self, product, intended_direction):
        score = self.leg_support_score(product, intended_direction)
        if score <= self.LEG_BAD_SCORE:
            return 0.40
        if score <= self.LEG_WEAK_SCORE:
            return 0.70
        return 1.00

    def static_prior_targets(self):
        targets = {p: 0 for p in self.PRODUCTS}
        basket_scale = self.static_scale()
        for product, direction in self.PRIOR_WEIGHTS.items():
            scale = basket_scale * self.leg_scale(product, direction)
            targets[product] = self.clamp_position(
                direction * self.PRIOR_BASE_SIZE * scale
            )
        return targets

    def dynamic_ranking_targets(self):
        targets = {p: 0 for p in self.PRODUCTS}
        scores = {}
        for product in self.PRODUCTS:
            s = self.risk_adjusted_momentum(product)
            if s is not None:
                scores[product] = s
        if len(scores) < len(self.PRODUCTS):
            return targets
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_product, best_score = ranked[0]
        second_product, second_score = ranked[1]
        middle_product, middle_score = ranked[2]
        second_worst_product, second_worst_score = ranked[3]
        worst_product, worst_score = ranked[4]
        if best_score - worst_score < self.MIN_SCORE_SPREAD:
            return targets
        targets[best_product] += self.DYN_TOP_SIZE
        targets[second_product] += self.DYN_SECOND_SIZE
        targets[middle_product] += 0
        targets[second_worst_product] += self.DYN_SECOND_BOTTOM_SIZE
        targets[worst_product] += self.DYN_BOTTOM_SIZE
        return {p: self.clamp_position(v) for p, v in targets.items()}

    def build_targets(self):
        targets = self.static_prior_targets()
        dyn = self.dynamic_ranking_targets()
        for product in self.PRODUCTS:
            targets[product] = self.clamp_position(
                targets.get(product, 0) + dyn.get(product, 0)
            )
        return targets

    def get_targets(self):
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets()
        return self.cached_targets

    def move_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ):
        orders = []
        target_pos = self.clamp_position(target_pos)
        needed = target_pos - current_pos
        if needed == 0:
            return orders
        if needed > 0:
            remaining = min(
                needed, self.POSITION_LIMIT - current_pos, self.MAX_TRADE_SIZE
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
                -needed, self.POSITION_LIMIT + current_pos, self.MAX_TRADE_SIZE
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

    def run(self, state):
        result = {p: [] for p in self.PRODUCTS}
        conversions = 0
        traderData = ""
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
        return result, conversions, traderData


class SleepTrader:
    POSITION_LIMIT = 10
    PRODUCTS = [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ]
    COTTON = "SLEEP_POD_COTTON"
    WOOL = "SLEEP_POD_LAMB_WOOL"
    COTTON_BASE_LONG = 8
    COTTON_STRONG_LONG = 10
    COTTON_REDUCED_LONG = 4
    COTTON_FLAT = 0
    COTTON_SMALL_SHORT = -2
    WOOL_BASE_SHORT = -8
    WOOL_STRONG_SHORT = -10
    WOOL_REDUCED_SHORT = -4
    WOOL_FLAT = 0
    WOOL_SMALL_LONG = 2
    FAST_LOOKBACK = 50
    SLOW_LOOKBACK = 180
    VOL_LOOKBACK = 85
    SHOCK_LOOKBACK = 18
    Z_LOOKBACK = 260
    REBALANCE_EVERY = 20
    COTTON_STRONG_SCORE = 0.65
    COTTON_REDUCE_SCORE = -0.55
    COTTON_FLAT_SCORE = -1.05
    COTTON_SHORT_SCORE = -1.75
    WOOL_STRONG_SHORT_SCORE = -0.65
    WOOL_REDUCE_SHORT_SCORE = 0.55
    WOOL_FLAT_SCORE = 1.05
    WOOL_LONG_SCORE = 1.75
    EXTREME_Z = 2.40
    MAX_GROSS_EXPOSURE = 18
    MAX_NET_EXPOSURE = 6
    MAX_TRADE_SIZE = 3

    def __init__(self):
        self.tick = 0
        self.mid_history: Dict[str, List[float]] = {
            p: [] for p in self.PRODUCTS
        }
        self.cached_targets: Dict[str, int] = {p: 0 for p in self.PRODUCTS}

    def bid(self):
        return 15

    def get_mid(self, order_depth):
        if (
            len(order_depth.buy_orders) == 0
            or len(order_depth.sell_orders) == 0
        ):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def mean(self, xs):
        return sum(xs) / len(xs)

    def std(self, xs):
        if len(xs) <= 1:
            return 0.0
        m = self.mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    def clamp_position(self, x):
        return max(
            -self.POSITION_LIMIT, min(self.POSITION_LIMIT, int(round(x)))
        )

    def update_history(self, state):
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                continue
            self.mid_history[product].append(mid)
            if len(self.mid_history[product]) > 2500:
                self.mid_history[product] = self.mid_history[product][-2500:]

    def pct_vol(self, product, lookback):
        hist = self.mid_history[product]
        if len(hist) < lookback + 1:
            return None
        recent = hist[-lookback:]
        changes = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                changes.append((recent[i] - recent[i - 1]) / recent[i - 1])
        vol = self.std(changes)
        if vol <= 1e-12:
            return None
        return vol

    def momentum_score(self, product: str, lookback: int, vol_lookback: int):
        hist = self.mid_history[product]
        need = max(lookback, vol_lookback) + 1
        if len(hist) < need:
            return None
        now = hist[-1]
        old = hist[-lookback]
        if old <= 0:
            return None
        ret = (now - old) / old
        vol = self.pct_vol(product, vol_lookback)
        if vol is None:
            return None
        return ret / (vol * math.sqrt(lookback))

    def shock_score(self, product):
        hist = self.mid_history[product]
        if len(hist) < self.SHOCK_LOOKBACK + self.VOL_LOOKBACK + 1:
            return 0.0
        now = hist[-1]
        old = hist[-self.SHOCK_LOOKBACK]
        if old <= 0:
            return 0.0
        ret = (now - old) / old
        vol = self.pct_vol(product, self.VOL_LOOKBACK)
        if vol is None:
            return 0.0
        raw = ret / (vol * math.sqrt(self.SHOCK_LOOKBACK))
        return max(-2.0, min(2.0, raw))

    def combined_score(self, product):
        fast = self.momentum_score(
            product, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        slow = self.momentum_score(
            product, self.SLOW_LOOKBACK, self.VOL_LOOKBACK
        )
        if fast is None and slow is None:
            return None
        if slow is None:
            base = fast
        elif fast is None:
            base = slow
        else:
            base = 0.70 * slow + 0.30 * fast
        shock = self.shock_score(product)
        return base + 0.15 * shock

    def rolling_z(self, product):
        hist = self.mid_history[product]
        if len(hist) < self.Z_LOOKBACK:
            return None
        window = hist[-self.Z_LOOKBACK :]
        s = self.std(window)
        if s <= 1e-9:
            return None
        return (window[-1] - self.mean(window)) / s

    def cotton_target(self):
        score = self.combined_score(self.COTTON)
        if score is None:
            target = self.COTTON_BASE_LONG
        elif score >= self.COTTON_STRONG_SCORE:
            target = self.COTTON_STRONG_LONG
        elif score <= self.COTTON_SHORT_SCORE:
            target = self.COTTON_SMALL_SHORT
        elif score <= self.COTTON_FLAT_SCORE:
            target = self.COTTON_FLAT
        elif score <= self.COTTON_REDUCE_SCORE:
            target = self.COTTON_REDUCED_LONG
        else:
            target = self.COTTON_BASE_LONG
        z = self.rolling_z(self.COTTON)
        fast = self.momentum_score(
            self.COTTON, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        if target > 0 and z is not None and fast is not None:
            if z > self.EXTREME_Z and fast < -0.10:
                target = min(target, self.COTTON_REDUCED_LONG)
        return self.clamp_position(target)

    def wool_target(self):
        score = self.combined_score(self.WOOL)
        if score is None:
            target = self.WOOL_BASE_SHORT
        elif score <= self.WOOL_STRONG_SHORT_SCORE:
            target = self.WOOL_STRONG_SHORT
        elif score >= self.WOOL_LONG_SCORE:
            target = self.WOOL_SMALL_LONG
        elif score >= self.WOOL_FLAT_SCORE:
            target = self.WOOL_FLAT
        elif score >= self.WOOL_REDUCE_SHORT_SCORE:
            target = self.WOOL_REDUCED_SHORT
        else:
            target = self.WOOL_BASE_SHORT
        z = self.rolling_z(self.WOOL)
        fast = self.momentum_score(
            self.WOOL, self.FAST_LOOKBACK, self.VOL_LOOKBACK
        )
        if target < 0 and z is not None and fast is not None:
            if z < -self.EXTREME_Z and fast > 0.10:
                target = max(target, self.WOOL_REDUCED_SHORT)
        return self.clamp_position(target)

    def apply_risk_caps(self, targets):
        gross = sum(abs(x) for x in targets.values())
        while gross > self.MAX_GROSS_EXPOSURE:
            nonzero = [p for p in self.PRODUCTS if targets[p] != 0]
            if not nonzero:
                break
            p = max(nonzero, key=lambda x: abs(targets[x]))
            if targets[p] > 0:
                targets[p] -= 1
            else:
                targets[p] += 1
            gross = sum(abs(x) for x in targets.values())
        net = sum(targets.values())
        while abs(net) > self.MAX_NET_EXPOSURE:
            if net > 0:
                longs = [p for p in self.PRODUCTS if targets[p] > 0]
                if not longs:
                    break
                p = max(longs, key=lambda x: targets[x])
                targets[p] -= 1
                net -= 1
            else:
                shorts = [p for p in self.PRODUCTS if targets[p] < 0]
                if not shorts:
                    break
                p = min(shorts, key=lambda x: targets[x])
                targets[p] += 1
                net += 1
        for p in self.PRODUCTS:
            targets[p] = self.clamp_position(targets[p])
        return targets

    def build_targets(self):
        targets = {p: 0 for p in self.PRODUCTS}
        targets[self.COTTON] = self.cotton_target()
        targets[self.WOOL] = self.wool_target()
        targets["SLEEP_POD_SUEDE"] = 0
        targets["SLEEP_POD_POLYESTER"] = 0
        targets["SLEEP_POD_NYLON"] = 0
        return self.apply_risk_caps(targets)

    def get_targets(self):
        if self.tick == 1 or self.tick % self.REBALANCE_EVERY == 0:
            self.cached_targets = self.build_targets()
        return self.cached_targets

    def move_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ):
        orders = []
        target_pos = self.clamp_position(target_pos)
        needed = target_pos - current_pos
        if needed == 0:
            return orders
        if needed > 0:
            remaining = min(
                needed, self.POSITION_LIMIT - current_pos, self.MAX_TRADE_SIZE
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
                -needed, self.POSITION_LIMIT + current_pos, self.MAX_TRADE_SIZE
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

    def run(self, state):
        result = {p: [] for p in self.PRODUCTS}
        conversions = 0
        traderData = ""
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
        return result, conversions, traderData


SNACKPACK_PROFILE = "risk_controlled"


class PairConfig:
    def __init__(
        self,
        product_a: str,
        product_b: str,
        weight: float,
        history_key: str,
        take_edge: int,
        passive_edge: int,
        close_edge: int,
        inventory_skew: float,
        take_size: int,
        passive_size: int,
        window: int,
        min_history: int,
        use_reduce_only: bool,
        reduce_only_pos: int,
        use_regime_filter: bool,
        regime_z_limit: float,
        max_pair_imbalance: float,
        repair_imbalance: bool,
        repair_size: int,
    ):
        self.product_a = product_a
        self.product_b = product_b
        self.weight = weight
        self.history_key = history_key
        self.take_edge = take_edge
        self.passive_edge = passive_edge
        self.close_edge = close_edge
        self.inventory_skew = inventory_skew
        self.take_size = take_size
        self.passive_size = passive_size
        self.window = window
        self.min_history = min_history
        self.use_reduce_only = use_reduce_only
        self.reduce_only_pos = reduce_only_pos
        self.use_regime_filter = use_regime_filter
        self.regime_z_limit = regime_z_limit
        self.max_pair_imbalance = max_pair_imbalance
        self.repair_imbalance = repair_imbalance
        self.repair_size = repair_size


class SnackpackTrader:
    POSITION_LIMIT = 10
    ENDGAME_REDUCE_ONLY_TIME = 98_000
    PRODUCT_TAKE_EDGE_EXTRA = {
        "SNACKPACK_PISTACHIO": 2,
    }
    PRODUCT_PASSIVE_EDGE_EXTRA = {
        "SNACKPACK_PISTACHIO": 1,
    }

    def __init__(self):
        pist_straw = dict(
            take_edge=22,
            passive_edge=14,
            close_edge=2,
            inventory_skew=1.5,
            take_size=2,
            passive_size=1,
            window=500,
            min_history=120,
            use_reduce_only=True,
            reduce_only_pos=8,
            use_regime_filter=True,
            regime_z_limit=4.0,
            max_pair_imbalance=4.0,
            repair_imbalance=True,
            repair_size=2,
        )
        van_choc = dict(
            take_edge=22,
            passive_edge=14,
            close_edge=2,
            inventory_skew=1.8,
            take_size=1,
            passive_size=1,
            window=500,
            min_history=120,
            use_reduce_only=True,
            reduce_only_pos=6,
            use_regime_filter=True,
            regime_z_limit=3.8,
            max_pair_imbalance=4.0,
            repair_imbalance=True,
            repair_size=1,
        )
        self.PAIRS: List[PairConfig] = [
            PairConfig(
                product_a="SNACKPACK_PISTACHIO",
                product_b="SNACKPACK_STRAWBERRY",
                weight=-0.91,
                history_key="h_pist_straw",
                **pist_straw,
            ),
            PairConfig(
                product_a="SNACKPACK_VANILLA",
                product_b="SNACKPACK_CHOCOLATE",
                weight=0.9,
                history_key="h_van_choc",
                **van_choc,
            ),
        ]

    def get_best_bid_ask(self, state, product):
        od = state.order_depths.get(product)
        if od is None or not od.buy_orders or not od.sell_orders:
            return None, None
        return max(od.buy_orders), min(od.sell_orders)

    def get_mid_price(self, state, product):
        bid, ask = self.get_best_bid_ask(state, product)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def add_buy_order(
        self, orders, product, price, quantity, current_position
    ):
        qty = min(quantity, self.POSITION_LIMIT - current_position)
        if qty > 0:
            orders.append(Order(product, int(price), qty))

    def add_sell_order(
        self, orders, product, price, quantity, current_position
    ):
        qty = min(quantity, self.POSITION_LIMIT + current_position)
        if qty > 0:
            orders.append(Order(product, int(price), -qty))

    def pair_imbalance(self, state, cfg):
        pos_a = state.position.get(cfg.product_a, 0)
        pos_b = state.position.get(cfg.product_b, 0)
        return pos_b - cfg.weight * pos_a

    def would_pass_pair_imbalance_filter(self, state, cfg, product, delta_qty):
        pos_a = state.position.get(cfg.product_a, 0)
        pos_b = state.position.get(cfg.product_b, 0)
        current = pos_b - cfg.weight * pos_a
        new_pos_a = pos_a
        new_pos_b = pos_b
        if product == cfg.product_a:
            new_pos_a += delta_qty
        elif product == cfg.product_b:
            new_pos_b += delta_qty
        else:
            return True
        proposed = new_pos_b - cfg.weight * new_pos_a
        if abs(proposed) <= cfg.max_pair_imbalance:
            return True
        if abs(current) > cfg.max_pair_imbalance and abs(proposed) < abs(
            current
        ):
            return True
        return False

    def filter_pair_imbalance_orders(self, state, cfg, orders_a, orders_b):
        filtered_a = [
            o
            for o in orders_a
            if self.would_pass_pair_imbalance_filter(
                state, cfg, cfg.product_a, o.quantity
            )
        ]
        filtered_b = [
            o
            for o in orders_b
            if self.would_pass_pair_imbalance_filter(
                state, cfg, cfg.product_b, o.quantity
            )
        ]
        return filtered_a, filtered_b

    def add_pair_imbalance_repair_orders(self, state, cfg, orders_a, orders_b):
        if not cfg.repair_imbalance:
            return orders_a, orders_b
        imbalance = self.pair_imbalance(state, cfg)
        if abs(imbalance) <= cfg.max_pair_imbalance:
            return orders_a, orders_b
        pos_a = state.position.get(cfg.product_a, 0)
        pos_b = state.position.get(cfg.product_b, 0)
        bid_a, ask_a = self.get_best_bid_ask(state, cfg.product_a)
        bid_b, ask_b = self.get_best_bid_ask(state, cfg.product_b)
        if bid_a is None or ask_a is None or bid_b is None or ask_b is None:
            return orders_a, orders_b
        candidates = []

        def add_candidate(product, delta, price):
            if delta == 0:
                return
            old_abs = abs(imbalance)
            new_pos_a = pos_a + (delta if product == cfg.product_a else 0)
            new_pos_b = pos_b + (delta if product == cfg.product_b else 0)
            new_imbalance = new_pos_b - cfg.weight * new_pos_a
            improvement = old_abs - abs(new_imbalance)
            if improvement <= 0:
                return
            if product == cfg.product_a:
                reduces_inventory = abs(new_pos_a) < abs(pos_a)
            else:
                reduces_inventory = abs(new_pos_b) < abs(pos_b)
            score = improvement + (10.0 if reduces_inventory else 0.0)
            candidates.append((score, product, delta, price))

        size = cfg.repair_size
        if pos_a > 0:
            add_candidate(cfg.product_a, -min(size, pos_a), bid_a)
        elif pos_a < 0:
            add_candidate(cfg.product_a, min(size, -pos_a), ask_a)
        if pos_b > 0:
            add_candidate(cfg.product_b, -min(size, pos_b), bid_b)
        elif pos_b < 0:
            add_candidate(cfg.product_b, min(size, -pos_b), ask_b)
        if not candidates:
            return orders_a, orders_b
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, product, delta, price = candidates[0]
        if product == cfg.product_a:
            if delta > 0:
                self.add_buy_order(orders_a, product, price, delta, pos_a)
            else:
                self.add_sell_order(orders_a, product, price, -delta, pos_a)
        else:
            if delta > 0:
                self.add_buy_order(orders_b, product, price, delta, pos_b)
            else:
                self.add_sell_order(orders_b, product, price, -delta, pos_b)
        return orders_a, orders_b

    def close_position_if_edge_disappeared(
        self,
        orders,
        product,
        position,
        best_bid,
        best_ask,
        adjusted_fair,
        close_edge,
    ):
        if position > 0 and best_bid >= adjusted_fair - close_edge:
            self.add_sell_order(
                orders, product, best_bid, min(2, position), position
            )
        elif position < 0 and best_ask <= adjusted_fair + close_edge:
            self.add_buy_order(
                orders, product, best_ask, min(2, -position), position
            )

    def product_take_edge(self, cfg, product):
        return cfg.take_edge + self.PRODUCT_TAKE_EDGE_EXTRA.get(product, 0)

    def product_passive_edge(self, cfg, product):
        return cfg.passive_edge + self.PRODUCT_PASSIVE_EDGE_EXTRA.get(
            product, 0
        )

    def is_endgame(self, state):
        return getattr(state, "timestamp", 0) >= self.ENDGAME_REDUCE_ONLY_TIME

    def trade_against_fair(self, state, product, fair_value, basket_z, cfg):
        orders = []
        best_bid, best_ask = self.get_best_bid_ask(state, product)
        if best_bid is None or best_ask is None:
            return orders
        position = state.position.get(product, 0)
        adjusted_fair = fair_value - cfg.inventory_skew * position
        buy_edge = adjusted_fair - best_ask
        sell_edge = best_bid - adjusted_fair
        take_edge = self.product_take_edge(cfg, product)
        passive_edge = self.product_passive_edge(cfg, product)
        reduce_only = (
            self.is_endgame(state)
            or (cfg.use_reduce_only and abs(position) >= cfg.reduce_only_pos)
            or (cfg.use_regime_filter and abs(basket_z) > cfg.regime_z_limit)
        )
        if reduce_only:
            self.close_position_if_edge_disappeared(
                orders,
                product,
                position,
                best_bid,
                best_ask,
                adjusted_fair,
                cfg.close_edge,
            )
            return orders
        if buy_edge > take_edge:
            self.add_buy_order(
                orders, product, best_ask, cfg.take_size, position
            )
            return orders
        if sell_edge > take_edge:
            self.add_sell_order(
                orders, product, best_bid, cfg.take_size, position
            )
            return orders
        passive_bid = best_bid + 1
        passive_ask = best_ask - 1
        if passive_bid >= best_ask:
            passive_bid = best_bid
        if passive_ask <= best_bid:
            passive_ask = best_ask
        passive_buy_edge = adjusted_fair - passive_bid
        passive_sell_edge = passive_ask - adjusted_fair
        if passive_buy_edge > passive_edge:
            self.add_buy_order(
                orders, product, passive_bid, cfg.passive_size, position
            )
        elif passive_sell_edge > passive_edge:
            self.add_sell_order(
                orders, product, passive_ask, cfg.passive_size, position
            )
        else:
            self.close_position_if_edge_disappeared(
                orders,
                product,
                position,
                best_bid,
                best_ask,
                adjusted_fair,
                cfg.close_edge,
            )
        return orders

    def trade_pair(self, state, cfg, data):
        orders_a = []
        orders_b = []
        mid_a = self.get_mid_price(state, cfg.product_a)
        mid_b = self.get_mid_price(state, cfg.product_b)
        if mid_a is None or mid_b is None:
            return orders_a, orders_b
        basket = mid_a + cfg.weight * mid_b
        history = data.setdefault(cfg.history_key, [])
        history.append(basket)
        if len(history) > cfg.window:
            history[:] = history[-cfg.window :]
        if len(history) < cfg.min_history:
            return orders_a, orders_b
        basket_mean = _statistics.mean(history)
        basket_std = _statistics.pstdev(history)
        if basket_std == 0:
            return orders_a, orders_b
        basket_z = (basket - basket_mean) / basket_std
        a_fair = basket_mean - cfg.weight * mid_b
        b_fair = (basket_mean - mid_a) / cfg.weight
        orders_a = self.trade_against_fair(
            state, cfg.product_a, a_fair, basket_z, cfg
        )
        orders_b = self.trade_against_fair(
            state, cfg.product_b, b_fair, basket_z, cfg
        )
        orders_a, orders_b = self.filter_pair_imbalance_orders(
            state, cfg, orders_a, orders_b
        )
        orders_a, orders_b = self.add_pair_imbalance_repair_orders(
            state, cfg, orders_a, orders_b
        )
        data[f"{cfg.history_key}_z"] = basket_z
        data[f"{cfg.history_key}_mean"] = basket_mean
        data[f"{cfg.history_key}_std"] = basket_std
        data[f"{cfg.history_key}_imbalance"] = self.pair_imbalance(state, cfg)
        data[f"{cfg.history_key}_weight"] = cfg.weight
        data[f"{cfg.history_key}_endgame_reduce_only"] = self.is_endgame(state)
        return orders_a, orders_b

    def run(self, state):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        result = {}
        conversions = 0
        for cfg in self.PAIRS:
            orders_a, orders_b = self.trade_pair(state, cfg, data)
            result.setdefault(cfg.product_a, []).extend(orders_a)
            result.setdefault(cfg.product_b, []).extend(orders_b)
        data["profile"] = SNACKPACK_PROFILE
        return result, conversions, json.dumps(data)
