from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


class VelvetfruitStrategy:
    PRODUCT = "VELVETFRUIT_EXTRACT"
    POSITION_LIMIT = 200

    # VE day-2 data shows a higher center than 5250.
    # This version treats VE as a bounded mean-reversion product, not a trend-following product.
    BASE_FAIR = 5255

    HISTORY_LENGTH = 160
    SHORT_WINDOW = 20
    LONG_WINDOW = 80
    REGIME_WINDOW = 60

    # Absolute bands from VE distribution.
    # These are intentionally more important than momentum.
    BUY_BAND_1 = 5235
    BUY_BAND_2 = 5225
    BUY_BAND_3 = 5215

    SELL_BAND_1 = 5272
    SELL_BAND_2 = 5282
    SELL_BAND_3 = 5292

    # Edge-based backup entries.
    SMALL_EDGE = 3
    BIG_EDGE = 10
    EXTREME_EDGE = 20

    # Sizes
    MM_SIZE = 6
    NORMAL_SIZE = 22
    BIG_SIZE = 38
    EXTREME_SIZE = 58
    UNWIND_SIZE = 22

    # Risk controls
    SOFT_LIMIT = 175
    NO_ADD_LEVEL = 150
    DANGER_LEVEL = 190

    INVENTORY_SKEW = 0.045

    def __init__(self):
        self.position = 0
        self.mid_history = []

    def load_state(self, data):
        self.mid_history = data.get("mid_history", [])

    def save_state(self):
        return {
            "mid_history": self.mid_history[-self.HISTORY_LENGTH:]
        }

    def get_mid_price(self, od):
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def update_history(self, mid):
        self.mid_history.append(mid)
        if len(self.mid_history) > self.HISTORY_LENGTH:
            self.mid_history.pop(0)

    def avg_last(self, n):
        if not self.mid_history:
            return self.BASE_FAIR
        vals = self.mid_history[-min(n, len(self.mid_history)):]
        return sum(vals) / len(vals)

    def get_center(self):
        short_mean = self.avg_last(self.SHORT_WINDOW)
        long_mean = self.avg_last(self.LONG_WINDOW)

        # Stronger anchor than v1/v2.
        # v2 failed because it followed rolling trend too much.
        center = (
            0.55 * self.BASE_FAIR
            + 0.20 * short_mean
            + 0.25 * long_mean
        )

        center -= self.INVENTORY_SKEW * self.position
        return center

    def get_trend(self):
        if len(self.mid_history) < self.SHORT_WINDOW:
            return 0
        return self.mid_history[-1] - self.mid_history[-self.SHORT_WINDOW]

    def get_regime(self):
        if len(self.mid_history) < 5:
            return 0, 0

        recent = self.mid_history[-min(self.REGIME_WINDOW, len(self.mid_history)):]
        high = max(recent)
        low = min(recent)
        mid = self.mid_history[-1]

        return high - mid, mid - low

    def buy_capacity(self):
        hard = self.POSITION_LIMIT - self.position
        soft = self.SOFT_LIMIT - self.position
        return max(0, min(hard, soft))

    def sell_capacity(self):
        hard = self.POSITION_LIMIT + self.position
        soft = self.SOFT_LIMIT + self.position
        return max(0, min(hard, soft))

    def hard_buy_capacity(self):
        return max(0, self.POSITION_LIMIT - self.position)

    def hard_sell_capacity(self):
        return max(0, self.POSITION_LIMIT + self.position)

    def buy(self, orders, price, qty, use_soft=True):
        cap = self.buy_capacity() if use_soft else self.hard_buy_capacity()
        qty = min(int(qty), cap)

        if qty > 0:
            orders.append(Order(self.PRODUCT, int(price), qty))
            self.position += qty

    def sell(self, orders, price, qty, use_soft=True):
        cap = self.sell_capacity() if use_soft else self.hard_sell_capacity()
        qty = min(int(qty), cap)

        if qty > 0:
            orders.append(Order(self.PRODUCT, int(price), -qty))
            self.position -= qty

    def trade(self, state, orders):
        od = state.order_depths[self.PRODUCT]

        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_history(mid)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        center = self.get_center()
        trend = self.get_trend()
        drop_from_high, rise_from_low = self.get_regime()

        buy_edge = center - best_ask
        sell_edge = best_bid - center

        # ------------------------------------------------
        # 0. Emergency inventory control
        # ------------------------------------------------
        if self.position > self.DANGER_LEVEL:
            self.sell(orders, best_bid, min(self.UNWIND_SIZE, self.position), use_soft=False)
            return

        if self.position < -self.DANGER_LEVEL:
            self.buy(orders, best_ask, min(self.UNWIND_SIZE, -self.position), use_soft=False)
            return

        # ------------------------------------------------
        # 1. Take profit / inventory release
        # ------------------------------------------------
        # For longs bought in lower bands, exit around center / upper center.
        if self.position > 25:
            if best_bid >= max(center + 8, 5264):
                self.sell(orders, best_bid, min(34, self.position), use_soft=False)
                return
            elif best_bid >= max(center + 4, 5258):
                self.sell(orders, best_bid, min(20, self.position), use_soft=False)
                return
            elif best_bid >= 5254 and self.position > 70:
                self.sell(orders, best_bid, min(16, self.position), use_soft=False)
                return

        # For shorts opened in upper bands, cover around center / lower center.
        if self.position < -25:
            if best_ask <= min(center - 8, 5244):
                self.buy(orders, best_ask, min(34, -self.position), use_soft=False)
                return
            elif best_ask <= min(center - 4, 5250):
                self.buy(orders, best_ask, min(20, -self.position), use_soft=False)
                return
            elif best_ask <= 5256 and self.position < -70:
                self.buy(orders, best_ask, min(16, -self.position), use_soft=False)
                return

        # ------------------------------------------------
        # 2. Absolute-band entries
        # ------------------------------------------------
        # Main idea: v2 trend-following lost money. VE is better treated as bounded.
        # Buy low bands, sell high bands, size increases with extremeness.
        can_add_long = self.position < self.NO_ADD_LEVEL
        can_add_short = self.position > -self.NO_ADD_LEVEL

        if can_add_long:
            if best_ask <= self.BUY_BAND_3:
                self.buy(orders, best_ask, self.EXTREME_SIZE)
                return
            elif best_ask <= self.BUY_BAND_2:
                # Avoid adding too aggressively if still collapsing without any bounce.
                if trend > -18 or rise_from_low >= 4:
                    self.buy(orders, best_ask, self.BIG_SIZE)
                    return
            elif best_ask <= self.BUY_BAND_1:
                if trend > -12 or rise_from_low >= 3:
                    self.buy(orders, best_ask, self.NORMAL_SIZE)
                    return

        if can_add_short:
            if best_bid >= self.SELL_BAND_3:
                self.sell(orders, best_bid, self.EXTREME_SIZE)
                return
            elif best_bid >= self.SELL_BAND_2:
                if trend < 18 or drop_from_high >= 4:
                    self.sell(orders, best_bid, self.BIG_SIZE)
                    return
            elif best_bid >= self.SELL_BAND_1:
                if trend < 12 or drop_from_high >= 3:
                    self.sell(orders, best_bid, self.NORMAL_SIZE)
                    return

        # ------------------------------------------------
        # 3. Edge-based backup mean reversion
        # ------------------------------------------------
        # This catches opportunities not exactly at absolute bands.
        if can_add_long:
            if buy_edge >= self.EXTREME_EDGE:
                if trend > -16 or rise_from_low >= 4:
                    for ask, vol in sorted(od.sell_orders.items()):
                        if ask <= center - self.EXTREME_EDGE:
                            self.buy(orders, ask, min(-vol, self.BIG_SIZE))
                            break

            elif buy_edge >= self.BIG_EDGE and mid < center - 8:
                if trend > -10 or rise_from_low >= 3:
                    for ask, vol in sorted(od.sell_orders.items()):
                        if ask <= center - self.BIG_EDGE:
                            self.buy(orders, ask, min(-vol, self.NORMAL_SIZE))
                            break

        if can_add_short:
            if sell_edge >= self.EXTREME_EDGE:
                if trend < 16 or drop_from_high >= 4:
                    for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                        if bid >= center + self.EXTREME_EDGE:
                            self.sell(orders, bid, min(vol, self.BIG_SIZE))
                            break

            elif sell_edge >= self.BIG_EDGE and mid > center + 8:
                if trend < 10 or drop_from_high >= 3:
                    for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                        if bid >= center + self.BIG_EDGE:
                            self.sell(orders, bid, min(vol, self.NORMAL_SIZE))
                            break

        # ------------------------------------------------
        # 4. Passive market making near center only
        # ------------------------------------------------
        # Do not MM aggressively at extremes; use band logic there.
        if abs(mid - center) <= 14:
            quote_bid = int(center - self.SMALL_EDGE)
            quote_ask = int(center + self.SMALL_EDGE)

            quote_bid = min(quote_bid, best_bid + 1)
            quote_ask = max(quote_ask, best_ask - 1)

            if quote_bid >= quote_ask:
                quote_bid = best_bid
                quote_ask = best_ask

            buy_size = self.MM_SIZE
            sell_size = self.MM_SIZE

            if self.position > 35:
                buy_size = 0
                sell_size = 10
            elif self.position < -35:
                buy_size = 10
                sell_size = 0

            if self.position > self.NO_ADD_LEVEL:
                buy_size = 0
                sell_size = max(sell_size, 14)

            if self.position < -self.NO_ADD_LEVEL:
                sell_size = 0
                buy_size = max(buy_size, 14)

            # If short-term trend is one-way, stop quoting against it.
            if trend > 12:
                sell_size = 0
                if self.position < 0:
                    buy_size = max(buy_size, 10)

            elif trend < -12:
                buy_size = 0
                if self.position > 0:
                    sell_size = max(sell_size, 10)

            if buy_size > 0:
                self.buy(orders, quote_bid, buy_size)

            if sell_size > 0:
                self.sell(orders, quote_ask, sell_size)


class HydrogelPackStrategy:
    PRODUCT = "HYDROGEL_PACK"
    POSITION_LIMIT = 200

    BASE_FAIR = 10000

    HISTORY_LENGTH = 150
    SHORT_WINDOW = 20
    LONG_WINDOW = 80
    REGIME_WINDOW = 60

    SMALL_EDGE = 5
    BIG_EDGE = 24
    EXTREME_EDGE = 42

    MM_SIZE = 5
    NORMAL_SIZE = 12
    BIG_SIZE = 25
    UNWIND_SIZE = 18

    SOFT_LIMIT = 95
    NO_ADD_LEVEL = 60
    DANGER_LEVEL = 115

    LONG_TP_1 = 9985
    LONG_TP_2 = 9995
    SHORT_TP_1 = 9970
    SHORT_TP_2 = 9955

    TP_COOLDOWN = 12500

    DEEP_REBUY_LEVEL = 9945
    DEEP_RESHORT_LEVEL = 10015
    REBOUND_CONFIRM = 6

    INVENTORY_SKEW = 0.10

    # Direction 2: flat-state light market making.
    # Keep micro tools very mild; main addition is low-risk activity in quiet/platform periods.
    MICRO_FAIR_WEIGHT = 0.25
    MICRO_CONFIRM = 1.0
    MICRO_CAP = 4.0

    LIGHT_MM_SIZE = 2
    LIGHT_MM_EDGE = 6
    FLAT_POSITION = 20
    LIGHT_TREND_LIMIT = 11
    LIGHT_SPREAD_MIN = 8

    def __init__(self):
        self.position = 0
        self.mid_history = []
        self.last_long_tp_time = -10**9
        self.last_short_tp_time = -10**9

    def load_state(self, data):
        self.mid_history = data.get("mid_history", [])
        self.last_long_tp_time = data.get("last_long_tp_time", -10**9)
        self.last_short_tp_time = data.get("last_short_tp_time", -10**9)

    def save_state(self):
        return {
            "mid_history": self.mid_history[-self.HISTORY_LENGTH:],
            "last_long_tp_time": self.last_long_tp_time,
            "last_short_tp_time": self.last_short_tp_time,
        }

    def get_mid_price(self, od):
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2

    def get_micro_signal(self, od):
        if not od.buy_orders or not od.sell_orders:
            return 0

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = -od.sell_orders[best_ask]

        total = bid_vol + ask_vol
        if total <= 0:
            return 0

        mid = (best_bid + best_ask) / 2
        micro = (best_bid * ask_vol + best_ask * bid_vol) / total
        return micro - mid

    def update_history(self, mid):
        self.mid_history.append(mid)
        if len(self.mid_history) > self.HISTORY_LENGTH:
            self.mid_history.pop(0)

    def avg_last(self, n):
        if not self.mid_history:
            return self.BASE_FAIR
        vals = self.mid_history[-min(n, len(self.mid_history)):]
        return sum(vals) / len(vals)

    def get_fair(self):
        short_mean = self.avg_last(self.SHORT_WINDOW)
        long_mean = self.avg_last(self.LONG_WINDOW)

        fair = (
            0.50 * self.BASE_FAIR
            + 0.25 * short_mean
            + 0.25 * long_mean
        )

        fair -= self.INVENTORY_SKEW * self.position
        return fair

    def get_trend(self):
        if len(self.mid_history) < self.SHORT_WINDOW:
            return 0
        return self.mid_history[-1] - self.mid_history[-self.SHORT_WINDOW]

    def get_regime(self):
        if len(self.mid_history) < 5:
            return 0, 0

        recent = self.mid_history[-min(self.REGIME_WINDOW, len(self.mid_history)):]
        recent_high = max(recent)
        recent_low = min(recent)
        mid = self.mid_history[-1]

        return recent_high - mid, mid - recent_low

    def buy_capacity(self):
        hard = self.POSITION_LIMIT - self.position
        soft = self.SOFT_LIMIT - self.position
        return max(0, min(hard, soft))

    def sell_capacity(self):
        hard = self.POSITION_LIMIT + self.position
        soft = self.SOFT_LIMIT + self.position
        return max(0, min(hard, soft))

    def hard_buy_capacity(self):
        return max(0, self.POSITION_LIMIT - self.position)

    def hard_sell_capacity(self):
        return max(0, self.POSITION_LIMIT + self.position)

    def buy(self, orders, price, qty, use_soft=True):
        cap = self.buy_capacity() if use_soft else self.hard_buy_capacity()
        qty = min(int(qty), cap)
        if qty > 0:
            orders.append(Order(self.PRODUCT, int(price), qty))
            self.position += qty

    def sell(self, orders, price, qty, use_soft=True):
        cap = self.sell_capacity() if use_soft else self.hard_sell_capacity()
        qty = min(int(qty), cap)
        if qty > 0:
            orders.append(Order(self.PRODUCT, int(price), -qty))
            self.position -= qty

    def trade(self, state, orders):
        od = state.order_depths[self.PRODUCT]
        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_history(mid)
        timestamp = state.timestamp

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        fair = self.get_fair()
        micro_signal = self.get_micro_signal(od)

        # Only use bounded micro adjustment, so it improves timing but does not dominate the anchor.
        if abs(micro_signal) <= self.MICRO_CAP:
            fair += self.MICRO_FAIR_WEIGHT * micro_signal

        trend = self.get_trend()
        drop_from_high, rise_from_low = self.get_regime()

        buy_edge = fair - best_ask
        sell_edge = best_bid - fair

        in_long_tp_cooldown = timestamp - self.last_long_tp_time < self.TP_COOLDOWN
        in_short_tp_cooldown = timestamp - self.last_short_tp_time < self.TP_COOLDOWN

        if self.position > self.DANGER_LEVEL:
            if best_bid >= fair - 15:
                self.sell(orders, best_bid, min(self.UNWIND_SIZE, self.position), use_soft=False)
                return

        if self.position < -self.DANGER_LEVEL:
            if best_ask <= fair + 15:
                self.buy(orders, best_ask, min(self.UNWIND_SIZE, -self.position), use_soft=False)
                return

        if self.position > 25:
            if best_bid >= self.LONG_TP_2:
                self.sell(orders, best_bid, min(35, self.position), use_soft=False)
                self.last_long_tp_time = timestamp
                return
            elif best_bid >= self.LONG_TP_1:
                self.sell(orders, best_bid, min(22, self.position), use_soft=False)
                self.last_long_tp_time = timestamp
                return
            elif best_bid >= fair + 12:
                self.sell(orders, best_bid, min(20, self.position), use_soft=False)
                self.last_long_tp_time = timestamp
                return

        if self.position < -25:
            if best_ask <= self.SHORT_TP_2:
                self.buy(orders, best_ask, min(35, -self.position), use_soft=False)
                self.last_short_tp_time = timestamp
                return
            elif best_ask <= self.SHORT_TP_1:
                self.buy(orders, best_ask, min(22, -self.position), use_soft=False)
                self.last_short_tp_time = timestamp
                return
            elif best_ask <= fair - 12:
                self.buy(orders, best_ask, min(20, -self.position), use_soft=False)
                self.last_short_tp_time = timestamp
                return

        if in_long_tp_cooldown and self.position > 0:
            if best_bid >= 9978:
                self.sell(orders, best_bid, min(12, self.position), use_soft=False)
                return

        if in_short_tp_cooldown and self.position < 0:
            if best_ask <= 9985:
                self.buy(orders, best_ask, min(12, -self.position), use_soft=False)
                return

        rebound_confirmed = rise_from_low >= self.REBOUND_CONFIRM or micro_signal >= self.MICRO_CONFIRM
        pullback_confirmed = drop_from_high >= self.REBOUND_CONFIRM or micro_signal <= -self.MICRO_CONFIRM

        allow_deep_rebuy_after_cooldown = (
            (best_ask <= self.DEEP_REBUY_LEVEL or buy_edge >= self.EXTREME_EDGE + 8)
            and rebound_confirmed
            and trend > -10
        )

        allow_deep_reshort_after_cooldown = (
            (best_bid >= self.DEEP_RESHORT_LEVEL or sell_edge >= self.EXTREME_EDGE + 8)
            and pullback_confirmed
            and trend < 10
        )

        block_long_after_tp = in_long_tp_cooldown and not allow_deep_rebuy_after_cooldown
        block_short_after_tp = in_short_tp_cooldown and not allow_deep_reshort_after_cooldown

        can_add_long = self.position < self.NO_ADD_LEVEL and not block_long_after_tp
        can_add_short = self.position > -self.NO_ADD_LEVEL and not block_short_after_tp

        # Controlled cooldown re-entry:
        # After taking profit, the old version can become too inactive.
        # This permits only small re-entry when price is meaningfully dislocated
        # AND micro/regime confirms a rebound/pullback.
        if in_long_tp_cooldown and self.position < 25:
            if buy_edge >= self.BIG_EDGE + 12 and rebound_confirmed and trend > -8:
                for ask, vol in sorted(od.sell_orders.items()):
                    if ask <= fair - (self.BIG_EDGE + 8):
                        self.buy(orders, ask, min(-vol, 8))
                        return

        if in_short_tp_cooldown and self.position > -25:
            if sell_edge >= self.BIG_EDGE + 12 and pullback_confirmed and trend < 8:
                for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                    if bid >= fair + (self.BIG_EDGE + 8):
                        self.sell(orders, bid, min(vol, 8))
                        return

        if can_add_long:
            if buy_edge >= self.EXTREME_EDGE:
                if trend > -25 or micro_signal >= self.MICRO_CONFIRM:
                    for ask, vol in sorted(od.sell_orders.items()):
                        if ask <= fair - self.EXTREME_EDGE:
                            self.buy(orders, ask, min(-vol, self.BIG_SIZE))
                            break
            elif buy_edge >= self.BIG_EDGE:
                if trend > -12 and micro_signal > -1.5:
                    for ask, vol in sorted(od.sell_orders.items()):
                        if ask <= fair - self.BIG_EDGE:
                            self.buy(orders, ask, min(-vol, self.NORMAL_SIZE))
                            break

        if can_add_short:
            if sell_edge >= self.EXTREME_EDGE:
                if trend < 25 or micro_signal <= -self.MICRO_CONFIRM:
                    for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                        if bid >= fair + self.EXTREME_EDGE:
                            self.sell(orders, bid, min(vol, self.BIG_SIZE))
                            break
            elif sell_edge >= self.BIG_EDGE:
                if trend < 12 and micro_signal < 1.5:
                    for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                        if bid >= fair + self.BIG_EDGE:
                            self.sell(orders, bid, min(vol, self.NORMAL_SIZE))
                            break

        quote_bid = int(fair - self.SMALL_EDGE)
        quote_ask = int(fair + self.SMALL_EDGE)

        quote_bid = min(quote_bid, best_bid + 1)
        quote_ask = max(quote_ask, best_ask - 1)

        if quote_bid >= quote_ask:
            quote_bid = best_bid
            quote_ask = best_ask

        buy_size = self.MM_SIZE
        sell_size = self.MM_SIZE

        if self.position > 35:
            buy_size = 0
            sell_size = 10
        elif self.position < -35:
            buy_size = 10
            sell_size = 0

        if self.position > self.NO_ADD_LEVEL:
            buy_size = 0
            sell_size = max(sell_size, 12)

        if self.position < -self.NO_ADD_LEVEL:
            sell_size = 0
            buy_size = max(buy_size, 12)

        if block_long_after_tp:
            buy_size = 0

        if block_short_after_tp:
            sell_size = 0

        if trend < -12:
            buy_size = 0
            if self.position > 0 and best_bid >= 9970:
                sell_size = max(sell_size, 8)
            else:
                sell_size = 0

        elif trend > 12:
            sell_size = 0
            if self.position < 0 and best_ask <= 9990:
                buy_size = max(buy_size, 8)
            else:
                buy_size = max(buy_size, 5)

        if abs(trend) > 35:
            if trend < 0 and self.position > 0 and best_bid >= 9968:
                self.sell(orders, best_bid, min(8, self.position), use_soft=False)
            elif trend > 0 and self.position < 0 and best_ask <= 9992:
                self.buy(orders, best_ask, min(8, -self.position), use_soft=False)
            return

        # Micro exhaustion rescue:
        # If we hold inventory and price/micro moves favorably, release a small piece
        # before returning to normal light MM. This tries to reduce late drawdown
        # without changing the core entry rules.
        if self.position > 45 and micro_signal <= -self.MICRO_CONFIRM and best_bid >= fair - 8:
            self.sell(orders, best_bid, min(10, self.position), use_soft=False)
            return

        if self.position < -45 and micro_signal >= self.MICRO_CONFIRM and best_ask <= fair + 8:
            self.buy(orders, best_ask, min(10, -self.position), use_soft=False)
            return

        # Flat-state light MM: only active when inventory is small and the market is quiet.
        # This tries to mine platform periods without relaxing the main aggressive-entry rules.
        spread = best_ask - best_bid
        if (
            abs(self.position) <= self.FLAT_POSITION
            and abs(trend) <= self.LIGHT_TREND_LIMIT
            and spread >= self.LIGHT_SPREAD_MIN
            and not block_long_after_tp
            and not block_short_after_tp
        ):
            light_bid = min(int(fair - self.LIGHT_MM_EDGE), best_bid + 1)
            light_ask = max(int(fair + self.LIGHT_MM_EDGE), best_ask - 1)

            if light_bid < light_ask:
                self.buy(orders, light_bid, self.LIGHT_MM_SIZE)
                self.sell(orders, light_ask, self.LIGHT_MM_SIZE)
                return

        if buy_size > 0:
            self.buy(orders, quote_bid, buy_size)

        if sell_size > 0:
            self.sell(orders, quote_ask, sell_size)

    def run(self, state: TradingState):
        result = {self.PRODUCT: []}
        self.position = state.position.get(self.PRODUCT, 0)
        self.load_state(state)

        if self.PRODUCT in state.order_depths:
            self.trade_pack(state, result[self.PRODUCT])

        return result, 0, self.save_state()


class VEVoucherStrategy:

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    # Main curve products plus separate conservative logic for 4500 / 6000 / 6500.
    TRADE_VOUCHERS = [
        # "VEV_4500",  # disabled: latest no-5400 log still showed small negative drag
        # "VEV_5000",  # disabled: latest no-5400 log still showed small negative drag
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        # "VEV_5400",  # disabled: current best log showed ~-2801 PnL drag
        "VEV_6000",
        "VEV_6500",
    ]

    # -------- VEV fitted cubic bid/ask/mid IV curve parameters --------
    # Fit used m = log(K / S) / sqrt(T)
    # iv = a*m^3 + b*m^2 + c*m + d
    A_BID = -0.08172118511232661
    B_BID = -0.14249484809927185
    C_BID = 0.05838696566433157
    D_BID = 0.24682907179977334

    A_ASK = 0.08456588372853172
    B_ASK = 0.36803352136770356
    C_ASK = -0.05405966827772763
    D_ASK = 0.23921117868175876

    A_MID = -0.008686297498649971
    B_MID = -0.051288378835760776
    C_MID = 0.0016334219912074119
    D_MID = 0.25489135617663367

    DAYS_PER_YEAR = 365.0

    # For day 2 backtest use 5.0.
    # If testing day 0, try 7.0.
    # If testing day 1, try 6.0.
    STARTING_DAYS_TO_EXPIRY = 5.0

    # Conservative sizes first.
    MAX_TAKE_SIZE = 5
    MAX_MAKE_SIZE = 6

    # Require at least this much price edge before crossing spread.
    PRICE_EDGE = 0.8

    # Skip markets that are too wide.
    MAX_MARKET_SPREAD = 8

    # Optional passive market making.
    ENABLE_PASSIVE_QUOTES = True

    LOTTERY_SELL_ONLY = {"VEV_6000", "VEV_6500"}
    LOTTERY_SELL_SIZE = 4

    DEEP_ITM = {"VEV_4500"}
    DEEP_ITM_SIZE = 3
    DEEP_ITM_PREMIUM = {
        "VEV_4500": 8,
    }
    DEEP_ITM_EDGE = {
        "VEV_4500": 6,
    }

    POSITION_LIMITS = {
        "VEV_4500": 80,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_6000": 80,
        "VEV_6500": 80,
        "VELVETFRUIT_EXTRACT": 200,
    }

    DO_DELTA_HEDGE = False

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        S = self.get_mid_price(state, self.UNDERLYING)
        if S is None:
            return result, 0, self.safe_trader_data(state)

        T = self.get_time_to_expiry(state)
        if T <= 0:
            return result, 0, self.safe_trader_data(state)

        for voucher in self.TRADE_VOUCHERS:
            if voucher not in state.order_depths:
                continue

            # Do not let one product crash the whole trader.
            try:
                orders = self.trade_vev_voucher(state, voucher, S, T)
                if orders:
                    result[voucher] = orders
            except Exception:
                continue

        return result, 0, self.safe_trader_data(state)

    # ============================================================
    # Core VEV voucher strategy
    # ============================================================

    def trade_vev_voucher(
        self,
        state: TradingState,
        product: str,
        S: float,
        T: float,
    ) -> List[Order]:
        orders: List[Order] = []
        order_depth = state.order_depths[product]

        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)

        if best_bid is None or best_ask is None:
            return orders

        # Avoid garbage 0-bid options.
        if best_bid <= 0:
            return orders

        market_spread = best_ask - best_bid
        if market_spread <= 0:
            return orders

        if market_spread > self.MAX_MARKET_SPREAD:
            return orders

        K = self.get_strike(product)

        # ============================================================
        # Special logic for far OTM lottery vouchers: VEV_6000 / VEV_6500
        # Only sell them when someone bids at least 1.
        # Never buy these products.
        # ============================================================
        if product in self.LOTTERY_SELL_ONLY:
            position = self.get_position(state, product)
            limit = self.get_position_limit(product)

            if best_bid >= 1:
                bid_volume = order_depth.buy_orders[best_bid]
                max_can_sell = limit + position
                qty = min(bid_volume, max_can_sell, self.LOTTERY_SELL_SIZE)

                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            return orders

        # ============================================================
        # Special logic for deep ITM voucher: VEV_4500
        # Trade around intrinsic value + small premium.
        # This avoids treating deep ITM calls like normal IV scalping products.
        # ============================================================
        if product in self.DEEP_ITM:
            position = self.get_position(state, product)
            limit = self.get_position_limit(product)

            intrinsic = max(0.0, S - K)
            premium = self.DEEP_ITM_PREMIUM.get(product, 8)
            edge = self.DEEP_ITM_EDGE.get(product, 6)

            fair = intrinsic + premium
            max_size = self.DEEP_ITM_SIZE

            # Buy only if it is clearly cheaper than intrinsic + premium.
            if best_ask < fair - edge:
                ask_volume = -order_depth.sell_orders[best_ask]
                max_can_buy = limit - position
                qty = min(ask_volume, max_can_buy, max_size)

                if qty > 0:
                    orders.append(Order(product, best_ask, qty))

                return orders

            # Sell only if it is clearly richer than intrinsic + premium.
            if best_bid > fair + edge:
                bid_volume = order_depth.buy_orders[best_bid]
                max_can_sell = limit + position
                qty = min(bid_volume, max_can_sell, max_size)

                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

                return orders

            return orders

        mid_price = (best_bid + best_ask) / 2.0
        if mid_price < 5:
            return orders

        bid_iv = self.get_curve_iv(S, K, T, side="bid")
        ask_iv = self.get_curve_iv(S, K, T, side="ask")
        mid_iv = self.get_curve_iv(S, K, T, side="mid")

        predicted_bid = self.bs_call_price(S, K, T, bid_iv)
        predicted_ask = self.bs_call_price(S, K, T, ask_iv)
        predicted_mid = self.bs_call_price(S, K, T, mid_iv)

        # Integer quote prices.
        our_bid = math.floor(predicted_bid)
        our_ask = math.ceil(predicted_ask)

        # Prevent nonsensical crossed model market.
        if our_bid >= our_ask:
            our_bid = math.floor(predicted_mid) - 1
            our_ask = math.ceil(predicted_mid) + 1

        intrinsic = max(0.0, S - K)
        our_bid = max(0, our_bid)
        our_ask = max(math.ceil(intrinsic), our_ask)

        position = self.get_position(state, product)
        limit = self.get_position_limit(product)

        # -------------------------
        # 1. TAKE cheap ask
        # If our BID curve price is higher than market ask, even our conservative buy model
        # thinks the ask is cheap.
        # -------------------------
        if predicted_bid > best_ask + self.PRICE_EDGE:
            ask_volume = -order_depth.sell_orders[best_ask]
            max_can_buy = limit - position
            qty = min(ask_volume, max_can_buy, self.MAX_TAKE_SIZE)

            if qty > 0:
                orders.append(Order(product, best_ask, qty))

            # Avoid both buy and sell in same timestamp.
            return orders

        # -------------------------
        # 2. TAKE expensive bid
        # If our ASK curve price is lower than market bid, even our conservative sell model
        # thinks the bid is expensive.
        # -------------------------
        if predicted_ask < best_bid - self.PRICE_EDGE:
            bid_volume = order_depth.buy_orders[best_bid]
            max_can_sell = limit + position
            qty = min(bid_volume, max_can_sell, self.MAX_TAKE_SIZE)

            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

            return orders

        # -------------------------
        # 3. Passive market making around IV curve prices.
        # -------------------------
        if not self.ENABLE_PASSIVE_QUOTES:
            return orders

        quote_bid = min(our_bid, best_bid + 1, best_ask - 1)
        quote_ask = max(our_ask, best_ask - 1, best_bid + 1)

        if quote_bid <= 0 or quote_ask <= 0:
            return orders

        if quote_bid >= quote_ask:
            return orders

        max_buy = limit - position
        max_sell = limit + position

        buy_size = min(max_buy, self.MAX_MAKE_SIZE)
        sell_size = min(max_sell, self.MAX_MAKE_SIZE)

        # Inventory skew.
        if position > 100:
            buy_size = 0
            sell_size = min(max_sell, self.MAX_MAKE_SIZE + 4)
        elif position < -100:
            buy_size = min(max_buy, self.MAX_MAKE_SIZE + 4)
            sell_size = 0

        if buy_size > 0:
            orders.append(Order(product, quote_bid, buy_size))

        if sell_size > 0:
            orders.append(Order(product, quote_ask, -sell_size))

        return orders

    # ============================================================
    # IV curve / Black-Scholes
    # ============================================================

    def get_curve_iv(self, S: float, K: float, T: float, side: str) -> float:
        if side == "bid":
            a, b, c, d = self.A_BID, self.B_BID, self.C_BID, self.D_BID
        elif side == "ask":
            a, b, c, d = self.A_ASK, self.B_ASK, self.C_ASK, self.D_ASK
        else:
            a, b, c, d = self.A_MID, self.B_MID, self.C_MID, self.D_MID

        if S <= 0 or K <= 0 or T <= 0:
            return d

        # IMPORTANT: fitted with log(K / S), not log(S / K).
        m = math.log(K / S) / math.sqrt(T)
        iv = a * m * m * m + b * m * m + c * m + d

        return min(max(iv, 0.01), 3.0)

    def bs_call_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float = 0.0,
    ) -> float:
        if T <= 0:
            return max(0.0, S - K)

        if sigma <= 0:
            return max(0.0, S - K)

        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        return S * self.normal_cdf(d1) - K * math.exp(-r * T) * self.normal_cdf(d2)

    def normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # ============================================================
    # Helpers
    # ============================================================

    def get_time_to_expiry(self, state: TradingState) -> float:
        day_fraction_passed = state.timestamp / 1_000_000.0
        days_left = self.STARTING_DAYS_TO_EXPIRY - day_fraction_passed
        days_left = max(0.05, days_left)
        return days_left / self.DAYS_PER_YEAR

    def get_strike(self, product: str) -> int:
        return int(product.split("_")[-1])

    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def get_position_limit(self, product: str) -> int:
        return self.POSITION_LIMITS.get(product, 100)

    def get_best_bid(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders:
            return None
        return max(order_depth.buy_orders.keys())

    def get_best_ask(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys())

    def get_mid_price(self, state: TradingState, product: str) -> Optional[float]:
        if product not in state.order_depths:
            return None

        order_depth = state.order_depths[product]
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)

        return None

    def safe_trader_data(self, state: TradingState) -> str:
        data = getattr(state, "traderData", "")
        if data is None:
            return ""
        return data


# ============================================================
# Round 4 Mark14 tactical HP follower
# ============================================================
# This is intentionally NOT the previous full target-position mode.
# Mark14's historical edge is measured from Mark14's own trade price.
# Therefore we only follow when we can enter near that price, and we cap
# order size to visible top-level volume so no large residual order rests
# at a bad price.

def _best_bid_px_vol(order_depth: OrderDepth):
    if not order_depth.buy_orders:
        return None, 0
    price = max(order_depth.buy_orders.keys())
    return price, order_depth.buy_orders[price]


def _best_ask_px_vol(order_depth: OrderDepth):
    if not order_depth.sell_orders:
        return None, 0
    price = min(order_depth.sell_orders.keys())
    return price, -order_depth.sell_orders[price]


class Mark14HPSignal:
    PRODUCT = "HYDROGEL_PACK"

    # Signal extraction controls
    LOOKBACK = 300       # wider than 100 so we can verify whether Mark14 is actually visible
    TTL = 1500           # how long the last Mark14 direction stays active
    MIN_ABS_FLOW = 1     # minimum net Mark14 quantity required to refresh direction

    # Execution controls
    MAX_CHASE = 4        # diagnostic version: not as tight as 2, but still blocks bad chasing
    TARGET = 70          # tactical target, not full-position Olivia mode
    CLIP = 10            # never send more than first-level visible volume / this cap

    def __init__(self):
        self.direction = 0       # +1 BUY, -1 SELL, 0 none
        self.price = None        # recent Mark14 VWAP for HP
        self.time = -10**9
        self.last_flow = 0

        # Diagnostics: these are persisted in traderData.
        self.diag = self._empty_diag()

    def _empty_diag(self):
        return {
            "seen_ticks": 0,          # ticks where any recent Mark14 HP trade was visible
            "seen_buy_qty": 0,        # total Mark14 buy qty observed
            "seen_sell_qty": 0,       # total Mark14 sell qty observed
            "signal_refreshes": 0,    # times direction was refreshed by Mark14 flow
            "active_ticks": 0,        # ticks where stored signal was active after update
            "orders_sent": 0,         # Mark14 module actually sent an order
            "buy_orders_sent": 0,
            "sell_orders_sent": 0,
            "reject_inactive": 0,
            "reject_no_book": 0,
            "reject_chase": 0,
            "reject_qty": 0,
            "last_reason": "INIT",
            "last_direction": 0,
            "last_mark_price": None,
            "last_best_bid": None,
            "last_best_ask": None,
            "last_flow": 0,
            "last_timestamp": None,
        }

    def load_state(self, data):
        self.direction = int(data.get("direction", 0))
        self.price = data.get("price", None)
        self.time = int(data.get("time", -10**9))
        self.last_flow = int(data.get("last_flow", 0))
        old_diag = data.get("diag", {})
        self.diag = self._empty_diag()
        for k, v in old_diag.items():
            if k in self.diag:
                self.diag[k] = v

    def save_state(self):
        return {
            "direction": self.direction,
            "price": self.price,
            "time": self.time,
            "last_flow": self.last_flow,
            "diag": self.diag,
        }

    def update(self, state: TradingState):
        """Detect Mark14 flow on HP from both market_trades and own_trades.

        This mirrors the Olivia idea: buyer means BUY signal, seller means SELL signal.
        The difference is that we also store Mark14's recent VWAP so execution can avoid
        chasing too far away from the insider's own price.
        """
        now = state.timestamp
        signed_qty = 0
        signed_value = 0.0
        buy_qty = 0
        sell_qty = 0

        for source in (getattr(state, "market_trades", {}) or {}, getattr(state, "own_trades", {}) or {}):
            for trade in source.get(self.PRODUCT, []):
                ts = getattr(trade, "timestamp", now)
                if abs(ts - now) > self.LOOKBACK:
                    continue

                qty = abs(getattr(trade, "quantity", 0) or 0)
                price = float(getattr(trade, "price", 0) or 0)
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)

                if buyer == "Mark 14":
                    signed_qty += qty
                    signed_value += qty * price
                    buy_qty += qty

                if seller == "Mark 14":
                    signed_qty -= qty
                    signed_value -= qty * price
                    sell_qty += qty

        self.last_flow = signed_qty
        self.diag["last_flow"] = signed_qty
        self.diag["last_timestamp"] = now

        if buy_qty or sell_qty:
            self.diag["seen_ticks"] += 1
            self.diag["seen_buy_qty"] += int(buy_qty)
            self.diag["seen_sell_qty"] += int(sell_qty)

        if abs(signed_qty) >= self.MIN_ABS_FLOW:
            self.direction = 1 if signed_qty > 0 else -1
            self.price = abs(signed_value / signed_qty)
            self.time = now
            self.diag["signal_refreshes"] += 1
            self.diag["last_direction"] = self.direction
            self.diag["last_mark_price"] = self.price
            self.diag["last_reason"] = "SIGNAL_REFRESH"

    def is_active(self, state: TradingState):
        return (
            self.direction != 0
            and self.price is not None
            and state.timestamp - self.time <= self.TTL
        )

    def trade(self, state: TradingState, result: Dict[str, List[Order]]) -> bool:
        """Return True only when this module actually sends a Mark14 tactical order.

        Diagnostics distinguish four cases:
        - inactive: Mark14 not visible / signal expired
        - no_book: HP order book unavailable
        - chase: Mark14 visible but current best price is too far from Mark14 VWAP
        - qty: target already reached or no visible first-level volume
        """
        self.update(state)

        if not self.is_active(state):
            self.diag["reject_inactive"] += 1
            self.diag["last_reason"] = "INACTIVE"
            return False

        self.diag["active_ticks"] += 1

        if self.PRODUCT not in state.order_depths:
            self.diag["reject_no_book"] += 1
            self.diag["last_reason"] = "NO_BOOK"
            return False

        od = state.order_depths[self.PRODUCT]
        pos = state.position.get(self.PRODUCT, 0)
        result.setdefault(self.PRODUCT, [])

        bid, bid_vol = _best_bid_px_vol(od)
        ask, ask_vol = _best_ask_px_vol(od)
        self.diag["last_best_bid"] = bid
        self.diag["last_best_ask"] = ask

        if self.direction > 0:
            if ask is None:
                self.diag["reject_no_book"] += 1
                self.diag["last_reason"] = "NO_ASK"
                return False

            if ask > self.price + self.MAX_CHASE:
                self.diag["reject_chase"] += 1
                self.diag["last_reason"] = "CHASE_BUY"
                return False

            desired = self.TARGET - pos
            qty = min(desired, ask_vol, self.CLIP)
            if qty > 0:
                result[self.PRODUCT].append(Order(self.PRODUCT, int(ask), int(qty)))
                self.diag["orders_sent"] += 1
                self.diag["buy_orders_sent"] += 1
                self.diag["last_reason"] = "ORDER_BUY"
                return True

            self.diag["reject_qty"] += 1
            self.diag["last_reason"] = "NO_BUY_QTY"
            return False

        if self.direction < 0:
            if bid is None:
                self.diag["reject_no_book"] += 1
                self.diag["last_reason"] = "NO_BID"
                return False

            if bid < self.price - self.MAX_CHASE:
                self.diag["reject_chase"] += 1
                self.diag["last_reason"] = "CHASE_SELL"
                return False

            desired = pos + self.TARGET
            qty = min(desired, bid_vol, self.CLIP)
            if qty > 0:
                result[self.PRODUCT].append(Order(self.PRODUCT, int(bid), -int(qty)))
                self.diag["orders_sent"] += 1
                self.diag["sell_orders_sent"] += 1
                self.diag["last_reason"] = "ORDER_SELL"
                return True

            self.diag["reject_qty"] += 1
            self.diag["last_reason"] = "NO_SELL_QTY"
            return False

        self.diag["reject_inactive"] += 1
        self.diag["last_reason"] = "ZERO_DIRECTION"
        return False

class Trader:
    def __init__(self):
        self.ve = VelvetfruitStrategy()
        self.hp = HydrogelPackStrategy()
        self.vev = VEVoucherStrategy()
        self.mark14_hp = Mark14HPSignal()

    def load_state(self, state):
        if not state.traderData:
            return

        try:
            data = json.loads(state.traderData)
            self.ve.load_state(data.get("VELVETFRUIT_EXTRACT", {}))
            self.hp.load_state(data.get("HYDROGEL_PACK", {}))
            self.mark14_hp.load_state(data.get("MARK14_HP", {}))
        except Exception:
            self.ve.load_state({})
            self.hp.load_state({})
            self.mark14_hp.load_state({})

    def save_state(self):
        return json.dumps({
            "VELVETFRUIT_EXTRACT": self.ve.save_state(),
            "HYDROGEL_PACK": self.hp.save_state(),
            "VEVOUCHERS": {},
            "MARK14_HP": self.mark14_hp.save_state(),
        })

    def run(self, state: TradingState):
        result = {
            self.ve.PRODUCT: [],
            self.hp.PRODUCT: [],
        }

        self.load_state(state)

        # Keep VE original. Mark14 on VE is weaker and should not override yet.
        if self.ve.PRODUCT in state.order_depths:
            self.ve.position = state.position.get(self.ve.PRODUCT, 0)
            self.ve.trade(state, result[self.ve.PRODUCT])

        # HP: first try a tightly controlled Mark14 follow.
        # If it actually sends an order, skip original HP for this tick so we do not
        # immediately fight the insider direction. If it cannot enter near Mark14's
        # price, fall back to the old HP baseline.
        hp_mark14_traded = False
        if self.hp.PRODUCT in state.order_depths:
            hp_mark14_traded = self.mark14_hp.trade(state, result)

        if self.hp.PRODUCT in state.order_depths and not hp_mark14_traded:
            self.hp.position = state.position.get(self.hp.PRODUCT, 0)
            self.hp.trade(state, result[self.hp.PRODUCT])

        # VEV original model only. Do NOT trade VEV_4000 with Mark14 yet:
        # the spread is too wide, and full target mode caused catastrophic bleed.
        self.vev.DO_DELTA_HEDGE = False
        voucher_result, _, _ = self.vev.run(state)
        for product, orders in voucher_result.items():
            if orders:
                result.setdefault(product, [])
                result[product].extend(orders)

        # Print compact Mark14 diagnostics into the run log so we can verify whether
        # the signal is visible and whether orders are blocked by chase/qty filters.
        # Sparse printing avoids log spam.
        if state.timestamp % 10000 == 0 or state.timestamp >= 99900:
            d = self.mark14_hp.diag
            compact_diag = {
                "ts": state.timestamp,
                "seen": d.get("seen_ticks"),
                "buyq": d.get("seen_buy_qty"),
                "sellq": d.get("seen_sell_qty"),
                "refresh": d.get("signal_refreshes"),
                "active": d.get("active_ticks"),
                "orders": d.get("orders_sent"),
                "buy_orders": d.get("buy_orders_sent"),
                "sell_orders": d.get("sell_orders_sent"),
                "inactive": d.get("reject_inactive"),
                "chase": d.get("reject_chase"),
                "qty_rej": d.get("reject_qty"),
                "reason": d.get("last_reason"),
                "dir": d.get("last_direction"),
                "mark_px": d.get("last_mark_price"),
                "bid": d.get("last_best_bid"),
                "ask": d.get("last_best_ask"),
                "flow": d.get("last_flow"),
            }
            print("MARK14_HP_DIAG", json.dumps(compact_diag, separators=(",", ":")))

        return result, 0, self.save_state()
