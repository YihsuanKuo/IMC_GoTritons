from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


class VelvetfruitStrategy:
    PRODUCT = "VELVETFRUIT_EXTRACT"
    POSITION_LIMIT = 200
    BASE_FAIR = 5255
    HISTORY_LENGTH = 160
    SHORT_WINDOW = 20
    LONG_WINDOW = 80
    REGIME_WINDOW = 60
    BUY_BAND_1 = 5235
    BUY_BAND_2 = 5225
    BUY_BAND_3 = 5215
    SELL_BAND_1 = 5272
    SELL_BAND_2 = 5282
    SELL_BAND_3 = 5292
    SMALL_EDGE = 3
    BIG_EDGE = 10
    EXTREME_EDGE = 20
    MM_SIZE = 6
    NORMAL_SIZE = 22
    BIG_SIZE = 38
    EXTREME_SIZE = 58
    UNWIND_SIZE = 22
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
        return {"mid_history": self.mid_history[-self.HISTORY_LENGTH :]}

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
        vals = self.mid_history[-min(n, len(self.mid_history)) :]
        return sum(vals) / len(vals)

    def get_center(self):
        short_mean = self.avg_last(self.SHORT_WINDOW)
        long_mean = self.avg_last(self.LONG_WINDOW)
        center = 0.55 * self.BASE_FAIR + 0.2 * short_mean + 0.25 * long_mean
        center -= self.INVENTORY_SKEW * self.position
        return center

    def get_trend(self):
        if len(self.mid_history) < self.SHORT_WINDOW:
            return 0
        return self.mid_history[-1] - self.mid_history[-self.SHORT_WINDOW]

    def get_regime(self):
        if len(self.mid_history) < 5:
            return (0, 0)
        recent = self.mid_history[
            -min(self.REGIME_WINDOW, len(self.mid_history)) :
        ]
        high = max(recent)
        low = min(recent)
        mid = self.mid_history[-1]
        return (high - mid, mid - low)

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
        if self.position > self.DANGER_LEVEL:
            self.sell(
                orders,
                best_bid,
                min(self.UNWIND_SIZE, self.position),
                use_soft=False,
            )
            return
        if self.position < -self.DANGER_LEVEL:
            self.buy(
                orders,
                best_ask,
                min(self.UNWIND_SIZE, -self.position),
                use_soft=False,
            )
            return
        if self.position > 25:
            if best_bid >= max(center + 8, 5264):
                self.sell(
                    orders, best_bid, min(34, self.position), use_soft=False
                )
                return
            elif best_bid >= max(center + 4, 5258):
                self.sell(
                    orders, best_bid, min(20, self.position), use_soft=False
                )
                return
            elif best_bid >= 5254 and self.position > 70:
                self.sell(
                    orders, best_bid, min(16, self.position), use_soft=False
                )
                return
        if self.position < -25:
            if best_ask <= min(center - 8, 5244):
                self.buy(
                    orders, best_ask, min(34, -self.position), use_soft=False
                )
                return
            elif best_ask <= min(center - 4, 5250):
                self.buy(
                    orders, best_ask, min(20, -self.position), use_soft=False
                )
                return
            elif best_ask <= 5256 and self.position < -70:
                self.buy(
                    orders, best_ask, min(16, -self.position), use_soft=False
                )
                return
        can_add_long = self.position < self.NO_ADD_LEVEL
        can_add_short = self.position > -self.NO_ADD_LEVEL
        if can_add_long:
            if best_ask <= self.BUY_BAND_3:
                self.buy(orders, best_ask, self.EXTREME_SIZE)
                return
            elif best_ask <= self.BUY_BAND_2:
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
                    for bid, vol in sorted(
                        od.buy_orders.items(), reverse=True
                    ):
                        if bid >= center + self.EXTREME_EDGE:
                            self.sell(orders, bid, min(vol, self.BIG_SIZE))
                            break
            elif sell_edge >= self.BIG_EDGE and mid > center + 8:
                if trend < 10 or drop_from_high >= 3:
                    for bid, vol in sorted(
                        od.buy_orders.items(), reverse=True
                    ):
                        if bid >= center + self.BIG_EDGE:
                            self.sell(orders, bid, min(vol, self.NORMAL_SIZE))
                            break
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
    INVENTORY_SKEW = 0.1
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
        self.last_long_tp_time = -(10**9)
        self.last_short_tp_time = -(10**9)

    def load_state(self, data):
        self.mid_history = data.get("mid_history", [])
        self.last_long_tp_time = data.get("last_long_tp_time", -(10**9))
        self.last_short_tp_time = data.get("last_short_tp_time", -(10**9))

    def save_state(self):
        return {
            "mid_history": self.mid_history[-self.HISTORY_LENGTH :],
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
        vals = self.mid_history[-min(n, len(self.mid_history)) :]
        return sum(vals) / len(vals)

    def get_fair(self):
        short_mean = self.avg_last(self.SHORT_WINDOW)
        long_mean = self.avg_last(self.LONG_WINDOW)
        fair = 0.5 * self.BASE_FAIR + 0.25 * short_mean + 0.25 * long_mean
        fair -= self.INVENTORY_SKEW * self.position
        return fair

    def get_trend(self):
        if len(self.mid_history) < self.SHORT_WINDOW:
            return 0
        return self.mid_history[-1] - self.mid_history[-self.SHORT_WINDOW]

    def get_regime(self):
        if len(self.mid_history) < 5:
            return (0, 0)
        recent = self.mid_history[
            -min(self.REGIME_WINDOW, len(self.mid_history)) :
        ]
        recent_high = max(recent)
        recent_low = min(recent)
        mid = self.mid_history[-1]
        return (recent_high - mid, mid - recent_low)

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
        if abs(micro_signal) <= self.MICRO_CAP:
            fair += self.MICRO_FAIR_WEIGHT * micro_signal
        trend = self.get_trend()
        drop_from_high, rise_from_low = self.get_regime()
        buy_edge = fair - best_ask
        sell_edge = best_bid - fair
        in_long_tp_cooldown = (
            timestamp - self.last_long_tp_time < self.TP_COOLDOWN
        )
        in_short_tp_cooldown = (
            timestamp - self.last_short_tp_time < self.TP_COOLDOWN
        )
        if self.position > self.DANGER_LEVEL:
            if best_bid >= fair - 15:
                self.sell(
                    orders,
                    best_bid,
                    min(self.UNWIND_SIZE, self.position),
                    use_soft=False,
                )
                return
        if self.position < -self.DANGER_LEVEL:
            if best_ask <= fair + 15:
                self.buy(
                    orders,
                    best_ask,
                    min(self.UNWIND_SIZE, -self.position),
                    use_soft=False,
                )
                return
        if self.position > 25:
            if best_bid >= self.LONG_TP_2:
                self.sell(
                    orders, best_bid, min(35, self.position), use_soft=False
                )
                self.last_long_tp_time = timestamp
                return
            elif best_bid >= self.LONG_TP_1:
                self.sell(
                    orders, best_bid, min(22, self.position), use_soft=False
                )
                self.last_long_tp_time = timestamp
                return
            elif best_bid >= fair + 12:
                self.sell(
                    orders, best_bid, min(20, self.position), use_soft=False
                )
                self.last_long_tp_time = timestamp
                return
        if self.position < -25:
            if best_ask <= self.SHORT_TP_2:
                self.buy(
                    orders, best_ask, min(35, -self.position), use_soft=False
                )
                self.last_short_tp_time = timestamp
                return
            elif best_ask <= self.SHORT_TP_1:
                self.buy(
                    orders, best_ask, min(22, -self.position), use_soft=False
                )
                self.last_short_tp_time = timestamp
                return
            elif best_ask <= fair - 12:
                self.buy(
                    orders, best_ask, min(20, -self.position), use_soft=False
                )
                self.last_short_tp_time = timestamp
                return
        if in_long_tp_cooldown and self.position > 0:
            if best_bid >= 9978:
                self.sell(
                    orders, best_bid, min(12, self.position), use_soft=False
                )
                return
        if in_short_tp_cooldown and self.position < 0:
            if best_ask <= 9985:
                self.buy(
                    orders, best_ask, min(12, -self.position), use_soft=False
                )
                return
        rebound_confirmed = (
            rise_from_low >= self.REBOUND_CONFIRM
            or micro_signal >= self.MICRO_CONFIRM
        )
        pullback_confirmed = (
            drop_from_high >= self.REBOUND_CONFIRM
            or micro_signal <= -self.MICRO_CONFIRM
        )
        allow_deep_rebuy_after_cooldown = (
            (
                best_ask <= self.DEEP_REBUY_LEVEL
                or buy_edge >= self.EXTREME_EDGE + 8
            )
            and rebound_confirmed
            and (trend > -10)
        )
        allow_deep_reshort_after_cooldown = (
            (
                best_bid >= self.DEEP_RESHORT_LEVEL
                or sell_edge >= self.EXTREME_EDGE + 8
            )
            and pullback_confirmed
            and (trend < 10)
        )
        block_long_after_tp = in_long_tp_cooldown and (
            not allow_deep_rebuy_after_cooldown
        )
        block_short_after_tp = in_short_tp_cooldown and (
            not allow_deep_reshort_after_cooldown
        )
        can_add_long = self.position < self.NO_ADD_LEVEL and (
            not block_long_after_tp
        )
        can_add_short = self.position > -self.NO_ADD_LEVEL and (
            not block_short_after_tp
        )
        if in_long_tp_cooldown and self.position < 25:
            if (
                buy_edge >= self.BIG_EDGE + 12
                and rebound_confirmed
                and (trend > -8)
            ):
                for ask, vol in sorted(od.sell_orders.items()):
                    if ask <= fair - (self.BIG_EDGE + 8):
                        self.buy(orders, ask, min(-vol, 8))
                        return
        if in_short_tp_cooldown and self.position > -25:
            if (
                sell_edge >= self.BIG_EDGE + 12
                and pullback_confirmed
                and (trend < 8)
            ):
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
                    for bid, vol in sorted(
                        od.buy_orders.items(), reverse=True
                    ):
                        if bid >= fair + self.EXTREME_EDGE:
                            self.sell(orders, bid, min(vol, self.BIG_SIZE))
                            break
            elif sell_edge >= self.BIG_EDGE:
                if trend < 12 and micro_signal < 1.5:
                    for bid, vol in sorted(
                        od.buy_orders.items(), reverse=True
                    ):
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
            if trend < 0 and self.position > 0 and (best_bid >= 9968):
                self.sell(
                    orders, best_bid, min(8, self.position), use_soft=False
                )
            elif trend > 0 and self.position < 0 and (best_ask <= 9992):
                self.buy(
                    orders, best_ask, min(8, -self.position), use_soft=False
                )
            return
        if (
            self.position > 45
            and micro_signal <= -self.MICRO_CONFIRM
            and (best_bid >= fair - 8)
        ):
            self.sell(orders, best_bid, min(10, self.position), use_soft=False)
            return
        if (
            self.position < -45
            and micro_signal >= self.MICRO_CONFIRM
            and (best_ask <= fair + 8)
        ):
            self.buy(orders, best_ask, min(10, -self.position), use_soft=False)
            return
        spread = best_ask - best_bid
        if (
            abs(self.position) <= self.FLAT_POSITION
            and abs(trend) <= self.LIGHT_TREND_LIMIT
            and (spread >= self.LIGHT_SPREAD_MIN)
            and (not block_long_after_tp)
            and (not block_short_after_tp)
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


class VEVoucherStrategy:
    UNDERLYING = "VELVETFRUIT_EXTRACT"
    TRADE_VOUCHERS = [
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_6000",
        "VEV_6500",
    ]
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
    STARTING_DAYS_TO_EXPIRY = 5.0
    MAX_TAKE_SIZE = 5
    MAX_MAKE_SIZE = 6
    PRICE_EDGE = 0.8
    MAX_MARKET_SPREAD = 8
    ENABLE_PASSIVE_QUOTES = True
    LOTTERY_SELL_ONLY = {"VEV_6000", "VEV_6500"}
    LOTTERY_SELL_SIZE = 4
    DEEP_ITM = {"VEV_4500"}
    DEEP_ITM_SIZE = 3
    DEEP_ITM_PREMIUM = {"VEV_4500": 8}
    DEEP_ITM_EDGE = {"VEV_4500": 6}
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
            return (result, 0, self.safe_trader_data(state))
        T = self.get_time_to_expiry(state)
        if T <= 0:
            return (result, 0, self.safe_trader_data(state))
        for voucher in self.TRADE_VOUCHERS:
            if voucher not in state.order_depths:
                continue
            try:
                orders = self.trade_vev_voucher(state, voucher, S, T)
                if orders:
                    result[voucher] = orders
            except Exception:
                continue
        return (result, 0, self.safe_trader_data(state))

    def trade_vev_voucher(
        self, state: TradingState, product: str, S: float, T: float
    ) -> List[Order]:
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders
        if best_bid <= 0:
            return orders
        market_spread = best_ask - best_bid
        if market_spread <= 0:
            return orders
        if market_spread > self.MAX_MARKET_SPREAD:
            return orders
        K = self.get_strike(product)
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
        if product in self.DEEP_ITM:
            position = self.get_position(state, product)
            limit = self.get_position_limit(product)
            intrinsic = max(0.0, S - K)
            premium = self.DEEP_ITM_PREMIUM.get(product, 8)
            edge = self.DEEP_ITM_EDGE.get(product, 6)
            fair = intrinsic + premium
            max_size = self.DEEP_ITM_SIZE
            if best_ask < fair - edge:
                ask_volume = -order_depth.sell_orders[best_ask]
                max_can_buy = limit - position
                qty = min(ask_volume, max_can_buy, max_size)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                return orders
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
        our_bid = math.floor(predicted_bid)
        our_ask = math.ceil(predicted_ask)
        if our_bid >= our_ask:
            our_bid = math.floor(predicted_mid) - 1
            our_ask = math.ceil(predicted_mid) + 1
        intrinsic = max(0.0, S - K)
        our_bid = max(0, our_bid)
        our_ask = max(math.ceil(intrinsic), our_ask)
        position = self.get_position(state, product)
        limit = self.get_position_limit(product)
        if predicted_bid > best_ask + self.PRICE_EDGE:
            ask_volume = -order_depth.sell_orders[best_ask]
            max_can_buy = limit - position
            qty = min(ask_volume, max_can_buy, self.MAX_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
            return orders
        if predicted_ask < best_bid - self.PRICE_EDGE:
            bid_volume = order_depth.buy_orders[best_bid]
            max_can_sell = limit + position
            qty = min(bid_volume, max_can_sell, self.MAX_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
            return orders
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

    def get_curve_iv(self, S: float, K: float, T: float, side: str) -> float:
        if side == "bid":
            a, b, c, d = (self.A_BID, self.B_BID, self.C_BID, self.D_BID)
        elif side == "ask":
            a, b, c, d = (self.A_ASK, self.B_ASK, self.C_ASK, self.D_ASK)
        else:
            a, b, c, d = (self.A_MID, self.B_MID, self.C_MID, self.D_MID)
        if S <= 0 or K <= 0 or T <= 0:
            return d
        m = math.log(K / S) / math.sqrt(T)
        iv = a * m * m * m + b * m * m + c * m + d
        return min(max(iv, 0.01), 3.0)

    def bs_call_price(
        self, S: float, K: float, T: float, sigma: float, r: float = 0.0
    ) -> float:
        if T <= 0:
            return max(0.0, S - K)
        if sigma <= 0:
            return max(0.0, S - K)
        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (
            sigma * sqrt_t
        )
        d2 = d1 - sigma * sqrt_t
        return S * self.normal_cdf(d1) - K * math.exp(
            -r * T
        ) * self.normal_cdf(d2)

    def normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def get_time_to_expiry(self, state: TradingState) -> float:
        day_fraction_passed = state.timestamp / 1000000.0
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

    def get_mid_price(
        self, state: TradingState, product: str
    ) -> Optional[float]:
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


def _best_bid_px_vol(order_depth: OrderDepth):
    if not order_depth.buy_orders:
        return (None, 0)
    price = max(order_depth.buy_orders.keys())
    return (price, order_depth.buy_orders[price])


def _best_ask_px_vol(order_depth: OrderDepth):
    if not order_depth.sell_orders:
        return (None, 0)
    price = min(order_depth.sell_orders.keys())
    return (price, -order_depth.sell_orders[price])


class HPMark14OnlyOverlayV3:
    PRODUCT = "HYDROGEL_PACK"
    LOOKBACK = 350
    TTL = 1200
    MIN_ABS_FLOW = 1
    STRONG_FLOW = 10
    TARGET = 75
    STRONG_TARGET = 95
    ACTIVE_CLIP = 7
    PASSIVE_CLIP = 5
    ACTIVE_EDGE = 4
    COVER_EDGE = 10
    PASSIVE_EDGE = 8
    MAX_ACTIVE_SPREAD = 18
    MAX_PASSIVE_SPREAD = 26

    def __init__(self):
        self.direction = 0
        self.price = None
        self.time = -(10**9)
        self.last_flow = 0
        self.diag = self._empty_diag()

    def _empty_diag(self):
        return {
            "seen_ticks": 0,
            "seen_buy_qty": 0,
            "seen_sell_qty": 0,
            "signal_refreshes": 0,
            "active_ticks": 0,
            "orders_sent": 0,
            "buy_orders_sent": 0,
            "sell_orders_sent": 0,
            "active_orders": 0,
            "passive_orders": 0,
            "reject_inactive": 0,
            "reject_no_book": 0,
            "reject_price_filter": 0,
            "reject_spread": 0,
            "reject_qty": 0,
            "last_reason": "INIT",
            "last_direction": 0,
            "last_mark_price": None,
            "last_best_bid": None,
            "last_best_ask": None,
            "last_fair": None,
            "last_flow": 0,
            "last_timestamp": None,
            "last_order_px": None,
            "last_order_qty": 0,
            "last_mode": None,
        }

    def load_state(self, data):
        self.direction = int(data.get("direction", 0))
        self.price = data.get("price", None)
        self.time = int(data.get("time", -(10**9)))
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

    def _virtual_position(
        self, state: TradingState, result: Dict[str, List[Order]]
    ):
        pos = state.position.get(self.PRODUCT, 0)
        for order in result.get(self.PRODUCT, []):
            pos += order.quantity
        return pos

    def update(self, state: TradingState):
        now = state.timestamp
        signed_qty = 0
        signed_value = 0.0
        buy_qty = 0
        sell_qty = 0
        for source in (
            getattr(state, "market_trades", {}) or {},
            getattr(state, "own_trades", {}) or {},
        ):
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
            self.price = (
                abs(signed_value / signed_qty) if signed_qty != 0 else None
            )
            self.time = now
            self.diag["signal_refreshes"] += 1
            self.diag["last_direction"] = self.direction
            self.diag["last_mark_price"] = self.price
            self.diag["last_reason"] = "SIGNAL_REFRESH"

    def is_active(self, state: TradingState):
        return (
            self.direction != 0
            and self.price is not None
            and (state.timestamp - self.time <= self.TTL)
        )

    def _append_order(self, result, price, qty, mode):
        result.setdefault(self.PRODUCT, [])
        if qty == 0:
            return False
        result[self.PRODUCT].append(Order(self.PRODUCT, int(price), int(qty)))
        self.diag["orders_sent"] += 1
        self.diag["last_order_px"] = int(price)
        self.diag["last_order_qty"] = int(qty)
        self.diag["last_mode"] = mode
        if qty > 0:
            self.diag["buy_orders_sent"] += 1
        else:
            self.diag["sell_orders_sent"] += 1
        if mode == "ACTIVE":
            self.diag["active_orders"] += 1
        else:
            self.diag["passive_orders"] += 1
        return True

    def trade(
        self,
        state: TradingState,
        result: Dict[str, List[Order]],
        fair: Optional[float] = None,
    ) -> bool:
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
        bid, bid_vol = _best_bid_px_vol(od)
        ask, ask_vol = _best_ask_px_vol(od)
        self.diag["last_best_bid"] = bid
        self.diag["last_best_ask"] = ask
        if bid is None or ask is None:
            self.diag["reject_no_book"] += 1
            self.diag["last_reason"] = "NO_TOP"
            return False
        spread = ask - bid
        if fair is None:
            fair = (bid + ask) / 2
        self.diag["last_fair"] = fair
        pos = self._virtual_position(state, result)
        strong = abs(self.last_flow) >= self.STRONG_FLOW
        target = self.STRONG_TARGET if strong else self.TARGET
        if self.direction > 0:
            desired = target - pos
            if desired <= 0:
                self.diag["reject_qty"] += 1
                self.diag["last_reason"] = "NO_BUY_TARGET"
                return False
            active_edge = self.COVER_EDGE if pos < -20 else self.ACTIVE_EDGE
            if spread <= self.MAX_ACTIVE_SPREAD and ask <= fair + active_edge:
                qty = min(desired, ask_vol, self.ACTIVE_CLIP)
                if qty > 0:
                    self.diag["last_reason"] = "ACTIVE_BUY_FAIR_FILTERED"
                    return self._append_order(result, ask, qty, "ACTIVE")
            if spread > self.MAX_PASSIVE_SPREAD:
                self.diag["reject_spread"] += 1
                self.diag["last_reason"] = "BUY_SPREAD_TOO_WIDE"
                return False
            px = min(
                bid + 1, ask - 1, int(math.floor(fair + self.PASSIVE_EDGE))
            )
            if px <= bid or px >= ask:
                self.diag["reject_price_filter"] += 1
                self.diag["last_reason"] = "BUY_NO_SAFE_PASSIVE_PRICE"
                return False
            qty = min(desired, self.PASSIVE_CLIP)
            if qty > 0:
                self.diag["last_reason"] = "PASSIVE_BUY_BIAS"
                return self._append_order(result, px, qty, "PASSIVE")
            self.diag["reject_qty"] += 1
            self.diag["last_reason"] = "NO_BUY_QTY"
            return False
        if self.direction < 0:
            desired = pos + target
            if desired <= 0:
                self.diag["reject_qty"] += 1
                self.diag["last_reason"] = "NO_SELL_TARGET"
                return False
            active_edge = self.COVER_EDGE if pos > 20 else self.ACTIVE_EDGE
            if spread <= self.MAX_ACTIVE_SPREAD and bid >= fair - active_edge:
                qty = min(desired, bid_vol, self.ACTIVE_CLIP)
                if qty > 0:
                    self.diag["last_reason"] = "ACTIVE_SELL_FAIR_FILTERED"
                    return self._append_order(result, bid, -qty, "ACTIVE")
            if spread > self.MAX_PASSIVE_SPREAD:
                self.diag["reject_spread"] += 1
                self.diag["last_reason"] = "SELL_SPREAD_TOO_WIDE"
                return False
            px = max(
                ask - 1, bid + 1, int(math.ceil(fair - self.PASSIVE_EDGE))
            )
            if px <= bid or px >= ask:
                self.diag["reject_price_filter"] += 1
                self.diag["last_reason"] = "SELL_NO_SAFE_PASSIVE_PRICE"
                return False
            qty = min(desired, self.PASSIVE_CLIP)
            if qty > 0:
                self.diag["last_reason"] = "PASSIVE_SELL_BIAS"
                return self._append_order(result, px, -qty, "PASSIVE")
            self.diag["reject_qty"] += 1
            self.diag["last_reason"] = "NO_SELL_QTY"
            return False
        self.diag["reject_inactive"] += 1
        self.diag["last_reason"] = "ZERO_DIRECTION"
        return False


class VEInsiderOverlay:
    PRODUCT = "VELVETFRUIT_EXTRACT"
    LOOKBACK = 400
    TTL = 1300
    MIN_FLOW = 9.0
    PASSIVE_TARGET = 100
    ACTIVE_TARGET = 125
    PASSIVE_CLIP = 4
    ACTIVE_CLIP = 5
    MAX_PASSIVE_SPREAD = 8
    MAX_ACTIVE_SPREAD = 4
    STRONG_FLOW = 24.0
    EXTREME_POS_GUARD = 175

    def __init__(self):
        self.direction = 0
        self.flow = 0.0
        self.time = -(10**9)
        self.diag = self._empty_diag()

    def _empty_diag(self):
        return {
            "refresh": 0,
            "active": 0,
            "orders": 0,
            "passive_orders": 0,
            "active_orders": 0,
            "buy_orders": 0,
            "sell_orders": 0,
            "skip_spread": 0,
            "skip_qty": 0,
            "skip_pos": 0,
            "last_dir": 0,
            "last_flow": 0,
            "last_reason": "INIT",
            "last_bid": None,
            "last_ask": None,
            "last_pos": 0,
            "last_px": None,
        }

    def load_state(self, data):
        self.direction = int(data.get("direction", 0))
        self.flow = float(data.get("flow", 0.0))
        self.time = int(data.get("time", -(10**9)))
        old = data.get("diag", {})
        self.diag = self._empty_diag()
        for k, v in old.items():
            if k in self.diag:
                self.diag[k] = v

    def save_state(self):
        return {
            "direction": self.direction,
            "flow": self.flow,
            "time": self.time,
            "diag": self.diag,
        }

    def _weighted_flow(self, state: TradingState):
        now = state.timestamp
        flow = 0.0
        for source in (
            getattr(state, "market_trades", {}) or {},
            getattr(state, "own_trades", {}) or {},
        ):
            for trade in source.get(self.PRODUCT, []):
                ts = getattr(trade, "timestamp", now)
                if abs(ts - now) > self.LOOKBACK:
                    continue
                qty = abs(getattr(trade, "quantity", 0) or 0)
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)
                if buyer == "Mark 14":
                    flow += 1.5 * qty
                if seller == "Mark 14":
                    flow -= 1.5 * qty
                if buyer == "Mark 55":
                    flow -= 1.0 * qty
                if seller == "Mark 55":
                    flow += 1.0 * qty
                if buyer == "Mark 67":
                    flow += 0.8 * qty
                if seller == "Mark 49":
                    flow -= 0.7 * qty
                if seller == "Mark 22":
                    flow -= 0.45 * qty
                if buyer == "Mark 22":
                    flow += 0.25 * qty
        return flow

    def update(self, state: TradingState):
        flow = self._weighted_flow(state)
        self.diag["last_flow"] = round(flow, 2)
        if abs(flow) >= self.MIN_FLOW:
            self.direction = 1 if flow > 0 else -1
            self.flow = flow
            self.time = state.timestamp
            self.diag["refresh"] += 1
            self.diag["last_dir"] = self.direction
            self.diag["last_reason"] = "SIGNAL_REFRESH"

    def is_active(self, state: TradingState):
        return self.direction != 0 and state.timestamp - self.time <= self.TTL

    def apply(self, state: TradingState, result: Dict[str, List[Order]]):
        self.update(state)
        if not self.is_active(state):
            self.diag["last_reason"] = "INACTIVE"
            return
        if self.PRODUCT not in state.order_depths:
            self.diag["last_reason"] = "NO_BOOK"
            return
        od = state.order_depths[self.PRODUCT]
        bid, bid_vol = _best_bid_px_vol(od)
        ask, ask_vol = _best_ask_px_vol(od)
        self.diag["last_bid"] = bid
        self.diag["last_ask"] = ask
        if bid is None or ask is None:
            self.diag["last_reason"] = "NO_TOP"
            return
        spread = ask - bid
        pos = state.position.get(self.PRODUCT, 0)
        self.diag["last_pos"] = pos
        self.diag["active"] += 1
        if abs(pos) >= self.EXTREME_POS_GUARD:
            self.diag["skip_pos"] += 1
            self.diag["last_reason"] = "POS_GUARD"
            return
        result.setdefault(self.PRODUCT, [])
        is_strong = abs(self.flow) >= self.STRONG_FLOW
        if self.direction > 0:
            if pos >= self.PASSIVE_TARGET and (not is_strong):
                self.diag["skip_qty"] += 1
                self.diag["last_reason"] = "LONG_TARGET"
                return
            if (
                is_strong
                and spread <= self.MAX_ACTIVE_SPREAD
                and (pos < self.ACTIVE_TARGET)
            ):
                qty = min(self.ACTIVE_TARGET - pos, ask_vol, self.ACTIVE_CLIP)
                if qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(ask), int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["active_orders"] += 1
                    self.diag["buy_orders"] += 1
                    self.diag["last_px"] = ask
                    self.diag["last_reason"] = "ACTIVE_BUY_SCORE"
                    return
            if spread <= self.MAX_PASSIVE_SPREAD and pos < self.PASSIVE_TARGET:
                quote = min(bid + 1, ask - 1)
                qty = min(self.PASSIVE_TARGET - pos, self.PASSIVE_CLIP)
                if quote > bid and qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(quote), int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["passive_orders"] += 1
                    self.diag["buy_orders"] += 1
                    self.diag["last_px"] = quote
                    self.diag["last_reason"] = "PASSIVE_BUY_SCORE"
                    return
        else:
            if pos <= -self.PASSIVE_TARGET and (not is_strong):
                self.diag["skip_qty"] += 1
                self.diag["last_reason"] = "SHORT_TARGET"
                return
            if (
                is_strong
                and spread <= self.MAX_ACTIVE_SPREAD
                and (pos > -self.ACTIVE_TARGET)
            ):
                qty = min(pos + self.ACTIVE_TARGET, bid_vol, self.ACTIVE_CLIP)
                if qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(bid), -int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["active_orders"] += 1
                    self.diag["sell_orders"] += 1
                    self.diag["last_px"] = bid
                    self.diag["last_reason"] = "ACTIVE_SELL_SCORE"
                    return
            if (
                spread <= self.MAX_PASSIVE_SPREAD
                and pos > -self.PASSIVE_TARGET
            ):
                quote = max(ask - 1, bid + 1)
                qty = min(pos + self.PASSIVE_TARGET, self.PASSIVE_CLIP)
                if quote < ask and qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(quote), -int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["passive_orders"] += 1
                    self.diag["sell_orders"] += 1
                    self.diag["last_px"] = quote
                    self.diag["last_reason"] = "PASSIVE_SELL_SCORE"
                    return
        if spread > self.MAX_PASSIVE_SPREAD:
            self.diag["skip_spread"] += 1
            self.diag["last_reason"] = "SPREAD_SKIP"
        else:
            self.diag["skip_qty"] += 1
            self.diag["last_reason"] = "NO_QTY"


class VEV4000ShortInsider:
    PRODUCT = "VEV_4000"
    LOOKBACK = 350
    TTL = 1400
    MIN_SHORT_FLOW = 2.0
    PASSIVE_TARGET_SHORT = 90
    ACTIVE_TARGET_SHORT = 110
    PASSIVE_CLIP = 5
    ACTIVE_CLIP = 3
    MAX_PASSIVE_SPREAD = 36
    MAX_ACTIVE_SPREAD = 22
    MAX_ACTIVE_CHASE = 5
    STRONG_FLOW = 5.5
    COVER_TRIGGER = -65
    COVER_CLIP = 4
    COVER_MAX_SPREAD = 32

    def __init__(self):
        self.short_signal = 0
        self.ref_price = None
        self.time = -(10**9)
        self.last_flow = 0.0
        self.diag = self._empty_diag()

    def _empty_diag(self):
        return {
            "refresh": 0,
            "active": 0,
            "orders": 0,
            "sell_orders": 0,
            "passive_sells": 0,
            "active_sells": 0,
            "cover_orders": 0,
            "reject_spread": 0,
            "reject_chase": 0,
            "reject_qty": 0,
            "last_flow": 0,
            "last_ref": None,
            "last_bid": None,
            "last_ask": None,
            "last_pos": 0,
            "last_px": None,
            "last_reason": "INIT",
        }

    def load_state(self, data):
        self.short_signal = int(data.get("short_signal", 0))
        self.ref_price = data.get("ref_price", None)
        self.time = int(data.get("time", -(10**9)))
        self.last_flow = float(data.get("last_flow", 0))
        old = data.get("diag", {})
        self.diag = self._empty_diag()
        for k, v in old.items():
            if k in self.diag:
                self.diag[k] = v

    def save_state(self):
        return {
            "short_signal": self.short_signal,
            "ref_price": self.ref_price,
            "time": self.time,
            "last_flow": self.last_flow,
            "diag": self.diag,
        }

    def update(self, state: TradingState):
        now = state.timestamp
        short_flow = 0.0
        notional = 0.0
        qty_total = 0
        for source in (
            getattr(state, "market_trades", {}) or {},
            getattr(state, "own_trades", {}) or {},
        ):
            for trade in source.get(self.PRODUCT, []):
                ts = getattr(trade, "timestamp", now)
                if abs(ts - now) > self.LOOKBACK:
                    continue
                qty = abs(getattr(trade, "quantity", 0) or 0)
                price = float(getattr(trade, "price", 0) or 0)
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)
                if seller == "Mark 14":
                    short_flow += 1.9 * qty
                    notional += price * qty
                    qty_total += qty
                if buyer == "Mark 38":
                    short_flow += 1.2 * qty
                    notional += price * qty
                    qty_total += qty
                if seller == "Mark 22":
                    short_flow += 0.55 * qty
                    notional += price * qty
                    qty_total += qty
                if buyer == "Mark 14":
                    short_flow -= 1.3 * qty
                if seller == "Mark 38":
                    short_flow -= 0.8 * qty
        self.last_flow = short_flow
        self.diag["last_flow"] = round(short_flow, 2)
        if short_flow >= self.MIN_SHORT_FLOW and qty_total > 0:
            self.short_signal = 1
            self.ref_price = notional / qty_total
            self.time = now
            self.diag["refresh"] += 1
            self.diag["last_ref"] = round(self.ref_price, 2)
            self.diag["last_reason"] = "SHORT_SIGNAL"

    def is_active(self, state: TradingState):
        return (
            self.short_signal == 1
            and self.ref_price is not None
            and (state.timestamp - self.time <= self.TTL)
        )

    def apply(self, state: TradingState, result: Dict[str, List[Order]]):
        self.update(state)
        if self.PRODUCT not in state.order_depths:
            self.diag["last_reason"] = "NO_BOOK"
            return
        od = state.order_depths[self.PRODUCT]
        bid, bid_vol = _best_bid_px_vol(od)
        ask, ask_vol = _best_ask_px_vol(od)
        self.diag["last_bid"] = bid
        self.diag["last_ask"] = ask
        if bid is None or ask is None:
            self.diag["last_reason"] = "NO_TOP"
            return
        spread = ask - bid
        pos = state.position.get(self.PRODUCT, 0)
        self.diag["last_pos"] = pos
        result.setdefault(self.PRODUCT, [])
        if not self.is_active(state):
            if pos <= self.COVER_TRIGGER and spread <= self.COVER_MAX_SPREAD:
                qty = min(-pos, ask_vol, self.COVER_CLIP)
                if qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(ask), int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["cover_orders"] += 1
                    self.diag["last_px"] = ask
                    self.diag["last_reason"] = "STALE_COVER"
                    return
            self.diag["last_reason"] = "INACTIVE"
            return
        self.diag["active"] += 1
        if spread <= 0 or spread > self.MAX_PASSIVE_SPREAD:
            self.diag["reject_spread"] += 1
            self.diag["last_reason"] = "SPREAD_SKIP"
            return
        if (
            self.last_flow >= self.STRONG_FLOW
            and spread <= self.MAX_ACTIVE_SPREAD
        ):
            if (
                bid >= self.ref_price - self.MAX_ACTIVE_CHASE
                and pos > -self.ACTIVE_TARGET_SHORT
            ):
                qty = min(
                    pos + self.ACTIVE_TARGET_SHORT, bid_vol, self.ACTIVE_CLIP
                )
                if qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(bid), -int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["sell_orders"] += 1
                    self.diag["active_sells"] += 1
                    self.diag["last_px"] = bid
                    self.diag["last_reason"] = "ACTIVE_SELL_V4000"
                    return
            else:
                self.diag["reject_chase"] += 1
        if pos > -self.PASSIVE_TARGET_SHORT:
            quote = max(bid + 1, ask - 1)
            if quote < ask:
                qty = min(pos + self.PASSIVE_TARGET_SHORT, self.PASSIVE_CLIP)
                if qty > 0:
                    result[self.PRODUCT].append(
                        Order(self.PRODUCT, int(quote), -int(qty))
                    )
                    self.diag["orders"] += 1
                    self.diag["sell_orders"] += 1
                    self.diag["passive_sells"] += 1
                    self.diag["last_px"] = quote
                    self.diag["last_reason"] = "PASSIVE_SELL_V4000"
                    return
        self.diag["reject_qty"] += 1
        self.diag["last_reason"] = "NO_QTY"


class HighStrikeDecayShort:
    PRODUCTS = ["VEV_5400", "VEV_5500"]
    LOOKBACK = 400
    TTL = 1800
    MIN_FLOW = 2.0
    PARAMS = {
        "VEV_5400": {
            "target": 110,
            "clip": 5,
            "min_bid": 7,
            "max_spread": 3,
            "active_spread": 2,
        },
        "VEV_5500": {
            "target": 70,
            "clip": 4,
            "min_bid": 3,
            "max_spread": 2,
            "active_spread": 1,
        },
    }

    def __init__(self):
        self.signals = {
            p: {"flow": 0.0, "time": -(10**9), "ref": None}
            for p in self.PRODUCTS
        }
        self.diag = {p: self._empty_diag() for p in self.PRODUCTS}

    def _empty_diag(self):
        return {
            "refresh": 0,
            "active": 0,
            "orders": 0,
            "sells": 0,
            "reject": 0,
            "flow": 0,
            "pos": 0,
            "bid": None,
            "ask": None,
            "px": None,
            "reason": "INIT",
        }

    def load_state(self, data):
        sig = data.get("signals", {})
        for p in self.PRODUCTS:
            if p in sig:
                self.signals[p] = sig[p]
        old = data.get("diag", {})
        self.diag = {p: self._empty_diag() for p in self.PRODUCTS}
        for p, d in old.items():
            if p in self.diag:
                for k, v in d.items():
                    if k in self.diag[p]:
                        self.diag[p][k] = v

    def save_state(self):
        return {"signals": self.signals, "diag": self.diag}

    def update_product(self, state: TradingState, product: str):
        now = state.timestamp
        flow = 0.0
        notional = 0.0
        qty_total = 0
        for source in (
            getattr(state, "market_trades", {}) or {},
            getattr(state, "own_trades", {}) or {},
        ):
            for trade in source.get(product, []):
                ts = getattr(trade, "timestamp", now)
                if abs(ts - now) > self.LOOKBACK:
                    continue
                qty = abs(getattr(trade, "quantity", 0) or 0)
                price = float(getattr(trade, "price", 0) or 0)
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)
                if seller == "Mark 22":
                    flow += 1.25 * qty
                    notional += price * qty
                    qty_total += qty
                if buyer == "Mark 01":
                    flow += 0.75 * qty
                    notional += price * qty
                    qty_total += qty
                if buyer == "Mark 22":
                    flow -= 0.75 * qty
                if seller == "Mark 01":
                    flow -= 0.45 * qty
        self.diag[product]["flow"] = round(flow, 2)
        if flow >= self.MIN_FLOW and qty_total > 0:
            self.signals[product] = {
                "flow": flow,
                "time": now,
                "ref": notional / qty_total,
            }
            self.diag[product]["refresh"] += 1
            self.diag[product]["reason"] = "SHORTVOL_SIGNAL"

    def is_active(self, state: TradingState, product: str):
        sig = self.signals.get(product, {})
        return (
            sig.get("flow", 0) >= self.MIN_FLOW
            and state.timestamp - int(sig.get("time", -(10**9))) <= self.TTL
        )

    def apply(self, state: TradingState, result: Dict[str, List[Order]]):
        for p in self.PRODUCTS:
            self.update_product(state, p)
            if p not in state.order_depths:
                continue
            od = state.order_depths[p]
            bid, bid_vol = _best_bid_px_vol(od)
            ask, ask_vol = _best_ask_px_vol(od)
            d = self.diag[p]
            d["bid"] = bid
            d["ask"] = ask
            if bid is None or ask is None:
                d["reason"] = "NO_TOP"
                continue
            pos = state.position.get(p, 0)
            d["pos"] = pos
            if not self.is_active(state, p):
                d["reason"] = "INACTIVE"
                continue
            params = self.PARAMS[p]
            spread = ask - bid
            d["active"] += 1
            if (
                spread <= 0
                or spread > params["max_spread"]
                or bid < params["min_bid"]
            ):
                d["reject"] += 1
                d["reason"] = "SPREAD_OR_PRICE_SKIP"
                continue
            target = params["target"]
            clip = params["clip"]
            if pos <= -target:
                d["reject"] += 1
                d["reason"] = "TARGET_REACHED"
                continue
            result.setdefault(p, [])
            sig = self.signals[p]
            if (
                spread <= params["active_spread"]
                and bid >= (sig.get("ref") or bid) - 1
            ):
                qty = min(pos + target, bid_vol, clip)
                if qty > 0:
                    result[p].append(Order(p, int(bid), -int(qty)))
                    d["orders"] += 1
                    d["sells"] += 1
                    d["px"] = bid
                    d["reason"] = "ACTIVE_SHORT_HIGH_STRIKE"
                    continue
            quote = max(bid + 1, ask - 1)
            if quote < ask:
                qty = min(pos + target, clip)
                if qty > 0:
                    result[p].append(Order(p, int(quote), -int(qty)))
                    d["orders"] += 1
                    d["sells"] += 1
                    d["px"] = quote
                    d["reason"] = "PASSIVE_SHORT_HIGH_STRIKE"
                    continue
            d["reject"] += 1
            d["reason"] = "NO_QTY"


class VEV5300DecayShort:
    PRODUCT = "VEV_5300"
    LOOKBACK = 400
    TTL = 1600
    MIN_FLOW = 2.0
    BASE_TARGET = 160
    MID_TARGET = 230
    MAX_TARGET = 300
    CLIP_BASE = 3
    CLIP_MID = 5
    CLIP_MAX = 6
    COVER_CLIP = 3
    MIN_BID = 38
    MID_BID = 48
    HIGH_BID = 54
    MAX_SPREAD = 3
    ACTIVE_SPREAD = 2

    def __init__(self):
        self.signal = {"flow": 0.0, "time": -(10**9), "ref": None}
        self.diag = self._empty_diag()

    def _empty_diag(self):
        return {
            "refresh": 0,
            "active": 0,
            "orders": 0,
            "sells": 0,
            "covers": 0,
            "reject": 0,
            "flow": 0,
            "pos": 0,
            "bid": None,
            "ask": None,
            "ref": None,
            "px": None,
            "target": 0,
            "reason": "INIT",
        }

    def load_state(self, data):
        self.signal = data.get("signal", self.signal)
        old = data.get("diag", {})
        self.diag = self._empty_diag()
        for k, v in old.items():
            if k in self.diag:
                self.diag[k] = v

    def save_state(self):
        return {"signal": self.signal, "diag": self.diag}

    def update(self, state: TradingState):
        now = state.timestamp
        flow = 0.0
        notional = 0.0
        qty_total = 0
        for source in (
            getattr(state, "market_trades", {}) or {},
            getattr(state, "own_trades", {}) or {},
        ):
            for trade in source.get(self.PRODUCT, []):
                ts = getattr(trade, "timestamp", now)
                if abs(ts - now) > self.LOOKBACK:
                    continue
                qty = abs(getattr(trade, "quantity", 0) or 0)
                price = float(getattr(trade, "price", 0) or 0)
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)
                if buyer == "Mark 01":
                    flow += 0.85 * qty
                    notional += price * qty
                    qty_total += qty
                if seller == "Mark 14":
                    flow += 1.15 * qty
                    notional += price * qty
                    qty_total += qty
                if seller == "Mark 22":
                    flow += 0.65 * qty
                    notional += price * qty
                    qty_total += qty
                if seller == "Mark 01":
                    flow -= 0.45 * qty
                if buyer == "Mark 14":
                    flow -= 0.75 * qty
                if buyer == "Mark 22":
                    flow -= 0.35 * qty
        self.diag["flow"] = round(flow, 2)
        if flow >= self.MIN_FLOW and qty_total > 0:
            self.signal = {
                "flow": flow,
                "time": now,
                "ref": notional / qty_total,
            }
            self.diag["refresh"] += 1
            self.diag["ref"] = round(self.signal["ref"], 2)
            self.diag["reason"] = "VEV5300_SHORT_SIGNAL"

    def is_active(self, state: TradingState):
        return (
            self.signal.get("flow", 0) >= self.MIN_FLOW
            and state.timestamp - int(self.signal.get("time", -(10**9)))
            <= self.TTL
        )

    def target_clip(self, bid, ref, flow, pos):
        target = self.BASE_TARGET
        clip = self.CLIP_BASE
        high_ref = ref is not None and ref >= self.HIGH_BID
        mid_ref = ref is not None and ref >= self.MID_BID
        if bid >= self.HIGH_BID or high_ref or flow >= 8:
            target = self.MAX_TARGET
            clip = self.CLIP_MAX
        elif bid >= self.MID_BID or mid_ref or flow >= 4.5:
            target = self.MID_TARGET
            clip = self.CLIP_MID
        if pos < -220 and bid < self.MID_BID and (flow < 6):
            target = min(target, 230)
            clip = min(clip, 4)
        return (target, clip)

    def apply(self, state: TradingState, result: Dict[str, List[Order]]):
        self.update(state)
        p = self.PRODUCT
        if p not in state.order_depths:
            self.diag["reason"] = "NO_BOOK"
            return
        od = state.order_depths[p]
        bid, bid_vol = _best_bid_px_vol(od)
        ask, ask_vol = _best_ask_px_vol(od)
        self.diag["bid"] = bid
        self.diag["ask"] = ask
        if bid is None or ask is None:
            self.diag["reason"] = "NO_TOP"
            return
        pos = state.position.get(p, 0)
        self.diag["pos"] = pos
        spread = ask - bid
        ref = self.signal.get("ref")
        flow = float(self.signal.get("flow", 0) or 0)
        if (
            pos < -200
            and ref is not None
            and (ask <= ref - 5)
            and (spread <= self.MAX_SPREAD)
            and (bid < self.MID_BID)
        ):
            qty = min(
                -pos - 160 if -pos > 160 else 0, ask_vol, self.COVER_CLIP
            )
            if qty > 0:
                result.setdefault(p, []).append(Order(p, int(ask), int(qty)))
                self.diag["orders"] += 1
                self.diag["covers"] += 1
                self.diag["px"] = ask
                self.diag["reason"] = "COVER_DECAY_WIN"
                return
        if not self.is_active(state):
            self.diag["reason"] = "INACTIVE"
            return
        self.diag["active"] += 1
        if spread <= 0 or spread > self.MAX_SPREAD or bid < self.MIN_BID:
            self.diag["reject"] += 1
            self.diag["reason"] = "SPREAD_OR_PRICE_SKIP"
            return
        target, clip = self.target_clip(bid, ref, flow, pos)
        self.diag["target"] = target
        if pos <= -target:
            self.diag["reject"] += 1
            self.diag["reason"] = "TARGET_REACHED"
            return
        result.setdefault(p, [])
        if spread <= self.ACTIVE_SPREAD and (
            ref is None or bid >= ref - 2 or bid >= self.HIGH_BID
        ):
            qty = min(pos + target, bid_vol, clip)
            if qty > 0:
                result[p].append(Order(p, int(bid), -int(qty)))
                self.diag["orders"] += 1
                self.diag["sells"] += 1
                self.diag["px"] = bid
                self.diag["reason"] = "ACTIVE_SHORT_5300_STAGE"
                return
        quote = max(bid + 1, ask - 1)
        if quote < ask:
            qty = min(pos + target, clip)
            if qty > 0:
                result[p].append(Order(p, int(quote), -int(qty)))
                self.diag["orders"] += 1
                self.diag["sells"] += 1
                self.diag["px"] = quote
                self.diag["reason"] = "PASSIVE_SHORT_5300_STAGE"
                return
        self.diag["reject"] += 1
        self.diag["reason"] = "NO_QTY"


class VEVPremiumReversionOverlay:
    UNDERLYING = "VELVETFRUIT_EXTRACT"
    PRODUCTS = {
        "VEV_4000": {
            "strike": 4000,
            "max_spread": 32,
            "base": 165,
            "strong": 225,
            "extreme": 270,
            "clip": 5,
            "sclip": 8,
            "xclip": 10,
            "abs": 2.35,
            "z": 1.10,
            "sabs": 3.35,
            "sz": 1.55,
            "xabs": 4.55,
            "xz": 2.05,
            "flat": 0.55,
        },
        "VEV_4500": {
            "strike": 4500,
            "max_spread": 26,
            "base": 155,
            "strong": 210,
            "extreme": 255,
            "clip": 5,
            "sclip": 8,
            "xclip": 10,
            "abs": 1.95,
            "z": 1.05,
            "sabs": 2.95,
            "sz": 1.50,
            "xabs": 4.10,
            "xz": 2.00,
            "flat": 0.50,
        },
        "VEV_5000": {
            "strike": 5000,
            "max_spread": 18,
            "base": 45,
            "strong": 80,
            "extreme": 110,
            "clip": 3,
            "sclip": 4,
            "xclip": 5,
            "abs": 1.65,
            "z": 1.25,
            "sabs": 2.45,
            "sz": 1.70,
            "xabs": 3.45,
            "xz": 2.20,
            "flat": 0.35,
        },
    }
    HISTORY_LENGTH = 260
    MIN_HISTORY = 35
    POSITION_LIMIT = 300

    def __init__(self):
        self.premium_history = {p: [] for p in self.PRODUCTS}
        self.diag = self._empty_diag()

    def _empty_diag(self):
        return {
            "orders": 0,
            "buy_orders": 0,
            "sell_orders": 0,
            "rich_signals": 0,
            "cheap_signals": 0,
            "cover_orders": 0,
            "skip_history": 0,
            "skip_spread": 0,
            "skip_qty": 0,
            "last_product": None,
            "last_premium": None,
            "last_mean": None,
            "last_std": None,
            "last_z": None,
            "last_reason": "INIT",
            "last_pos": 0,
            "last_bid": None,
            "last_ask": None,
            "last_px": None,
            "last_qty": 0,
            "last_target": 0,
        }

    def load_state(self, data):
        h = data.get("premium_history", {})
        self.premium_history = {
            p: list(h.get(p, []))[-self.HISTORY_LENGTH :]
            for p in self.PRODUCTS
        }
        old = data.get("diag", {})
        self.diag = self._empty_diag()
        for k, v in old.items():
            if k in self.diag:
                self.diag[k] = v

    def save_state(self):
        return {
            "premium_history": {
                p: self.premium_history.get(p, [])[-self.HISTORY_LENGTH :]
                for p in self.PRODUCTS
            },
            "diag": self.diag,
        }

    def _mid_from_depth(self, od):
        bid, _ = _best_bid_px_vol(od)
        ask, _ = _best_ask_px_vol(od)
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        return None

    def _virtual_position(self, state, result, product):
        pos = state.position.get(product, 0)
        for o in result.get(product, []):
            pos += o.quantity
        return pos

    def _append_order(self, result, product, price, qty, reason):
        if qty == 0:
            return False
        result.setdefault(product, [])
        result[product].append(Order(product, int(price), int(qty)))
        d = self.diag
        d["orders"] += 1
        d["last_product"] = product
        d["last_px"] = int(price)
        d["last_qty"] = int(qty)
        d["last_reason"] = reason
        if qty > 0:
            d["buy_orders"] += 1
        else:
            d["sell_orders"] += 1
        if reason.startswith("COVER") or reason.startswith("RELEASE"):
            d["cover_orders"] += 1
        return True

    def _tier(self, cfg, edge, z):
        ae = abs(edge)
        az = abs(z)
        if ae >= cfg["xabs"] and az >= cfg["xz"]:
            return cfg["extreme"], cfg["xclip"], "X"
        if ae >= cfg["sabs"] and az >= cfg["sz"]:
            return cfg["strong"], cfg["sclip"], "S"
        return cfg["base"], cfg["clip"], "B"

    def apply(self, state, result):
        if self.UNDERLYING not in state.order_depths:
            return
        ve_mid = self._mid_from_depth(state.order_depths[self.UNDERLYING])
        if ve_mid is None:
            return
        for product, cfg in self.PRODUCTS.items():
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            bid, bid_vol = _best_bid_px_vol(od)
            ask, ask_vol = _best_ask_px_vol(od)
            if bid is None or ask is None:
                continue
            spread = ask - bid
            if spread <= 0 or spread > cfg["max_spread"]:
                self.diag["skip_spread"] += 1
                self.diag["last_reason"] = "SPREAD_SKIP"
                continue
            intrinsic = max(0.0, ve_mid - cfg["strike"])
            premium = (bid + ask) / 2.0 - intrinsic
            hist = self.premium_history.setdefault(product, [])
            hist.append(premium)
            if len(hist) > self.HISTORY_LENGTH:
                hist.pop(0)
            d = self.diag
            d["last_product"] = product
            d["last_premium"] = premium
            d["last_bid"] = bid
            d["last_ask"] = ask
            if len(hist) < self.MIN_HISTORY:
                d["skip_history"] += 1
                d["last_reason"] = "WARMUP"
                continue
            vals = hist[:-1]
            mean = sum(vals) / len(vals)
            var = sum((x - mean) * (x - mean) for x in vals) / max(
                1, len(vals) - 1
            )
            std = max(math.sqrt(var), 0.75)
            z = (premium - mean) / std
            edge = premium - mean
            d["last_mean"] = mean
            d["last_std"] = std
            d["last_z"] = z
            pos = self._virtual_position(state, result, product)
            d["last_pos"] = pos
            target, clip, tier = self._tier(cfg, edge, z)
            d["last_target"] = target
            if edge >= cfg["abs"] and z >= cfg["z"]:
                d["rich_signals"] += 1
                qty = min(target + pos, bid_vol, clip)
                if qty > 0:
                    self._append_order(
                        result, product, bid, -qty, "RICH_" + tier
                    )
                    continue
                d["skip_qty"] += 1
                d["last_reason"] = "NO_SELL_QTY"
                continue
            if edge <= -cfg["abs"] and z <= -cfg["z"]:
                d["cheap_signals"] += 1
                qty = min(target - pos, ask_vol, clip)
                if qty > 0:
                    self._append_order(
                        result, product, ask, qty, "CHEAP_" + tier
                    )
                    continue
                d["skip_qty"] += 1
                d["last_reason"] = "NO_BUY_QTY"
                continue
            flat = cfg["flat"]
            if pos < -cfg["base"] // 2 and edge <= flat:
                qty = min(-pos, ask_vol, clip)
                if qty > 0:
                    self._append_order(
                        result, product, ask, qty, "COVER_SHORT"
                    )
                    continue
            if pos > cfg["base"] // 2 and edge >= -flat:
                qty = min(pos, bid_vol, clip)
                if qty > 0:
                    self._append_order(
                        result, product, bid, -qty, "RELEASE_LONG"
                    )
                    continue
            d["last_reason"] = "NO_EDGE"


class VEVSurfaceVerticalOverlay:
    POSITION_LIMIT = 300
    SPREADS = [
        (
            "VEV_5200",
            "VEV_5300",
            47.8172,
            4.7026,
            92,
            122,
            5,
            1.52,
            2.25,
            8,
            5,
        ),
        (
            "VEV_5100",
            "VEV_5200",
            71.8698,
            3.9997,
            80,
            104,
            4,
            1.62,
            2.35,
            9,
            7,
        ),
        ("VEV_5300", "VEV_5400", 28.5465, 5.1413, 70, 94, 4, 1.6, 2.3, 6, 4),
    ]

    def _vpos(self, state, result, p):
        x = state.position.get(p, 0)
        for o in result.get(p, []):
            x += o.quantity
        return x

    def _buy(self, state, result, p, px, qty):
        q = min(
            int(qty),
            max(0, self.POSITION_LIMIT - self._vpos(state, result, p)),
        )
        if q > 0:
            result.setdefault(p, []).append(Order(p, int(px), q))
            return q
        return 0

    def _sell(self, state, result, p, px, qty):
        q = min(
            int(qty),
            max(0, self.POSITION_LIMIT + self._vpos(state, result, p)),
        )
        if q > 0:
            result.setdefault(p, []).append(Order(p, int(px), -q))
            return q
        return 0

    def apply(self, state, result):
        ods = state.order_depths
        for (
            a,
            b,
            mean,
            std,
            base,
            strong,
            clip,
            thr,
            strong_thr,
            maxsa,
            maxsb,
        ) in self.SPREADS:
            if a not in ods or b not in ods:
                continue
            ba, va = _best_bid_px_vol(ods[a])
            aa, ava = _best_ask_px_vol(ods[a])
            bb, vb = _best_bid_px_vol(ods[b])
            ab, avb = _best_ask_px_vol(ods[b])
            if ba is None or aa is None or bb is None or (ab is None):
                continue
            if aa - ba > maxsa or ab - bb > maxsb:
                continue
            z = ((ba + aa) / 2 - (bb + ab) / 2 - mean) / std
            target = strong if abs(z) >= strong_thr else base
            c = clip + 1 if abs(z) >= strong_thr else clip
            if z > thr:
                q = min(
                    max(0, target + self._vpos(state, result, a)),
                    max(0, target - self._vpos(state, result, b)),
                    c,
                    va,
                    avb,
                )
                if q > 0:
                    self._sell(state, result, a, ba, q)
                    self._buy(state, result, b, ab, q)
            elif z < -thr:
                q = min(
                    max(0, target - self._vpos(state, result, a)),
                    max(0, target + self._vpos(state, result, b)),
                    c,
                    ava,
                    vb,
                )
                if q > 0:
                    self._buy(state, result, a, aa, q)
                    self._sell(state, result, b, bb, q)


class Trader:

    def __init__(self):
        self.ve = VelvetfruitStrategy()
        self.hp = HydrogelPackStrategy()
        self.vev = VEVoucherStrategy()
        self.mark14_hp = HPMark14OnlyOverlayV3()
        self.ve_insider = VEInsiderOverlay()
        self.vev4000_insider = VEV4000ShortInsider()
        self.high_strike_short = HighStrikeDecayShort()
        self.vev5300_short = VEV5300DecayShort()
        self.vev_premium = VEVPremiumReversionOverlay()
        self.vev_surface = VEVSurfaceVerticalOverlay()

    def load_state(self, state):
        if not state.traderData:
            return
        try:
            data = json.loads(state.traderData)
            self.ve.load_state(data.get("VELVETFRUIT_EXTRACT", {}))
            self.hp.load_state(data.get("HYDROGEL_PACK", {}))
            self.mark14_hp.load_state(data.get("HP_MARK14_ONLY_V3", {}))
            self.ve_insider.load_state(data.get("VE_INSIDER", {}))
            self.vev4000_insider.load_state(
                data.get("VEV4000_SHORT_INSIDER", {})
            )
            self.high_strike_short.load_state(
                data.get("HIGH_STRIKE_SHORT", {})
            )
            self.vev5300_short.load_state(data.get("VEV5300_DECAY_SHORT", {}))
            self.vev_premium.load_state(data.get("VEV_PREMIUM_REVERSION", {}))
        except Exception:
            self.ve.load_state({})
            self.hp.load_state({})
            self.mark14_hp.load_state({})
            self.ve_insider.load_state({})
            self.vev4000_insider.load_state({})
            self.high_strike_short.load_state({})
            self.vev5300_short.load_state({})
            self.vev_premium.load_state({})

    def save_state(self):
        return json.dumps(
            {
                "VELVETFRUIT_EXTRACT": self.ve.save_state(),
                "HYDROGEL_PACK": self.hp.save_state(),
                "VEVOUCHERS": {},
                "HP_MARK14_ONLY_V3": self.mark14_hp.save_state(),
                "VE_INSIDER": self.ve_insider.save_state(),
                "VEV4000_SHORT_INSIDER": self.vev4000_insider.save_state(),
                "HIGH_STRIKE_SHORT": self.high_strike_short.save_state(),
                "VEV5300_DECAY_SHORT": self.vev5300_short.save_state(),
                "VEV_PREMIUM_REVERSION": self.vev_premium.save_state(),
            }
        )

    def run(self, state: TradingState):
        result = {self.ve.PRODUCT: [], self.hp.PRODUCT: []}
        self.load_state(state)
        if self.ve.PRODUCT in state.order_depths:
            self.ve.position = state.position.get(self.ve.PRODUCT, 0)
            self.ve.trade(state, result[self.ve.PRODUCT])
            self.ve_insider.apply(state, result)
        if self.hp.PRODUCT in state.order_depths:
            self.hp.position = state.position.get(self.hp.PRODUCT, 0)
            self.hp.trade(state, result[self.hp.PRODUCT])
            try:
                hp_fair = self.hp.get_fair()
            except Exception:
                hp_fair = None
            self.mark14_hp.trade(state, result, fair=hp_fair)
        self.vev.DO_DELTA_HEDGE = False
        voucher_result, _, _ = self.vev.run(state)
        for product, orders in voucher_result.items():
            if orders:
                result.setdefault(product, [])
                result[product].extend(orders)
        self.vev4000_insider.apply(state, result)
        self.vev_premium.apply(state, result)
        self.vev_surface.apply(state, result)
        self.vev5300_short.apply(state, result)
        self.high_strike_short.apply(state, result)
        return (result, 0, self.save_state())
