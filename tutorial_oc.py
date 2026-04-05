from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    HISTORY_LEN = 6
    EMERALDS_ANCHOR = 10000

    def bid(self):
        return 15

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)
        if abs_pos < 10:
            return 0.02
        elif abs_pos < 30:
            return 0.05
        else:
            return 0.1

    def update_history(self, saved_data, product, mid_price):
        if "history" not in saved_data:
            saved_data["history"] = {}

        if product not in saved_data["history"]:
            saved_data["history"][product] = []

        hist = saved_data["history"][product]
        hist.append(mid_price)

        if len(hist) > self.HISTORY_LEN:
            hist = hist[-self.HISTORY_LEN:]

        saved_data["history"][product] = hist
        return hist

    def calc_momentum(self, hist, lookback):
        if len(hist) <= lookback:
            return 0.0
        return hist[-1] - hist[-1 - lookback]

    def calc_volatility(self, hist):
        if len(hist) < 2:
            return 0.0

        diffs = []
        for i in range(1, len(hist)):
            diffs.append(abs(hist[i] - hist[i - 1]))

        if len(diffs) == 0:
            return 0.0

        return sum(diffs) / len(diffs)

    def calc_imbalance(self, best_bid_volume, best_ask_volume):
        ask_size = -best_ask_volume
        bid_size = best_bid_volume
        denom = bid_size + ask_size

        if denom <= 0:
            return 0.0

        return (bid_size - ask_size) / denom

    def calc_microprice(
        self, best_bid, best_ask, best_bid_volume, best_ask_volume
    ):
        ask_size = -best_ask_volume
        bid_size = best_bid_volume
        denom = bid_size + ask_size

        if denom <= 0:
            return (best_bid + best_ask) / 2

        return (best_ask * bid_size + best_bid * ask_size) / denom

    def clamp(self, x, low, high):
        return max(low, min(high, x))

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if state.traderData:
            try:
                saved_data = json.loads(state.traderData)
            except Exception:
                saved_data = {}
        else:
            saved_data = {}

        new_data = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            if (
                len(order_depth.buy_orders) == 0
                or len(order_depth.sell_orders) == 0
            ):
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]

            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]

            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            alpha = self.get_alpha(current_position)

            max_buy = limit - current_position
            max_sell = limit + current_position

            # ---------------- EMERALDS ----------------
            if product == "EMERALDS":
                prev_mid = saved_data.get("EMERALDS", mid_price)

                hist = self.update_history(saved_data, product, mid_price)
                vol = self.calc_volatility(hist)

                microprice = self.calc_microprice(
                    best_bid, best_ask, best_bid_volume, best_ask_volume
                )

                signal = microprice - mid_price
                clipped_signal = self.clamp(signal, -1.0, 1.0)

                # 稳定 fair：以 anchor + 平滑 mid + inventory 为主
                fair_price = (
                    0.50 * self.EMERALDS_ANCHOR
                    + 0.30 * prev_mid
                    + 0.20 * mid_price
                    - alpha * current_position
                )

                # 保存数据
                new_data[product] = mid_price
                new_data["history"] = saved_data.get("history", {})

                # ---------- 1) anchor 吃单 ----------
                # 只有在不太偏仓时才更积极吃单
                if max_buy > 0 and current_position < 60:
                    if best_ask <= self.EMERALDS_ANCHOR - 1:
                        take_buy_size = min(-best_ask_volume, max_buy, 16)
                        if take_buy_size > 0:
                            orders.append(Order(product, best_ask, take_buy_size))
                            max_buy -= take_buy_size

                if max_sell > 0 and current_position > -60:
                    if best_bid >= self.EMERALDS_ANCHOR + 1:
                        take_sell_size = min(best_bid_volume, max_sell, 16)
                        if take_sell_size > 0:
                            orders.append(Order(product, best_bid, -take_sell_size))
                            max_sell -= take_sell_size

                # ---------- 2) 报价斜偏化 ----------
                # 基础报价中心仍围绕 fair
                base_buy_quote = min(best_bid + 1, int(fair_price))
                base_sell_quote = max(best_ask - 1, int(fair_price))

                # signal 阈值
                skew_threshold = 0.20

                buy_quote = base_buy_quote
                sell_quote = base_sell_quote

                # 买盘更强 -> 买单更激进，卖单更保守
                if clipped_signal > skew_threshold:
                    buy_quote = min(best_ask - 1, max(base_buy_quote, best_bid + 1))
                    sell_quote = max(base_sell_quote, best_ask)

                # 卖盘更强 -> 卖单更激进，买单更保守
                elif clipped_signal < -skew_threshold:
                    buy_quote = min(base_buy_quote, best_bid)
                    sell_quote = max(best_bid + 1, min(base_sell_quote, best_ask - 1))

                # 中性 -> 正常双边挂
                else:
                    buy_quote = base_buy_quote
                    sell_quote = base_sell_quote

                # spread 太小就别过度 skew
                if spread <= 1:
                    buy_quote = best_bid
                    sell_quote = best_ask

                # 防止反向交叉
                if buy_quote >= sell_quote:
                    buy_quote = best_bid
                    sell_quote = best_ask

                # ---------- 3) inventory skew ----------
                buy_size = min(15, max_buy)
                sell_size = min(15, max_sell)

                if current_position > 45:
                    buy_size = min(5, max_buy)
                    sell_size = min(20, max_sell)
                    # 偏多时卖得更积极一点
                    sell_quote = max(best_bid + 1, min(sell_quote, best_ask - 1))

                elif current_position < -45:
                    buy_size = min(20, max_buy)
                    sell_size = min(5, max_sell)
                    # 偏空时买得更积极一点
                    buy_quote = min(best_ask - 1, max(buy_quote, best_bid + 1))

                if buy_size > 0:
                    orders.append(Order(product, buy_quote, buy_size))

                if sell_size > 0:
                    orders.append(Order(product, sell_quote, -sell_size))

            # ---------------- TOMATOES ----------------
            elif product == "TOMATOES":
                prev_mid = saved_data.get("TOMATOES", mid_price)

                fair_price = (
                    0.4 * prev_mid + 0.6 * mid_price - alpha * current_position
                )

                hist = self.update_history(saved_data, product, mid_price)
                mom5 = self.calc_momentum(hist, 5)
                vol = self.calc_volatility(hist)
                imbalance = self.calc_imbalance(best_bid_volume, best_ask_volume)

                new_data[product] = mid_price
                new_data["history"] = saved_data.get("history", {})

                # regime switching
                theta_1 = 3.0
                theta_2 = 1.2
                theta_3 = 0.35

                if abs(mom5) < theta_1 and vol < theta_2:
                    regime = "A"
                elif abs(mom5) >= theta_1 or abs(imbalance) >= theta_3:
                    regime = "B"
                else:
                    regime = "A"

                # Regime A: 双边挂单，少吃单
                if regime == "A":
                    edge = 1

                    if best_ask <= fair_price - edge:
                        buy_volume = min(-best_ask_volume, max_buy, 4)
                        if buy_volume > 0:
                            orders.append(Order(product, best_ask, buy_volume))

                    if best_bid >= fair_price + edge:
                        sell_volume = min(best_bid_volume, max_sell, 4)
                        if sell_volume > 0:
                            orders.append(Order(product, best_bid, -sell_volume))

                    passive_buy = min(best_bid + 1, int(fair_price))
                    passive_sell = max(best_ask - 1, int(fair_price))

                    if passive_buy >= passive_sell:
                        passive_buy = best_bid
                        passive_sell = best_ask

                    if max_buy > 0:
                        orders.append(
                            Order(product, passive_buy, min(10, max_buy))
                        )

                    if max_sell > 0:
                        orders.append(
                            Order(product, passive_sell, -min(10, max_sell))
                        )

                # Regime B: 减小被动挂单规模，提高主动单权重
                else:
                    edge = 0

                    if best_ask <= fair_price - edge:
                        buy_volume = min(-best_ask_volume, max_buy, 16)
                        if buy_volume > 0:
                            orders.append(Order(product, best_ask, buy_volume))

                    if best_bid >= fair_price + edge:
                        sell_volume = min(best_bid_volume, max_sell, 16)
                        if sell_volume > 0:
                            orders.append(Order(product, best_bid, -sell_volume))

                    passive_buy = min(best_bid + 1, int(fair_price))
                    passive_sell = max(best_ask - 1, int(fair_price))

                    if passive_buy >= passive_sell:
                        passive_buy = best_bid
                        passive_sell = best_ask

                    if max_buy > 0:
                        orders.append(
                            Order(product, passive_buy, min(4, max_buy))
                        )

                    if max_sell > 0:
                        orders.append(
                            Order(product, passive_sell, -min(4, max_sell))
                        )

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData