from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def __init__(self):
        # EMERALDS: keep simple and anchored
        self.emerald_fair = 10000.0

        # TOMATOES: use fast/slow EMA to detect trend
        self.tom_fast_beta = 0.30
        self.tom_slow_beta = 0.10
        self.tom_trend_coeff = 1.2
        self.tom_trend_threshold = 1.5

        # Trading aggressiveness
        self.tom_take_edge = 2.0
        self.em_take_edge = 2.0

        # Passive quote sizes
        self.em_passive_size = 12
        self.tom_passive_size = 8

    def bid(self):
        return 15

    def get_alpha(self, position: int) -> float:
        abs_pos = abs(position)
        if abs_pos < 10:
            return 0.02
        elif abs_pos < 30:
            return 0.05
        else:
            return 0.10

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # Load saved state
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                saved = {}
        else:
            saved = {}

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

            max_buy = limit - current_position
            max_sell = limit + current_position

            alpha = self.get_alpha(current_position)

            # ==================== EMERALDS ====================
            if product == "EMERALDS":
                # Fixed fair value around 10000 with inventory penalty
                fair_price = self.emerald_fair - alpha * current_position

                # Aggressive taking only if clearly favorable
                if best_ask <= fair_price - self.em_take_edge:
                    buy_volume = min(-best_ask_volume, max_buy)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if best_bid >= fair_price + self.em_take_edge:
                    sell_volume = min(best_bid_volume, max_sell)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # Passive MM around fair value
                if spread >= 2:
                    passive_buy = min(best_bid + 1, int(fair_price))
                    passive_sell = max(best_ask - 1, int(fair_price))
                else:
                    passive_buy = min(best_bid, int(fair_price))
                    passive_sell = max(best_ask, int(fair_price))

                if max_buy > 0:
                    orders.append(
                        Order(
                            product,
                            passive_buy,
                            min(self.em_passive_size, max_buy),
                        )
                    )

                if max_sell > 0:
                    orders.append(
                        Order(
                            product,
                            passive_sell,
                            -min(self.em_passive_size, max_sell),
                        )
                    )

                new_data["EMERALDS"] = {"last_mid": mid_price}

            # ==================== TOMATOES ====================
            elif product == "TOMATOES":
                prev_fast = saved.get("TOMATOES", {}).get("fast", mid_price)
                prev_slow = saved.get("TOMATOES", {}).get("slow", mid_price)

                fast = (
                    self.tom_fast_beta * mid_price
                    + (1 - self.tom_fast_beta) * prev_fast
                )
                slow = (
                    self.tom_slow_beta * mid_price
                    + (1 - self.tom_slow_beta) * prev_slow
                )
                trend = fast - slow

                # Fair price includes trend signal and inventory penalty
                fair_price = (
                    mid_price
                    + self.tom_trend_coeff * trend
                    - alpha * current_position
                )

                bullish = trend > self.tom_trend_threshold
                bearish = trend < -self.tom_trend_threshold
                neutral = not bullish and not bearish

                # Aggressive taking only when edge is meaningful
                if not bearish and best_ask <= fair_price - self.tom_take_edge:
                    buy_volume = min(-best_ask_volume, max_buy)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))

                if not bullish and best_bid >= fair_price + self.tom_take_edge:
                    sell_volume = min(best_bid_volume, max_sell)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))

                # Passive quoting logic by regime
                if neutral:
                    passive_buy = min(best_bid + 1, int(fair_price))
                    passive_sell = max(best_ask - 1, int(fair_price))

                elif bullish:
                    # In uptrend: still bid, but do not sell too aggressively
                    passive_buy = min(best_bid + 1, int(fair_price))
                    passive_sell = max(best_ask, int(fair_price) + 1)

                else:  # bearish
                    # In downtrend: still ask, but do not buy too aggressively
                    passive_buy = min(best_bid, int(fair_price) - 1)
                    passive_sell = max(best_ask - 1, int(fair_price))

                # Optional one-sided bias in strong trends
                if bullish:
                    if max_buy > 0:
                        orders.append(
                            Order(
                                product,
                                passive_buy,
                                min(self.tom_passive_size, max_buy),
                            )
                        )
                    if max_sell > 0 and current_position > 0:
                        orders.append(
                            Order(
                                product,
                                passive_sell,
                                -min(self.tom_passive_size // 2, max_sell),
                            )
                        )

                elif bearish:
                    if max_sell > 0:
                        orders.append(
                            Order(
                                product,
                                passive_sell,
                                -min(self.tom_passive_size, max_sell),
                            )
                        )
                    if max_buy > 0 and current_position < 0:
                        orders.append(
                            Order(
                                product,
                                passive_buy,
                                min(self.tom_passive_size // 2, max_buy),
                            )
                        )

                else:
                    if max_buy > 0:
                        orders.append(
                            Order(
                                product,
                                passive_buy,
                                min(self.tom_passive_size, max_buy),
                            )
                        )
                    if max_sell > 0:
                        orders.append(
                            Order(
                                product,
                                passive_sell,
                                -min(self.tom_passive_size, max_sell),
                            )
                        )

                new_data["TOMATOES"] = {
                    "fast": fast,
                    "slow": slow,
                    "last_mid": mid_price,
                    "trend": trend,
                }

            result[product] = orders

        traderData = json.dumps(new_data)
        conversions = 0
        return result, conversions, traderData
