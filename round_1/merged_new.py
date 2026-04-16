from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


class Trader:
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    # -------- INTARIAN_PEPPER_ROOT params --------
    LINEAR_SLOPE = 0.001

    # -------- ASH_COATED_OSMIUM params --------
    ASH_PRODUCT = "ASH_COATED_OSMIUM"
    ASH_HISTORY_LENGTH = 3
    ASH_ENTRY_Z = 1.0
    ASH_EXIT_Z = 0.4
    ASH_BASE_QUOTE_OFFSET = 1
    ASH_INVENTORY_SKEW = 0.02

    def __init__(self, lam=[0.6, 0.6], alpha=[None, None]):
        # INTARIAN_PEPPER_ROOT parameters/state
        self.er_lam = lam[0]
        self.tom_lam = lam[1]
        self.er_alpha = alpha[0]
        self.tom_alpha = alpha[1]

        # ASH_COATED_OSMIUM state
        self.position = 0
        self.buy_orders_sent = 0
        self.sell_orders_sent = 0
        self.mid_history = []

    # ==================== Shared / PEPPER helpers ====================
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

    # ==================== ASH helpers ====================
    def send_sell_order(self, orders, product, price, amount):
        orders.append(Order(product, int(price), int(amount)))

    def send_buy_order(self, orders, product, price, amount):
        orders.append(Order(product, int(price), int(amount)))

    def get_product_pos(self, state, product):
        return state.position.get(product, 0)

    def load_saved_data(self, state):
        if not state.traderData:
            return {}
        try:
            saved = json.loads(state.traderData)
            if isinstance(saved, dict):
                return saved
            return {}
        except Exception:
            return {}

    def load_history(self, saved_data):
        ash_data = saved_data.get(self.ASH_PRODUCT, {})
        self.mid_history = ash_data.get("mid_history", []) if isinstance(ash_data, dict) else []

    def save_data(self, pepper_mid_prices):
        return json.dumps(
            {
                "INTARIAN_PEPPER_ROOT": pepper_mid_prices.get("INTARIAN_PEPPER_ROOT"),
                self.ASH_PRODUCT: {
                    "mid_history": self.mid_history[-self.ASH_HISTORY_LENGTH :]
                },
            }
        )

    def get_mid_price(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def update_mid_history(self, mid):
        self.mid_history.append(mid)
        if len(self.mid_history) > self.ASH_HISTORY_LENGTH:
            self.mid_history.pop(0)

    def get_mean_std(self):
        if not self.mid_history:
            return None, None
        mean = sum(self.mid_history) / len(self.mid_history)
        var = sum((x - mean) ** 2 for x in self.mid_history) / len(self.mid_history)
        return mean, math.sqrt(var)

    def get_zscore(self, mid):
        mean, std = self.get_mean_std()
        if mean is None or std < 1e-6:
            return 0, mid
        return (mid - mean) / std, mean

    def trade_ash_coated_osmium(self, state, orders):
        od = state.order_depths[self.ASH_PRODUCT]
        mid = self.get_mid_price(od)
        if mid is None:
            return

        self.update_mid_history(mid)
        z, mean = self.get_zscore(mid)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        # mean reversion
        if z < -self.ASH_ENTRY_Z:
            for ask, vol in sorted(od.sell_orders.items()):
                if ask <= mean:
                    size = min(-vol, self.POSITION_LIMITS[self.ASH_PRODUCT])
                    self.send_buy_order(orders, self.ASH_PRODUCT, ask, size)

        elif z > self.ASH_ENTRY_Z:
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid >= mean:
                    size = min(vol, self.POSITION_LIMITS[self.ASH_PRODUCT])
                    self.send_sell_order(orders, self.ASH_PRODUCT, bid, -size)

        # market making
        fair = mean 
        bid_price = min(best_bid + 1, int(fair - self.ASH_BASE_QUOTE_OFFSET))
        ask_price = max(best_ask - 1, int(fair + self.ASH_BASE_QUOTE_OFFSET))

        self.send_buy_order(orders, self.ASH_PRODUCT, bid_price, 10)
        self.send_sell_order(orders, self.ASH_PRODUCT, ask_price, -10)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        pepper_mid_prices = {}

        saved_data = self.load_saved_data(state)
        self.load_history(saved_data)

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
            pepper_mid_prices[product] = mid_price

            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            max_buy = limit - current_position

            # ---------------- INTARIAN_PEPPER_ROOT ----------------
            if product == "INTARIAN_PEPPER_ROOT":
                if current_position < limit and max_buy > 0:
                    remaining_to_buy = max_buy

                    # 1. AGGRESSIVE BUYING (with a price cap)
                    # Only buy asks that are at or just 1 tick above the best_ask
                    acceptable_ask = best_ask + 1

                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if remaining_to_buy <= 0:
                            break
                        # Stop buying if the order book gets too expensive
                        if ask_price > acceptable_ask:
                            break

                        vol = min(remaining_to_buy, -order_depth.sell_orders[ask_price])
                        orders.append(Order(product, ask_price, vol))
                        remaining_to_buy -= vol

                    # 2. PASSIVE BUYING
                    # If we haven't reached our limit of 80, place a passive bid
                    # to try and get filled at a cheaper price
                    if remaining_to_buy > 0:
                        # Place a bid 1 tick below the best ask (or at the best bid)
                        passive_bid_price = best_ask - 1
                        orders.append(Order(product, passive_bid_price, remaining_to_buy))

            # ---------------- ASH_COATED_OSMIUM ----------------
            elif product == self.ASH_PRODUCT:
                self.position = self.get_product_pos(state, self.ASH_PRODUCT)
                self.buy_orders_sent = 0
                self.sell_orders_sent = 0
                self.trade_ash_coated_osmium(state, orders)

            result[product] = orders

        traderData = self.save_data(pepper_mid_prices)
        conversions = 0
        return result, conversions, traderData
