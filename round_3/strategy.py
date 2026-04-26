from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math


class Trader:
    UNDERLYING = "VELVETFRUIT_EXTRACT"

    TRADE_VOUCHERS = [
        "VEV_4500",
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
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

    # Extremely aggressive
    MAX_TAKE_SIZE = 60
    MAX_MAKE_SIZE = 30
    PRICE_EDGE = 0.2
    MAX_MARKET_SPREAD = 20
    ENABLE_PASSIVE_QUOTES = True

    LOTTERY_SELL_ONLY = {"VEV_6000", "VEV_6500"}
    LOTTERY_SELL_SIZE = 25

    DEEP_ITM = {"VEV_4500"}
    DEEP_ITM_SIZE = 20
    DEEP_ITM_PREMIUM = {"VEV_4500": 8}
    DEEP_ITM_EDGE = {"VEV_4500": 3}

    DO_DELTA_HEDGE = False

    def run(self, state):
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
            try:
                orders = self.trade_vev_voucher(state, voucher, S, T)
                if orders:
                    result[voucher] = orders
            except Exception:
                continue

        return result, 0, self.safe_trader_data(state)

    def trade_vev_voucher(self, state, product, S, T):
        orders: List[Order] = []
        od = state.order_depths[product]

        best_bid = self.get_best_bid(od)
        best_ask = self.get_best_ask(od)

        if best_bid is None or best_ask is None or best_bid <= 0:
            return orders

        market_spread = best_ask - best_bid
        if market_spread <= 0 or market_spread > self.MAX_MARKET_SPREAD:
            return orders

        K = self.get_strike(product)
        pos = self.get_position(state, product)
        limit = self.get_position_limit(product)

        # Aggressive lottery selling
        if product in self.LOTTERY_SELL_ONLY:
            if best_bid >= 1:
                qty = min(
                    od.buy_orders[best_bid],
                    limit + pos,
                    self.LOTTERY_SELL_SIZE,
                )
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
            return orders

        # Aggressive deep ITM value trade
        if product in self.DEEP_ITM:
            intrinsic = max(0.0, S - K)
            fair = intrinsic + self.DEEP_ITM_PREMIUM[product]

            if best_ask < fair - self.DEEP_ITM_EDGE:
                qty = min(
                    -od.sell_orders[best_ask],
                    limit - pos,
                    self.DEEP_ITM_SIZE,
                )
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                return orders

            if best_bid > fair + self.DEEP_ITM_EDGE:
                qty = min(
                    od.buy_orders[best_bid],
                    limit + pos,
                    self.DEEP_ITM_SIZE,
                )
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                return orders

            return orders

        mid_price = (best_bid + best_ask) / 2.0
        if mid_price < 2:
            return orders

        bid_iv = self.get_curve_iv(S, K, T, "bid")
        ask_iv = self.get_curve_iv(S, K, T, "ask")
        mid_iv = self.get_curve_iv(S, K, T, "mid")

        predicted_bid = self.bs_call_price(S, K, T, bid_iv)
        predicted_ask = self.bs_call_price(S, K, T, ask_iv)
        predicted_mid = self.bs_call_price(S, K, T, mid_iv)

        # Aggressive take: buy cheap
        if predicted_bid > best_ask + self.PRICE_EDGE:
            qty = min(
                -od.sell_orders[best_ask],
                limit - pos,
                self.MAX_TAKE_SIZE,
            )
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
            return orders

        # Aggressive take: sell rich
        if predicted_ask < best_bid - self.PRICE_EDGE:
            qty = min(
                od.buy_orders[best_bid],
                limit + pos,
                self.MAX_TAKE_SIZE,
            )
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
            return orders

        # Passive quote aggressively around model
        if self.ENABLE_PASSIVE_QUOTES:
            our_bid = math.floor(predicted_bid)
            our_ask = math.ceil(predicted_ask)

            if our_bid >= our_ask:
                our_bid = math.floor(predicted_mid) - 1
                our_ask = math.ceil(predicted_mid) + 1

            intrinsic = max(0.0, S - K)
            our_bid = max(0, our_bid)
            our_ask = max(math.ceil(intrinsic), our_ask)

            quote_bid = min(our_bid, best_bid + 1, best_ask - 1)
            quote_ask = max(our_ask, best_ask - 1, best_bid + 1)

            if quote_bid > 0 and quote_ask > 0 and quote_bid < quote_ask:
                buy_qty = min(limit - pos, self.MAX_MAKE_SIZE)
                sell_qty = min(limit + pos, self.MAX_MAKE_SIZE)

                if buy_qty > 0:
                    orders.append(Order(product, quote_bid, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, quote_ask, -sell_qty))

        return orders

    def get_curve_iv(self, S, K, T, side):
        if side == "bid":
            a, b, c, d = self.A_BID, self.B_BID, self.C_BID, self.D_BID
        elif side == "ask":
            a, b, c, d = self.A_ASK, self.B_ASK, self.C_ASK, self.D_ASK
        else:
            a, b, c, d = self.A_MID, self.B_MID, self.C_MID, self.D_MID

        if S <= 0 or K <= 0 or T <= 0:
            return d

        m = math.log(K / S) / math.sqrt(T)
        iv = a * m**3 + b * m**2 + c * m + d
        return min(max(iv, 0.01), 3.0)

    def bs_call_price(self, S, K, T, sigma, r=0.0):
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)

        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (
            sigma * sqrt_t
        )
        d2 = d1 - sigma * sqrt_t

        return S * self.normal_cdf(d1) - K * math.exp(
            -r * T
        ) * self.normal_cdf(d2)

    def normal_cdf(self, x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def get_time_to_expiry(self, state):
        day_fraction_passed = state.timestamp / 1_000_000.0
        days_left = self.STARTING_DAYS_TO_EXPIRY - day_fraction_passed
        days_left = max(0.05, days_left)
        return days_left / self.DAYS_PER_YEAR

    def get_strike(self, product):
        return int(product.split("_")[-1])

    def get_position(self, state, product):
        return state.position.get(product, 0)

    def get_position_limit(self, product):
        return self.POSITION_LIMITS.get(product, 100)

    def get_best_bid(self, od):
        return max(od.buy_orders.keys()) if od.buy_orders else None

    def get_best_ask(self, od):
        return min(od.sell_orders.keys()) if od.sell_orders else None

    def get_mid_price(self, state, product):
        if product not in state.order_depths:
            return None

        od = state.order_depths[product]
        bid = self.get_best_bid(od)
        ask = self.get_best_ask(od)

        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        return None

    def safe_trader_data(self, state):
        data = getattr(state, "traderData", "")
        return "" if data is None else data
