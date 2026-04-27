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

    # -------- VEV fitted bid/ask/mid IV curve parameters --------
    # Fit used m = log(K / S) / sqrt(T)
    A_BID = 0.115769
    B_BID = -0.038140
    C_BID = 0.249095

    A_ASK = 0.145857
    B_ASK = -0.128206
    C_ASK = 0.272402

    A_MID = 0.121318
    B_MID = -0.025456
    C_MID = 0.257274


    DAYS_PER_YEAR = 365.0

    # For day 2 backtest use 5.0.
    # If testing day 0, try 7.0.
    # If testing day 1, try 6.0.
    STARTING_DAYS_TO_EXPIRY = 5.0

    # Conservative sizes first.
    MAX_TAKE_SIZE = 6
    MAX_MAKE_SIZE = 8

    # Require at least this much price edge before crossing spread.
    PRICE_EDGE = 0.7

    # Skip markets that are too wide.
    MAX_MARKET_SPREAD = 12

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
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
        "VELVETFRUIT_EXTRACT": 200,
    }

    DO_DELTA_HEDGE = True
    HEDGE_RATIO = 0.5
    DELTA_HEDGE_THRESHOLD = 80
    MAX_HEDGE_SIZE = 5
    MAX_EXTRACT_SPREAD = 4

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

        if self.DO_DELTA_HEDGE:
            hedge_orders = self.delta_hedge_extract(state, result, S, T)
            if hedge_orders:
                result[self.UNDERLYING] = hedge_orders

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
        if position > 67:
            buy_size = 0
            sell_size = min(max_sell, self.MAX_MAKE_SIZE + 4)
        elif position < -67:
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
            a, b, c = self.A_BID, self.B_BID, self.C_BID
        elif side == "ask":
            a, b, c = self.A_ASK, self.B_ASK, self.C_ASK
        else:
            a, b, c = self.A_MID, self.B_MID, self.C_MID

        if S <= 0 or K <= 0 or T <= 0:
            return c

        # IMPORTANT: fitted with log(K / S), not log(S / K).
        m = math.log(K / S) / math.sqrt(T)
        iv = a * m * m + b * m + c

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

    def bs_call_delta(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float = 0.0,
    ) -> float:
        if T <= 0:
            return 1.0 if S > K else 0.0

        if sigma <= 0:
            return 1.0 if S > K else 0.0

        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        return self.normal_cdf(d1)

    def normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # ============================================================
    # Gentle delta hedge
    # ============================================================

    def delta_hedge_extract(
        self,
        state: TradingState,
        result: Dict[str, List[Order]],
        S: float,
        T: float,
    ) -> List[Order]:
        orders: List[Order] = []

        if self.UNDERLYING not in state.order_depths:
            return orders

        order_depth = state.order_depths[self.UNDERLYING]
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)

        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        if spread <= 0 or spread > self.MAX_EXTRACT_SPREAD:
            return orders

        total_option_delta = self.get_total_option_delta_after_orders(state, result, S, T)
        target_hedge = -self.HEDGE_RATIO * total_option_delta
        current_extract_position = self.get_position(state, self.UNDERLYING)
        hedge_needed = target_hedge - current_extract_position

        if abs(hedge_needed) < self.DELTA_HEDGE_THRESHOLD:
            return orders

        limit = self.get_position_limit(self.UNDERLYING)

        if hedge_needed > 0:
            ask_volume = -order_depth.sell_orders[best_ask]
            max_can_buy = limit - current_extract_position
            qty = min(int(round(hedge_needed)), ask_volume, max_can_buy, self.MAX_HEDGE_SIZE)
            if qty > 0:
                orders.append(Order(self.UNDERLYING, best_ask, qty))

        elif hedge_needed < 0:
            bid_volume = order_depth.buy_orders[best_bid]
            max_can_sell = limit + current_extract_position
            qty = min(int(round(-hedge_needed)), bid_volume, max_can_sell, self.MAX_HEDGE_SIZE)
            if qty > 0:
                orders.append(Order(self.UNDERLYING, best_bid, -qty))

        return orders

    def get_total_option_delta_after_orders(
        self,
        state: TradingState,
        result: Dict[str, List[Order]],
        S: float,
        T: float,
    ) -> float:
        total_delta = 0.0

        for voucher in self.TRADE_VOUCHERS:
            K = self.get_strike(voucher)
            mid_iv = self.get_curve_iv(S, K, T, side="mid")
            delta = self.bs_call_delta(S, K, T, mid_iv)

            projected_position = self.get_position(state, voucher)
            for order in result.get(voucher, []):
                projected_position += order.quantity

            total_delta += projected_position * delta

        return total_delta

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
