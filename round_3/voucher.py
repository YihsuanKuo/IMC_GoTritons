from datamodel import TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    if x < -8.0: return 0.0
    if x >  8.0: return 1.0
    a1,a2,a3,a4,a5 = 0.319381530,-0.356563782,1.781477937,-1.821255978,1.330274429
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = k*(a1+k*(a2+k*(a3+k*(a4+k*a5))))
    pdf  = math.exp(-0.5*x*x) / math.sqrt(2*math.pi)
    cdf  = 1.0 - pdf*poly
    return cdf if x >= 0 else 1.0 - cdf


class Trader:
    # ==========================================================================
    # Config
    # ==========================================================================
    VOUCHERS = [
        "VEV_4000","VEV_4500","VEV_5000","VEV_5100","VEV_5200",
        "VEV_5300","VEV_5400","VEV_5500","VEV_6000","VEV_6500",
    ]
    STRIKES: Dict[str, float] = {
        "VEV_4000": 4000.0, "VEV_4500": 4500.0, "VEV_5000": 5000.0,
        "VEV_5100": 5100.0, "VEV_5200": 5200.0, "VEV_5300": 5300.0,
        "VEV_5400": 5400.0, "VEV_5500": 5500.0, "VEV_6000": 6000.0,
        "VEV_6500": 6500.0,
    }
    UNDERLYING = "VELVETFRUIT_EXTRACT"

    POSITION_LIMITS: Dict[str, int] = {
        "VELVETFRUIT_EXTRACT": 200,
        **{v: 300 for v in VOUCHERS},
    }

    # ── Pre-fitted vol smile params (quadratic in normalised moneyness m_t) ───
    # m_t = log(S/K) / sqrt(T)
    # IV(m_t) = a * m_t^2 + b * m_t + c
    # Fitted from round_3 historical data (days 0-2), moneyness range [-0.35, 0.35]
    # testing gap = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
    gap = 0.5 
    ASK_PARAMS = {'a': 1/0.095592, 'b': 0.024920, 'c': (0.24+gap/2)}
    BID_PARAMS = {'a': 1/(-0.103375), 'b': (-0.025186), 'c': (0.24-gap/2)}

    # TTE: at live simulation start (round 3), TTE = 5 days
    TTE_START_DAYS = 5.0

    # Max position per voucher for MM sizing
    MAX_SIZE = 80

    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.buy_orders_sent: Dict[str, int] = {}
        self.sell_orders_sent: Dict[str, int] = {}
        self.underlying_price_history: List[float] = []
        self.volcanic_rock_buy_orders = 0
        self.volcanic_rock_sell_orders = 0

    # ==========================================================================
    # Helpers
    # ==========================================================================
    def get_position_limit(self, product: str) -> int:
        return self.POSITION_LIMITS.get(product, 20)

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def send_buy_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        if amount > 0:
            orders.append(Order(product, int(price), int(amount)))

    def send_sell_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        if amount > 0:
            orders.append(Order(product, int(price), -int(amount)))

    def get_best_bid_ask(self, od) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    # ==========================================================================
    # Persistence
    # ==========================================================================
    def load_state(self, state: TradingState) -> None:
        self.underlying_price_history = []
        if not state.traderData:
            return
        try:
            saved = json.loads(state.traderData)
            self.underlying_price_history = saved.get("price_history", [])
        except Exception:
            pass

    def save_state(self) -> str:
        return json.dumps({"price_history": self.underlying_price_history[-250:]})

    # ==========================================================================
    # Black-Scholes (r=0)
    # ==========================================================================
    def bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        return _norm_cdf(d1)

    # ==========================================================================
    # Pre-fitted smile: predict IV for bid or ask side
    # m_t = log(S/K) / sqrt(T)  — same normalised moneyness as big_volcano_man
    # ==========================================================================
    def predict_iv(self, m_t: float, bid: bool = False, ask: bool = False) -> float:
        p = self.BID_PARAMS if bid else self.ASK_PARAMS
        return p['a'] * m_t**2 + p['b'] * m_t + p['c']

    # ==========================================================================
    # Delta hedge underlying
    # ==========================================================================
    def trade_underlying(self, state: TradingState, orders: List[Order], size: int) -> None:
        """Buy (size<0) or sell (size>0) the underlying to offset delta."""
        if self.UNDERLYING not in state.order_depths:
            return
        od = state.order_depths[self.UNDERLYING]
        best_bid, best_ask = self.get_best_bid_ask(od)
        mid = self.underlying_price_history[-1] if self.underlying_price_history else None
        if mid is None:
            return

        limit = self.get_position_limit(self.UNDERLYING)
        cur_pos = self.get_product_pos(state, self.UNDERLYING)

        if size < 0:
            # need to buy
            price = math.ceil(mid)
            qty = min(abs(size), limit - cur_pos - self.volcanic_rock_buy_orders)
            if qty > 0:
                self.volcanic_rock_buy_orders += qty
                self.send_buy_order(orders, self.UNDERLYING, price, qty)
        elif size > 0:
            # need to sell
            price = math.floor(mid)
            qty = min(abs(size), limit + cur_pos - self.volcanic_rock_sell_orders)
            if qty > 0:
                self.volcanic_rock_sell_orders += qty
                self.send_sell_order(orders, self.UNDERLYING, price, qty)

    def delta_hedge(self, state: TradingState, vfe_orders: List[Order],
                    voucher_deltas: Dict[str, float]) -> None:
        """Compute total portfolio delta and hedge with underlying."""
        total_delta = 0.0
        for vev in self.VOUCHERS:
            delta = voucher_deltas.get(vev, 0.0)
            pos = self.get_product_pos(state, vev)
            total_delta += delta * pos

        hedge_size = int(
            self.get_product_pos(state, self.UNDERLYING)
            + self.volcanic_rock_buy_orders
            - self.volcanic_rock_sell_orders
            + total_delta
        )
        self.trade_underlying(state, vfe_orders, hedge_size)

    # ==========================================================================
    # Market-making on IV smile (core logic from big_volcano_man.py)
    # ==========================================================================
    def mm_on_iv(self, state: TradingState, result: Dict[str, List[Order]],
                 S: float, voucher_deltas: Dict[str, float]) -> None:
        vfe_orders = result[self.UNDERLYING]

        for vev in self.VOUCHERS:
            if vev not in state.order_depths:
                continue
            od = state.order_depths[vev]
            best_bid, best_bid_amount = None, None
            best_ask, best_ask_amount = None, None

            if od.sell_orders:
                best_ask, best_ask_amount = min(od.sell_orders.items(), key=lambda x: x[0])
            if od.buy_orders:
                best_bid, best_bid_amount = max(od.buy_orders.items(), key=lambda x: x[0])

            K = self.STRIKES[vev]
            T = max(self.TTE_START_DAYS - state.timestamp / 1_000_000.0, 1e-6) / 365.0

            m_t = math.log(S / K) / math.sqrt(T)

            bid_vol = self.predict_iv(m_t, bid=True)
            ask_vol = self.predict_iv(m_t, ask=True)
            avg_vol = (bid_vol + ask_vol) / 2.0

            # Store delta for this voucher
            voucher_deltas[vev] = self.bs_delta(S, K, T, avg_vol)

            # Convert IV quotes to price quotes
            predicted_bid = self.bs_call(S, K, T, bid_vol)
            predicted_ask = self.bs_call(S, K, T, ask_vol)

            # Ensure ask >= intrinsic value
            intrinsic = max(0.0, S - K)
            predicted_ask = max(predicted_ask, intrinsic)

            bid = math.floor(predicted_bid)
            ask = math.ceil(predicted_ask)

            pos = self.get_product_pos(state, vev)
            bid_size = self.MAX_SIZE - pos
            ask_size = self.MAX_SIZE + pos

            orders = result[vev]
            sent_buys = 0
            sent_sells = 0

            # Eat their market if our quote crosses it
            if best_ask is not None and bid > best_ask:
                eat = min(bid_size, abs(best_ask_amount))
                if eat > 0:
                    self.send_buy_order(orders, vev, best_ask, eat)
                    sent_buys += eat
                    # immediately delta hedge the fill
                    bought_delta = voucher_deltas[vev] * sent_buys
                    self.trade_underlying(state, vfe_orders, -int(bought_delta))
                    # reprice bid just above best bid
                    bid = (best_bid + 1) if best_bid is not None else (best_ask - 1)

            if best_bid is not None and ask < best_bid:
                eat = min(ask_size, abs(best_bid_amount))
                if eat > 0:
                    self.send_sell_order(orders, vev, best_bid, eat)
                    sent_sells += eat
                    # immediately delta hedge the fill
                    sold_delta = voucher_deltas[vev] * sent_sells
                    self.trade_underlying(state, vfe_orders, int(sold_delta))
                    # reprice ask just below best ask
                    ask = (best_ask - 1) if best_ask is not None else (best_bid + 1)

            # Recalculate remaining capacity
            bid_size = max(self.MAX_SIZE - pos - sent_buys, 0)
            ask_size = max(self.MAX_SIZE + pos - sent_sells, 0)

            # Place passive quotes
            if bid == ask:
                if bid_size > ask_size:
                    self.send_buy_order(orders, vev, bid, bid_size)
                else:
                    self.send_sell_order(orders, vev, ask, ask_size)
            else:
                if bid_size > 0:
                    self.send_buy_order(orders, vev, bid, bid_size)
                if ask_size > 0:
                    self.send_sell_order(orders, vev, ask, ask_size)

    # ==========================================================================
    # Main
    # ==========================================================================
    def run(self, state: TradingState):
        all_products = [self.UNDERLYING] + self.VOUCHERS
        result: Dict[str, List[Order]] = {p: [] for p in all_products}

        self.positions = {p: state.position.get(p, 0) for p in all_products}
        self.volcanic_rock_buy_orders = 0
        self.volcanic_rock_sell_orders = 0

        self.load_state(state)

        # Update underlying price history
        if self.UNDERLYING in state.order_depths:
            od = state.order_depths[self.UNDERLYING]
            best_bid, best_ask = self.get_best_bid_ask(od)
            if best_bid is not None and best_ask is not None:
                self.underlying_price_history.append((best_bid + best_ask) / 2.0)

        if not self.underlying_price_history:
            return result, 0, self.save_state()

        S = self.underlying_price_history[-1]
        voucher_deltas: Dict[str, float] = {}

        # 1. Hedge delta from previous positions first
        self.delta_hedge(state, result[self.UNDERLYING], voucher_deltas)

        # 2. Market-make on IV smile
        self.mm_on_iv(state, result, S, voucher_deltas)

        return result, 0, self.save_state()
