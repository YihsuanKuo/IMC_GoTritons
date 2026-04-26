from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


# ---------------------------------------------------------------------------
# Normal distribution helpers (no scipy available in competition)
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    """Approximation of standard normal CDF (Abramowitz & Stegun)."""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    cdf = 1.0 - pdf * poly
    return cdf if x >= 0 else 1.0 - cdf


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class Trader:
    # ==========================================================================
    # Product config
    # ==========================================================================
    VOUCHERS = [
        "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
        "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
    ]
    STRIKES: Dict[str, float] = {
        "VEV_4000": 4000.0, "VEV_4500": 4500.0, "VEV_5000": 5000.0,
        "VEV_5100": 5100.0, "VEV_5200": 5200.0, "VEV_5300": 5300.0,
        "VEV_5400": 5400.0, "VEV_5500": 5500.0, "VEV_6000": 6000.0,
        "VEV_6500": 6500.0,
    }
    PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"] + VOUCHERS

    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        **{v: 300 for v in VOUCHERS},
    }

    # ==========================================================================
    # Delta-1 (VFE & HGP) params
    # ==========================================================================
    VFE_HISTORY_LENGTH = 5
    VFE_ENTRY_Z = 0.8
    VFE_EXIT_Z = 0.5
    VFE_QUOTE_OFFSET = 1
    VFE_PASSIVE_SIZE = 30
    VFE_MAX_TAKE = 30
    VFE_SKEW = 1.5

    HGP_HISTORY_LENGTH = 5
    HGP_ENTRY_Z = 0.8
    HGP_EXIT_Z = 0.5
    HGP_QUOTE_OFFSET = 1
    HGP_PASSIVE_SIZE = 30
    HGP_MAX_TAKE = 30
    HGP_SKEW = 1.5

    # ==========================================================================
    # Options params
    # ==========================================================================
    INIT_SIGMA = 0.16        # initial historical vol before EWMA warms up
    EWMA_LAMBDA = 0.94       # RiskMetrics decay factor
    MIN_EDGE = 1.0           # min BS mispricing to trade aggressively
    MAX_TAKE_PER_TICK = 20   # max aggressive size per voucher per tick
    PASSIVE_QUOTE_SIZE = 5   # passive quote size for vouchers

    # TTE: at round 3 live simulation start, TTE = 5 days.
    # Timestamps run 0 → ~999_900 within a single day (1_000_000 units = 1 day).
    TTE_START_DAYS = 5.0

    def __init__(self):
        # per-tick runtime state
        self.positions: Dict[str, int] = {}
        self.buy_orders_sent: Dict[str, int] = {}
        self.sell_orders_sent: Dict[str, int] = {}

        # persistent state
        self.vfe_mid_history: List[float] = []
        self.hgp_mid_history: List[float] = []
        self.hist_vol: float = self.INIT_SIGMA

    # ==========================================================================
    # Generic helpers (adapted from round_2/strategy.py)
    # ==========================================================================
    def get_position_limit(self, product: str) -> int:
        return self.POSITION_LIMITS.get(product, 20)

    def get_product_pos(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def remaining_buy_capacity(self, product: str) -> int:
        limit = self.get_position_limit(product)
        pos = self.positions.get(product, 0)
        sent = self.buy_orders_sent.get(product, 0)
        return max(0, limit - pos - sent)

    def remaining_sell_capacity(self, product: str) -> int:
        limit = self.get_position_limit(product)
        pos = self.positions.get(product, 0)
        sent = self.sell_orders_sent.get(product, 0)
        return max(0, limit + pos - sent)

    def send_buy_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        size = min(amount, self.remaining_buy_capacity(product))
        if size > 0:
            orders.append(Order(product, int(price), int(size)))
            self.buy_orders_sent[product] = self.buy_orders_sent.get(product, 0) + int(size)

    def send_sell_order(self, orders: List[Order], product: str, price: int, amount: int) -> None:
        size = min(amount, self.remaining_sell_capacity(product))
        if size > 0:
            orders.append(Order(product, int(price), -int(size)))
            self.sell_orders_sent[product] = self.sell_orders_sent.get(product, 0) + int(size)

    def get_best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    # ==========================================================================
    # Persistence
    # ==========================================================================
    def load_state(self, state: TradingState) -> None:
        self.vfe_mid_history = []
        self.hgp_mid_history = []
        self.hist_vol = self.INIT_SIGMA

        if not state.traderData:
            return
        try:
            saved = json.loads(state.traderData)
            self.vfe_mid_history = saved.get("vfe_mid_history", [])
            self.hgp_mid_history = saved.get("hgp_mid_history", [])
            self.hist_vol = saved.get("hist_vol", self.INIT_SIGMA)
        except Exception:
            pass

    def save_state(self) -> str:
        return json.dumps({
            "vfe_mid_history": self.vfe_mid_history[-self.VFE_HISTORY_LENGTH:],
            "hgp_mid_history": self.hgp_mid_history[-self.HGP_HISTORY_LENGTH:],
            "hist_vol": self.hist_vol,
        })

    # ==========================================================================
    # Black-Scholes (risk-free rate = 0)
    # ==========================================================================
    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """European call price under Black-Scholes with r=0."""
        if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Call delta = N(d1)."""
        if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        return _norm_cdf(d1)

    def bs_implied_vol(self, mkt_price: float, S: float, K: float, T: float) -> Optional[float]:
        """Newton-Raphson implied vol solver. Returns None if it fails to converge."""
        if mkt_price <= 0.0 or T <= 0.0 or S <= 0.0:
            return None
        intrinsic = max(0.0, S - K)
        if mkt_price <= intrinsic:
            return None

        sigma = self.hist_vol  # initial guess
        for _ in range(20):
            price = self.bs_call_price(S, K, T, sigma)
            if T <= 0.0 or sigma <= 0.0:
                break
            d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
            vega = S * math.sqrt(T) * _norm_pdf(d1)
            if vega < 1e-8:
                break
            diff = price - mkt_price
            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5.0))
            if abs(diff) < 1e-5:
                return sigma
        return sigma

    # ==========================================================================
    # Volatility smile calibration
    # ==========================================================================
    def calibrate_vol_smile(
        self, state: TradingState, S: float, T: float
    ) -> Dict[str, float]:
        """
        Compute implied vol for each liquid voucher, then fit a quadratic
        in moneyness m = ln(S/K) to get smile-smoothed IVs for all strikes.
        Falls back to hist_vol for strikes where no liquid market exists.
        """
        ivs: List[Tuple[float, float]] = []  # (moneyness, iv)

        for vev in self.VOUCHERS:
            if vev not in state.order_depths:
                continue
            od = state.order_depths[vev]
            best_bid, best_ask = self.get_best_bid_ask(od)
            if best_bid is None or best_ask is None:
                continue
            mid = (best_bid + best_ask) / 2.0
            K = self.STRIKES[vev]
            iv = self.bs_implied_vol(mid, S, K, T)
            if iv is not None and 0.01 < iv < 3.0:
                moneyness = math.log(S / K)
                ivs.append((moneyness, iv))

        # Need at least 2 points to fit a quadratic; 3 for full quadratic
        result: Dict[str, float] = {}

        if len(ivs) >= 3:
            # Fit quadratic: iv = a + b*m + c*m^2  via least-squares
            n = len(ivs)
            ms = [x[0] for x in ivs]
            vs = [x[1] for x in ivs]
            # Build normal equations for [a, b, c]
            sum1 = n
            summ = sum(ms)
            summ2 = sum(m ** 2 for m in ms)
            summ3 = sum(m ** 3 for m in ms)
            summ4 = sum(m ** 4 for m in ms)
            sumv = sum(vs)
            summv = sum(ms[i] * vs[i] for i in range(n))
            summ2v = sum(ms[i] ** 2 * vs[i] for i in range(n))
            # Solve 3x3 system (Cramer's rule)
            A = [[sum1, summ, summ2], [summ, summ2, summ3], [summ2, summ3, summ4]]
            b_vec = [sumv, summv, summ2v]
            try:
                a_coef, b_coef, c_coef = _solve3x3(A, b_vec)
                for vev in self.VOUCHERS:
                    K = self.STRIKES[vev]
                    m = math.log(S / K)
                    iv_fit = a_coef + b_coef * m + c_coef * m ** 2
                    result[vev] = max(0.01, min(iv_fit, 3.0))
            except Exception:
                for vev in self.VOUCHERS:
                    result[vev] = self.hist_vol

        elif len(ivs) == 2:
            # Linear fit
            (m0, v0), (m1, v1) = ivs[0], ivs[1]
            if abs(m1 - m0) > 1e-9:
                slope = (v1 - v0) / (m1 - m0)
                intercept = v0 - slope * m0
            else:
                slope, intercept = 0.0, (v0 + v1) / 2.0
            for vev in self.VOUCHERS:
                K = self.STRIKES[vev]
                m = math.log(S / K)
                result[vev] = max(0.01, min(intercept + slope * m, 3.0))

        elif len(ivs) == 1:
            flat_iv = ivs[0][1]
            for vev in self.VOUCHERS:
                result[vev] = flat_iv

        else:
            for vev in self.VOUCHERS:
                result[vev] = self.hist_vol

        return result

    # ==========================================================================
    # Options trading
    # ==========================================================================
    def get_tte(self, state: TradingState) -> float:
        """Time-to-expiry in years. At timestamp=0, TTE=5 days."""
        tte_days = self.TTE_START_DAYS - state.timestamp / 1_000_000.0
        return max(tte_days, 1e-6) / 365.0

    def trade_vouchers(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        vfe_product = "VELVETFRUIT_EXTRACT"
        if vfe_product not in state.order_depths:
            return

        vfe_od = state.order_depths[vfe_product]
        vfe_bid, vfe_ask = self.get_best_bid_ask(vfe_od)
        if vfe_bid is None or vfe_ask is None:
            return
        S = (vfe_bid + vfe_ask) / 2.0
        T = self.get_tte(state)

        iv_map = self.calibrate_vol_smile(state, S, T)

        # Track total delta of options positions (for hedging)
        net_delta = 0.0
        for vev in self.VOUCHERS:
            pos = self.positions.get(vev, 0)
            K = self.STRIKES[vev]
            iv = iv_map.get(vev, self.hist_vol)
            net_delta += pos * self.bs_delta(S, K, T, iv)

        # Trade each voucher
        for vev in self.VOUCHERS:
            if vev not in state.order_depths:
                result.setdefault(vev, [])
                continue

            od = state.order_depths[vev]
            best_bid, best_ask = self.get_best_bid_ask(od)
            if best_bid is None or best_ask is None:
                result.setdefault(vev, [])
                continue

            mkt_mid = (best_bid + best_ask) / 2.0
            K = self.STRIKES[vev]
            iv = iv_map.get(vev, self.hist_vol)
            fair = self.bs_call_price(S, K, T, iv)
            edge = fair - mkt_mid

            orders = result.setdefault(vev, [])
            limit = self.get_position_limit(vev)

            # Kelly criterion sizing
            kelly_size = 0
            if fair > 0.0:
                kelly_frac = abs(edge) / fair
                kelly_size = min(int(kelly_frac * limit), self.MAX_TAKE_PER_TICK)

            if edge > self.MIN_EDGE and kelly_size > 0:
                # Underpriced — buy aggressively at best ask
                self.send_buy_order(orders, vev, best_ask, kelly_size)

            elif edge < -self.MIN_EDGE and kelly_size > 0:
                # Overpriced — sell aggressively at best bid
                self.send_sell_order(orders, vev, best_bid, kelly_size)

            # Passive quotes inside the spread
            if best_ask - best_bid >= 2:
                self.send_buy_order(orders, vev, best_bid + 1, self.PASSIVE_QUOTE_SIZE)
                self.send_sell_order(orders, vev, best_ask - 1, self.PASSIVE_QUOTE_SIZE)

        # Delta hedge: offset net options delta with VFE position
        hedge_qty = -int(round(net_delta))
        vfe_orders = result.setdefault(vfe_product, [])
        if hedge_qty > 0:
            # Need to buy VFE to hedge short delta
            if vfe_ask is not None:
                self.send_buy_order(vfe_orders, vfe_product, vfe_ask, hedge_qty)
        elif hedge_qty < 0:
            # Need to sell VFE to hedge long delta
            if vfe_bid is not None:
                self.send_sell_order(vfe_orders, vfe_product, vfe_bid, -hedge_qty)

    # ==========================================================================
    # Delta-1: z-score mean reversion (generalised from round_2 trade_ash)
    # ==========================================================================
    def _trade_delta1(
        self,
        product: str,
        state: TradingState,
        orders: List[Order],
        mid_history: List[float],
        history_length: int,
        entry_z: float,
        exit_z: float,
        quote_offset: int,
        passive_size: int,
        max_take: int,
        skew: float,
    ) -> None:
        if product not in state.order_depths:
            return
        od = state.order_depths[product]
        best_bid, best_ask = self.get_best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return

        mid = (best_bid + best_ask) / 2.0
        mid_history.append(mid)
        if len(mid_history) > history_length:
            mid_history.pop(0)

        if not mid_history:
            return

        mean = sum(mid_history) / len(mid_history)
        var = sum((x - mean) ** 2 for x in mid_history) / len(mid_history)
        std = math.sqrt(max(var, 0.0))

        z = (mid - mean) / std if std > 1e-6 else 0.0

        position = self.positions.get(product, 0)
        limit = self.get_position_limit(product)

        # Exit: unwind position as price reverts
        if position > 0 and z >= -exit_z:
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid >= mean and self.remaining_sell_capacity(product) > 0:
                    size = min(vol, position, self.remaining_sell_capacity(product), max_take)
                    if size > 0:
                        self.send_sell_order(orders, product, bid, size)
                    break
        elif position < 0 and z <= exit_z:
            for ask, vol in sorted(od.sell_orders.items()):
                if ask <= mean and self.remaining_buy_capacity(product) > 0:
                    size = min(-vol, -position, self.remaining_buy_capacity(product), max_take)
                    if size > 0:
                        self.send_buy_order(orders, product, ask, size)
                    break

        # Entry: mean reversion
        if z < -entry_z:
            taken = 0
            for ask, vol in sorted(od.sell_orders.items()):
                if ask <= mean and self.remaining_buy_capacity(product) > 0 and taken < max_take:
                    size = min(-vol, self.remaining_buy_capacity(product), max_take - taken)
                    if size > 0:
                        self.send_buy_order(orders, product, ask, size)
                        taken += size
        elif z > entry_z:
            taken = 0
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid >= mean and self.remaining_sell_capacity(product) > 0 and taken < max_take:
                    size = min(vol, self.remaining_sell_capacity(product), max_take - taken)
                    if size > 0:
                        self.send_sell_order(orders, product, bid, size)
                        taken += size

        # Passive quotes with inventory skew
        pos_ratio = position / limit
        inv_skew = skew * pos_ratio

        bid_price = int(mean - quote_offset - inv_skew)
        ask_price = int(mean + quote_offset - inv_skew)

        if best_bid is not None:
            bid_price = min(best_bid + 1, bid_price)
        if best_ask is not None:
            ask_price = max(best_ask - 1, ask_price)
        if best_ask is not None:
            bid_price = min(bid_price, best_ask - 1)
        if best_bid is not None:
            ask_price = max(ask_price, best_bid + 1)

        if bid_price < ask_price:
            buy_sz = min(passive_size, self.remaining_buy_capacity(product))
            sell_sz = min(passive_size, self.remaining_sell_capacity(product))
            if position > limit * 0.5:
                buy_sz = min(buy_sz, 6)
                sell_sz = min(self.remaining_sell_capacity(product), 12)
            elif position < -limit * 0.5:
                sell_sz = min(sell_sz, 6)
                buy_sz = min(self.remaining_buy_capacity(product), 12)
            if buy_sz > 0:
                self.send_buy_order(orders, product, bid_price, buy_sz)
            if sell_sz > 0:
                self.send_sell_order(orders, product, ask_price, sell_sz)

    def trade_vfe(self, state: TradingState, orders: List[Order]) -> None:
        self._trade_delta1(
            "VELVETFRUIT_EXTRACT", state, orders,
            self.vfe_mid_history, self.VFE_HISTORY_LENGTH,
            self.VFE_ENTRY_Z, self.VFE_EXIT_Z, self.VFE_QUOTE_OFFSET,
            self.VFE_PASSIVE_SIZE, self.VFE_MAX_TAKE, self.VFE_SKEW,
        )

    def trade_hgp(self, state: TradingState, orders: List[Order]) -> None:
        self._trade_delta1(
            "HYDROGEL_PACK", state, orders,
            self.hgp_mid_history, self.HGP_HISTORY_LENGTH,
            self.HGP_ENTRY_Z, self.HGP_EXIT_Z, self.HGP_QUOTE_OFFSET,
            self.HGP_PASSIVE_SIZE, self.HGP_MAX_TAKE, self.HGP_SKEW,
        )

    # ==========================================================================
    # EWMA volatility update
    # ==========================================================================
    def update_hist_vol(self) -> None:
        if len(self.vfe_mid_history) < 2:
            return
        prev = self.vfe_mid_history[-2]
        curr = self.vfe_mid_history[-1]
        if prev <= 0.0:
            return
        ret = (curr - prev) / prev
        self.hist_vol = math.sqrt(
            self.EWMA_LAMBDA * self.hist_vol ** 2 + (1 - self.EWMA_LAMBDA) * ret ** 2
        )
        self.hist_vol = max(0.01, min(self.hist_vol, 3.0))

    # ==========================================================================
    # Main
    # ==========================================================================
    def run(self, state: TradingState):
        all_products = self.PRODUCTS
        result: Dict[str, List[Order]] = {p: [] for p in all_products}

        # Reset per-tick counters
        self.positions = {p: self.get_product_pos(state, p) for p in all_products}
        self.buy_orders_sent = {p: 0 for p in all_products}
        self.sell_orders_sent = {p: 0 for p in all_products}

        self.load_state(state)

        # Delta-1 strategies
        self.trade_vfe(state, result["VELVETFRUIT_EXTRACT"])
        self.trade_hgp(state, result["HYDROGEL_PACK"])

        # Update EWMA vol after VFE mid has been appended by trade_vfe
        self.update_hist_vol()

        # Options + delta hedge
        self.trade_vouchers(state, result)

        return result, 0, self.save_state()


# ---------------------------------------------------------------------------
# 3x3 linear system solver (Cramer's rule)
# ---------------------------------------------------------------------------
def _solve3x3(A: List[List[float]], b: List[float]) -> Tuple[float, float, float]:
    def det3(m):
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    D = det3(A)
    if abs(D) < 1e-12:
        raise ValueError("Singular matrix")

    def replace_col(A, b, col):
        M = [row[:] for row in A]
        for i in range(3):
            M[i][col] = b[i]
        return M

    x = det3(replace_col(A, b, 0)) / D
    y = det3(replace_col(A, b, 1)) / D
    z = det3(replace_col(A, b, 2)) / D
    return x, y, z
