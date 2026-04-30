from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple, Any
import jsonpickle
import math


class Trader:
    """
    Hybrid Pebbles Trader - Version 4: XS/L priority + slightly more active side scanner.

    Trades only Pebbles.

    Idea:
    1. Use the XS/L specialist signal as the main attacking module when it is active.
       This preserves the high-upside behavior of our v11 XS/L strategy.
    2. When XS/L specialist is active, allow a small non-conflicting scanner only on XL/S.
       This tries to add the teammate scanner alpha while XS/L owns XS and L.
    3. When XS/L specialist is inactive, use the normal quality-filtered relative-z scanner.
       PEBBLES_M is tracked and flattened if needed, but excluded from new scanner pairs.
    4. All targets are centralized, so the two Pebbles strategies never fight on the same product.

    No PnL lock.
    No timestamp cutoff.
    No fixed static target basket.
    """

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

    # ---------- XS/L specialist parameters, inherited from v11 ----------
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

    # ---------- Dynamic scanner parameters, based on teammate version ----------
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

    # Side scanner used only while XS/L specialist is active.
    # It only trades XL/S, so it does not fight the specialist's XS/L inventory.
    SIDE_SCAN_PRODUCTS = [
        "PEBBLES_XL",
        "PEBBLES_S",
    ]
    # v4: lower than v3 to let XL/S side scanner pick up more non-conflicting opportunities.
    SIDE_ENTRY_Z = 2.10
    SIDE_EXIT_Z = 0.90
    SIDE_FLIP_BUFFER = 0.80
    SIDE_PASSIVE_SIZE = 1
    SIDE_MAX_TARGET = 4
    SIDE_NORMAL_IMPROVE = 1
    SIDE_ENTRY_IMPROVE = 2

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {p: [] for p in self.PRODUCTS}
        conversions = 0

        memory = self.load_memory(state.traderData)
        memory.setdefault("xl", {})
        memory.setdefault("scanner", {})
        memory.setdefault("side_scanner", {})
        memory.setdefault("owner", "scanner")

        # Need all Pebbles order books available.
        for product in self.PRODUCTS:
            if product not in state.order_depths:
                return result, conversions, jsonpickle.encode(memory)

        mids: Dict[str, float] = {}
        for product in self.PRODUCTS:
            mid = self.get_mid(state.order_depths[product])
            if mid is None:
                return result, conversions, jsonpickle.encode(memory)
            mids[product] = mid

        # Update both signal modules every tick, even if one is not currently trading.
        xl_mode, xl_ready = self.update_xs_l_specialist(memory["xl"], mids)
        scanner_targets, scanner_just_opened = self.update_scanner(memory["scanner"], mids)
        side_targets, side_just_opened = self.update_side_scanner(memory["side_scanner"], mids)

        positions = {p: state.position.get(p, 0) for p in self.PRODUCTS}
        targets = {p: 0 for p in self.PRODUCTS}
        use_specialist_execution = False

        xs_l_inventory = abs(positions[self.XS]) > 0 or abs(positions[self.L]) > 0

        # Priority rule:
        # If XS/L specialist has an active mode, it owns the Pebbles book.
        # If it just exited but still has inventory, it keeps ownership until flat.
        if xl_mode != 0:
            memory["owner"] = "xs_l"
            memory["scanner"]["active_rich"] = None
            memory["scanner"]["active_cheap"] = None

        elif memory.get("owner") == "xs_l" and xs_l_inventory:
            # Continue flattening specialist inventory before allowing scanner back in.
            pass

        else:
            memory["owner"] = "scanner"

        if memory.get("owner") == "xs_l":
            use_specialist_execution = True
            if xl_mode == -1:
                # short spread: sell XS, buy L
                targets[self.XS] = -self.XS_L_TARGET_SIZE
                targets[self.L] = self.XS_L_TARGET_SIZE
                # Add only non-conflicting XL/S side-scanner targets.
                targets["PEBBLES_XL"] = side_targets.get("PEBBLES_XL", 0)
                targets["PEBBLES_S"] = side_targets.get("PEBBLES_S", 0)
            elif xl_mode == 1:
                # long spread: buy XS, sell L
                targets[self.XS] = self.XS_L_TARGET_SIZE
                targets[self.L] = -self.XS_L_TARGET_SIZE
                targets["PEBBLES_XL"] = side_targets.get("PEBBLES_XL", 0)
                targets["PEBBLES_S"] = side_targets.get("PEBBLES_S", 0)
            else:
                # Specialist flat mode: flatten all Pebbles before handing control back.
                targets = {p: 0 for p in self.PRODUCTS}
        else:
            # Main scanner owns the book; side scanner must not run separately.
            targets = scanner_targets

        # Execute centralized targets.
        if use_specialist_execution:
            # Use taking-style execution for XS/L to preserve v11 behavior.
            for product in self.PRODUCTS:
                current_pos = positions[product]
                target_pos = self.clamp(targets[product], -self.POSITION_LIMIT, self.POSITION_LIMIT)

                if product in (self.XS, self.L):
                    result[product] += self.take_toward_target(
                        product,
                        state.order_depths[product],
                        current_pos,
                        target_pos,
                        self.XS_L_ORDER_SIZE,
                    )
                else:
                    # Non-conflicting side scanner on XL/S while specialist owns XS/L.
                    # Other Pebbles are quietly flattened.
                    if product in self.SIDE_SCAN_PRODUCTS:
                        side_improve = self.SIDE_ENTRY_IMPROVE if side_just_opened else self.SIDE_NORMAL_IMPROVE
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
            improve = self.SCAN_ENTRY_IMPROVE if scanner_just_opened else self.SCAN_NORMAL_IMPROVE
            for product in self.PRODUCTS:
                result[product] += self.passive_toward_target(
                    product,
                    state.order_depths[product],
                    positions[product],
                    self.clamp(targets[product], -self.POSITION_LIMIT, self.POSITION_LIMIT),
                    self.SCAN_PASSIVE_SIZE,
                    improve,
                )

        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData

    # =====================================================================
    # XS/L specialist signal. Updates memory and returns current mode.
    # mode = -1 short XS/long L, +1 long XS/short L, 0 flat.
    # =====================================================================
    def update_xs_l_specialist(self, data: Dict[str, Any], mids: Dict[str, float]) -> Tuple[int, bool]:
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

        max_hist = max(self.XL_WINDOW, self.ACCEL_LONG, self.SHORT_VOL_WINDOW) + 5
        data["spread_hist"] = data["spread_hist"][-max_hist:]
        data["raw_hist"] = data["raw_hist"][-max_hist:]

        hist = data["spread_hist"]
        raw_hist = data["raw_hist"]

        if len(hist) < self.XL_WINDOW or len(raw_hist) < self.XL_WINDOW:
            data["last_ready"] = False
            return int(data.get("mode", 0)), False

        long_hist = hist[-self.XL_WINDOW:]
        mean = self.mean(long_hist)
        std = self.std(long_hist)
        if std < 1e-9:
            data["last_ready"] = False
            return int(data.get("mode", 0)), False

        z = (spread - mean) / std
        raw_std = self.std(raw_hist[-self.XL_WINDOW:])
        if raw_std < 1e-9:
            raw_std = 1.0

        short_vol = self.std(hist[-self.SHORT_VOL_WINDOW:])
        vol_ratio = short_vol / std if std > 1e-9 else 1.0

        accel_z = 0.0
        if len(hist) > self.ACCEL_LONG:
            recent_change = hist[-1] - hist[-1 - self.ACCEL_SHORT]
            older_change = hist[-1 - self.ACCEL_SHORT] - hist[-1 - self.ACCEL_LONG]
            accel_z = (recent_change - older_change) / std

        mode = int(data.get("mode", 0))

        if mode == 0:
            data["hold_bars"] = 0

            if not data.get("reset_ready", True):
                if abs(z) < self.RESET_Z:
                    data["reset_ready"] = True

            if data.get("reset_ready", True):
                if z > self.SHORT_ENTRY_Z and accel_z <= self.SHORT_MAX_ACCEL_Z:
                    mode = -1
                    self.start_xl_trade(data, raw_diff, raw_std)
                elif z < -self.LONG_ENTRY_Z and vol_ratio >= self.LONG_MIN_VOL_RATIO:
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
                if favorable_move >= self.RAW_TRAIL_ACTIVATE_STD * entry_raw_std:
                    trail_active = True
                if trail_active and retrace >= self.RAW_TRAIL_RETRACE_STD * raw_std:
                    mode = 0
                    exit_reason = "raw_trail"

            elif mode == 1:
                if raw_diff > best_raw:
                    best_raw = raw_diff
                favorable_move = best_raw - entry_raw
                retrace = best_raw - raw_diff
                if favorable_move >= self.RAW_TRAIL_ACTIVATE_STD * entry_raw_std:
                    trail_active = True
                if trail_active and retrace >= self.RAW_TRAIL_RETRACE_STD * raw_std:
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
                data["reset_ready"] = True if exit_reason == "z_exit" else False

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
    def start_xl_trade(data: Dict[str, Any], raw_diff: float, raw_std: float) -> None:
        data["hold_bars"] = 0
        data["entry_raw_diff"] = raw_diff
        data["best_raw_diff"] = raw_diff
        data["entry_raw_std"] = raw_std
        data["raw_trail_active"] = False

    # =====================================================================
    # Non-conflicting side scanner used only when XS/L specialist is active.
    # It trades only XL/S and uses smaller size than the main scanner.
    # =====================================================================
    def update_side_scanner(self, data: Dict[str, Any], mids: Dict[str, float]) -> Tuple[Dict[str, int], bool]:
        data.setdefault("mid_history", {})
        data.setdefault("active_rich", None)
        data.setdefault("active_cheap", None)
        data.setdefault("just_opened_pair", False)
        data["just_opened_pair"] = False

        for product in self.SIDE_SCAN_PRODUCTS:
            data["mid_history"].setdefault(product, [])
            data["mid_history"][product].append(mids[product])
            data["mid_history"][product] = data["mid_history"][product][-self.SCAN_WINDOW:]

        targets = {p: 0 for p in self.PRODUCTS}
        min_hist_len = min(len(data["mid_history"][p]) for p in self.SIDE_SCAN_PRODUCTS)
        if min_hist_len < self.SCAN_MIN_HISTORY:
            return targets, False

        z_scores: Dict[str, float] = {}
        for product in self.SIDE_SCAN_PRODUCTS:
            hist = data["mid_history"][product]
            z_scores[product] = (mids[product] - self.mean(hist)) / self.std_sample(hist)

        avg_z = self.mean(list(z_scores.values()))
        rel_z = {p: z_scores[p] - avg_z for p in self.SIDE_SCAN_PRODUCTS}

        rich_product = max(self.SIDE_SCAN_PRODUCTS, key=lambda p: rel_z[p])
        cheap_product = min(self.SIDE_SCAN_PRODUCTS, key=lambda p: rel_z[p])
        best_z_spread = rel_z[rich_product] - rel_z[cheap_product]

        active_rich = data.get("active_rich")
        active_cheap = data.get("active_cheap")
        have_active_pair = active_rich in self.SIDE_SCAN_PRODUCTS and active_cheap in self.SIDE_SCAN_PRODUCTS
        active_pair_spread = rel_z[active_rich] - rel_z[active_cheap] if have_active_pair else 0.0

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
                and (rich_product != active_rich or cheap_product != active_cheap)
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
            size = self.side_scanner_size(max(active_pair_spread, self.SIDE_ENTRY_Z))
            targets[active_rich] = -size
            targets[active_cheap] = size

        return targets, bool(data.get("just_opened_pair", False))

    @classmethod
    def side_scanner_size(cls, z_spread: float) -> int:
        if z_spread < 2.90:
            return 2
        if z_spread < 3.70:
            return 3
        return cls.SIDE_MAX_TARGET

    # =====================================================================
    # 5-Pebble relative-z scanner.
    # =====================================================================
    def update_scanner(self, data: Dict[str, Any], mids: Dict[str, float]) -> Tuple[Dict[str, int], bool]:
        data.setdefault("mid_history", {})
        data.setdefault("active_rich", None)
        data.setdefault("active_cheap", None)
        data.setdefault("just_opened_pair", False)
        data["just_opened_pair"] = False

        for product in self.PRODUCTS:
            data["mid_history"].setdefault(product, [])
            data["mid_history"][product].append(mids[product])
            data["mid_history"][product] = data["mid_history"][product][-self.SCAN_WINDOW:]

        targets = {p: 0 for p in self.PRODUCTS}
        min_hist_len = min(len(data["mid_history"][p]) for p in self.SCAN_PRODUCTS)
        if min_hist_len < self.SCAN_MIN_HISTORY:
            return targets, False

        z_scores: Dict[str, float] = {}
        for product in self.SCAN_PRODUCTS:
            hist = data["mid_history"][product]
            z_scores[product] = (mids[product] - self.mean(hist)) / self.std_sample(hist)

        avg_z = self.mean(list(z_scores.values()))
        rel_z = {p: z_scores[p] - avg_z for p in self.SCAN_PRODUCTS}

        rich_product = max(self.SCAN_PRODUCTS, key=lambda p: rel_z[p])
        cheap_product = min(self.SCAN_PRODUCTS, key=lambda p: rel_z[p])
        best_z_spread = rel_z[rich_product] - rel_z[cheap_product]

        active_rich = data.get("active_rich")
        active_cheap = data.get("active_cheap")
        have_active_pair = active_rich in self.SCAN_PRODUCTS and active_cheap in self.SCAN_PRODUCTS
        active_pair_spread = rel_z[active_rich] - rel_z[active_cheap] if have_active_pair else 0.0

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
                and (rich_product != active_rich or cheap_product != active_cheap)
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
            size = self.scanner_size(max(active_pair_spread, self.SCAN_ENTRY_Z))
            targets[active_rich] = -size
            targets[active_cheap] = size

        return targets, bool(data.get("just_opened_pair", False))

    @staticmethod
    def scanner_size(z_spread: float) -> int:
        if z_spread < 2.75:
            return 3
        if z_spread < 3.55:
            return 6
        return 10

    # =====================================================================
    # Execution helpers.
    # =====================================================================
    def take_toward_target(self, product: str, od: OrderDepth, current_pos: int, target_pos: int, max_size: int) -> List[Order]:
        orders: List[Order] = []
        diff = target_pos - current_pos
        if diff > 0:
            if not od.sell_orders:
                return orders
            best_ask = min(od.sell_orders.keys())
            ask_volume = -od.sell_orders[best_ask]
            qty = min(diff, ask_volume, max_size, self.POSITION_LIMIT - current_pos)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        elif diff < 0:
            if not od.buy_orders:
                return orders
            best_bid = max(od.buy_orders.keys())
            bid_volume = od.buy_orders[best_bid]
            qty = min(-diff, bid_volume, max_size, self.POSITION_LIMIT + current_pos)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        return orders

    def passive_toward_target(self, product: str, od: OrderDepth, current_pos: int, target_pos: int, max_size: int, improve: int) -> List[Order]:
        orders: List[Order] = []
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
    def get_mid(od: OrderDepth) -> Optional[float]:
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def std(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))

    @staticmethod
    def std_sample(values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        m = sum(values) / len(values)
        return math.sqrt(max(sum((x - m) ** 2 for x in values) / (len(values) - 1), 1e-6))

    @staticmethod
    def clamp(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(x)))

    @staticmethod
    def load_memory(traderData: str) -> Dict[str, Any]:
        if not traderData:
            return {}
        try:
            decoded = jsonpickle.decode(traderData)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
        return {}
