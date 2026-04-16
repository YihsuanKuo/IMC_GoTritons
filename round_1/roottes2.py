from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # 读取历史数据
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except:
            data = {}

        if "pepper" not in data:
            data["pepper"] = {
                "last_mid": None,
                "drift": 0.0,
            }

        pepper_data = data["pepper"]

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            # ==================== INTARIAN_PEPPER_ROOT ====================
            if product == "INTARIAN_PEPPER_ROOT":
                buy_orders = order_depth.buy_orders
                sell_orders = order_depth.sell_orders

                best_bid = max(buy_orders.keys()) if len(buy_orders) > 0 else None
                best_ask = min(sell_orders.keys()) if len(sell_orders) > 0 else None

                # 更新 mid / drift
                mid = None
                if best_bid is not None and best_ask is not None:
                    mid = (best_bid + best_ask) / 2

                    if pepper_data["last_mid"] is not None:
                        change = mid - pepper_data["last_mid"]
                        pepper_data["drift"] = 0.8 * pepper_data["drift"] + 0.2 * change

                    pepper_data["last_mid"] = mid

                drift = pepper_data["drift"]
                remaining = limit - current_position

                if remaining > 0:
                    # -------- 1. 先挂 passive buy，争取更低成本成交 --------
                    if best_bid is not None and best_ask is not None and best_bid < best_ask:
                        spread = best_ask - best_bid

                        # 挂在 bid+1，尽量吃到更便宜的货
                        passive_price = best_bid + 1

                        # 不要挂成穿价单
                        if passive_price >= best_ask:
                            passive_price = best_bid

                        # passive 单量
                        # 前半仓更激进，后半仓更保守
                        if current_position < 40:
                            passive_qty = min(remaining, 12)
                        else:
                            passive_qty = min(remaining, 8)

                        if passive_qty > 0:
                            orders.append(Order(product, passive_price, passive_qty))

                        # -------- 2. 必要时用 taker 补仓 --------
                        # 条件：
                        # - 仓位太低（怕错过主升段）
                        # - 或 spread 很小，直接吃也不亏
                        # - 或 drift 明显为正，说明继续涨
                        should_take = False

                        if current_position < 20:
                            should_take = True
                        elif spread <= 2:
                            should_take = True
                        elif drift > 0.8:
                            should_take = True

                        if should_take and len(sell_orders) > 0:
                            take_remaining = remaining

                            # 已经挂了 passive 单，不要全扫太猛
                            if current_position < 20:
                                take_cap = min(take_remaining, 20)
                            elif current_position < 40:
                                take_cap = min(take_remaining, 10)
                            else:
                                take_cap = min(take_remaining, 5)

                            swept = 0
                            for ask_price in sorted(sell_orders.keys()):
                                if swept >= take_cap:
                                    break

                                ask_volume = -sell_orders[ask_price]
                                buy_volume = min(take_cap - swept, ask_volume)

                                if buy_volume > 0:
                                    orders.append(Order(product, ask_price, buy_volume))
                                    swept += buy_volume

                    # -------- 3. 如果只有卖盘没有买盘，仍然允许直接扫 --------
                    elif best_ask is not None:
                        # 这种情况不常见，但不能漏
                        take_cap = min(remaining, 15)
                        swept = 0

                        for ask_price in sorted(sell_orders.keys()):
                            if swept >= take_cap:
                                break

                            ask_volume = -sell_orders[ask_price]
                            buy_volume = min(take_cap - swept, ask_volume)

                            if buy_volume > 0:
                                orders.append(Order(product, ask_price, buy_volume))
                                swept += buy_volume

            # ==================== ASH_COATED_OSMIUM ====================
            elif product == "ASH_COATED_OSMIUM":
                # 暂时不交易，避免干扰 root 分析
                pass

            result[product] = orders

        traderData = json.dumps(data)
        conversions = 0
        return result, conversions, traderData