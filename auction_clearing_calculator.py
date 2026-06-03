#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Order book data from prompt
# -----------------------------
PRODUCT_BOOKS: Dict[str, Dict[str, Dict[int, int]]] = {
    "EMBER_MUSHROOM": {
        "bids": {
            20: 43000,
            19: 17000,
            18: 6000,
            17: 5000,
            16: 10000,
            15: 5000,
            14: 10000,
            13: 7000,
        },
        "asks": {
            12: 20000,
            13: 25000,
            14: 35000,
            15: 6000,
            16: 5000,
            17: 0,
            18: 10000,
            19: 12000,
        },
    },
    "DRYLAND_FLAX": {
        "bids": {
            30: 30000,
            29: 5000,
            28: 12000,
            27: 28000,
        },
        "asks": {
            28: 40000,
            31: 20000,
            32: 20000,
            33: 30000,
        },
    },
}
PRODUCT_NAMES = tuple(PRODUCT_BOOKS.keys())


def cumulative_bid_volume(bids: Dict[int, int], price: int) -> int:
    return sum(vol for p, vol in bids.items() if p >= price)


def cumulative_ask_volume(asks: Dict[int, int], price: int) -> int:
    return sum(vol for p, vol in asks.items() if p <= price)


def all_candidate_prices(bids: Dict[int, int], asks: Dict[int, int]) -> List[int]:
    return sorted(set(bids.keys()) | set(asks.keys()))


def compute_clearing_price_and_volume(
    bids: Dict[int, int], asks: Dict[int, int]
) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    diagnostics = []
    best_price = None
    best_matched = -1

    for price in all_candidate_prices(bids, asks):
        cbid = cumulative_bid_volume(bids, price)
        cask = cumulative_ask_volume(asks, price)
        matched = min(cbid, cask)
        diagnostics.append((price, cbid, cask, matched))

        # Rule 1: maximize volume
        # Rule 2: if tied, choose the higher price
        if matched > best_matched or (matched == best_matched and (best_price is None or price > best_price)):
            best_matched = matched
            best_price = price

    return best_price, best_matched, diagnostics


def allocate_buyer_fill(
    original_bids: Dict[int, int],
    asks: Dict[int, int],
    my_price: int,
    my_qty: int,
    clearing_price: int,
    total_matched: int,
) -> int:
    if my_price < clearing_price or my_qty <= 0:
        return 0

    # Volume ahead of you on buy side:
    higher_price_volume = sum(vol for p, vol in original_bids.items() if p > my_price and p >= clearing_price)
    same_price_existing_volume = original_bids.get(my_price, 0) if my_price >= clearing_price else 0
    ahead_of_you = higher_price_volume + same_price_existing_volume

    remaining_for_you = total_matched - ahead_of_you
    if remaining_for_you <= 0:
        return 0

    return min(my_qty, remaining_for_you)


def normalize_product(product: str) -> str:
    product = product.upper().strip()
    if product not in PRODUCT_BOOKS:
        raise ValueError("product must be 'DRYLAND_FLAX' or 'EMBER_MUSHROOM'")
    return product


def selected_products(selection: str) -> List[str]:
    selection = selection.upper().strip()
    if selection in {"ALL", "BOTH"}:
        return list(PRODUCT_NAMES)
    return [normalize_product(selection)]


def get_product_book(product: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    product = normalize_product(product)
    book = PRODUCT_BOOKS[product]
    return dict(book["bids"]), dict(book["asks"])


def product_terms(product: str) -> Tuple[float, float]:
    """Return (buyback_price, fee_per_unit)."""
    product = normalize_product(product)
    return (30.0, 0.0) if product == "DRYLAND_FLAX" else (20.0, 0.10)


def analyze_buy_order(
    my_price: int,
    my_qty: int,
    product: str = "EMBER_MUSHROOM",
) -> Dict:

    product = normalize_product(product)

    buyback_price, fee_per_unit = product_terms(product)

    original_bids, asks = get_product_book(product)

    # Add your buy order into bid book
    new_bids = dict(original_bids)
    new_bids[my_price] = new_bids.get(my_price, 0) + my_qty

    clearing_price, total_matched, diagnostics = compute_clearing_price_and_volume(new_bids, asks)

    my_fill = allocate_buyer_fill(
        original_bids=original_bids,
        asks=asks,
        my_price=my_price,
        my_qty=my_qty,
        clearing_price=clearing_price,
        total_matched=total_matched,
    )

    auction_cost = my_fill * clearing_price
    buyback_revenue = my_fill * buyback_price
    fees = my_fill * fee_per_unit
    profit = buyback_revenue - auction_cost - fees

    return {
        "product": product,
        "my_price": my_price,
        "my_qty": my_qty,
        "clearing_price": clearing_price,
        "total_matched_volume": total_matched,
        "my_filled_volume": my_fill,
        "auction_cost": auction_cost,
        "buyback_revenue": buyback_revenue,
        "fees": fees,
        "profit": profit,
        "diagnostics": diagnostics,
    }


def default_scan_price_range(product: str) -> Tuple[int, int]:
    """Reasonable default BUY price range for the automatic scan."""
    product = normalize_product(product)
    buyback_price, _ = product_terms(product)
    bids, asks = get_product_book(product)
    book_prices = all_candidate_prices(bids, asks)
    min_price = min(book_prices)
    max_price = max(max(book_prices), int(buyback_price))
    return min_price, max_price


def scan_buy_price_opportunities(
    product: str = "EMBER_MUSHROOM",
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    max_qty: Optional[int] = None,
) -> List[Dict]:

    product = normalize_product(product)

    default_min_price, default_max_price = default_scan_price_range(product)
    min_price = default_min_price if min_price is None else min_price
    max_price = default_max_price if max_price is None else max_price
    _, asks = get_product_book(product)
    max_qty = sum(asks.values()) if max_qty is None else max_qty

    if min_price > max_price:
        raise ValueError("min_price must be <= max_price")
    if max_qty <= 0:
        raise ValueError("max_qty must be > 0")

    best_rows: Dict[Tuple[int, int], Dict] = {}

    for my_price in range(min_price, max_price + 1):
        for my_qty in range(1, max_qty + 1):
            result = analyze_buy_order(my_price=my_price, my_qty=my_qty, product=product)
            if result["my_filled_volume"] <= 0:
                continue

            key = (my_price, result["clearing_price"])
            current_best = best_rows.get(key)

            is_better = (
                current_best is None
                or result["my_filled_volume"] > current_best["max_filled_volume"]
                or (
                    result["my_filled_volume"] == current_best["max_filled_volume"]
                    and result["profit"] > current_best["profit"]
                )
                or (
                    result["my_filled_volume"] == current_best["max_filled_volume"]
                    and result["profit"] == current_best["profit"]
                    and result["my_qty"] < current_best["submitted_qty"]
                )
            )

            if is_better:
                best_rows[key] = {
                    "product": product,
                    "buy_price": result["my_price"],
                    "submitted_qty": result["my_qty"],
                    "clearing_price": result["clearing_price"],
                    "max_filled_volume": result["my_filled_volume"],
                    "total_matched_volume": result["total_matched_volume"],
                    "auction_cost": result["auction_cost"],
                    "buyback_revenue": result["buyback_revenue"],
                    "fees": result["fees"],
                    "profit": result["profit"],
                }

    return sorted(best_rows.values(), key=lambda row: (row["buy_price"], row["clearing_price"]))


def print_result(result: Dict) -> None:
    print("=" * 60)
    print(f"Product              : {result['product']}")
    print(f"My buy order         : price={result['my_price']}, qty={result['my_qty']}")
    print(f"Clearing price       : {result['clearing_price']}")
    print(f"Total traded volume  : {result['total_matched_volume']}")
    print(f"My filled volume     : {result['my_filled_volume']}")
    print(f"Auction cost         : {result['auction_cost']:.2f}")
    print(f"Buyback revenue      : {result['buyback_revenue']:.2f}")
    print(f"Fees                 : {result['fees']:.2f}")
    print(f"Profit               : {result['profit']:.2f}")
    print("=" * 60)
    print("\nPrice-by-price diagnostics:")
    print("price | cum_bid | cum_ask | matched")
    for price, cbid, cask, matched in result["diagnostics"]:
        marker = "  <-- chosen" if price == result["clearing_price"] else ""
        print(f"{price:>5} | {cbid:>7} | {cask:>7} | {matched:>7}{marker}")


def print_scan_results(rows: List[Dict], title: str = "Best overall opportunities") -> None:
    if not rows:
        print("No profitable or executable BUY opportunities were found in the scan.")
        return

    multiple_products = len({row["product"] for row in rows}) > 1
    best_profit_row = max(
        rows,
        key=lambda row: (row["profit"], row["max_filled_volume"], -row["submitted_qty"]),
    )
    best_fill_row = max(
        rows,
        key=lambda row: (row["max_filled_volume"], row["profit"], -row["submitted_qty"]),
    )

    print("=" * 92)
    print(title)
    print(
        "Highest profit        : "
        f"product={best_profit_row['product']}, buy={best_profit_row['buy_price']}, "
        f"submit={best_profit_row['submitted_qty']}, "
        f"clear={best_profit_row['clearing_price']}, fill={best_profit_row['max_filled_volume']}, "
        f"profit={best_profit_row['profit']:.2f}"
    )
    print(
        "Highest filled volume : "
        f"product={best_fill_row['product']}, buy={best_fill_row['buy_price']}, "
        f"submit={best_fill_row['submitted_qty']}, "
        f"clear={best_fill_row['clearing_price']}, fill={best_fill_row['max_filled_volume']}, "
        f"profit={best_fill_row['profit']:.2f}"
    )
    print("=" * 92)
    if multiple_products:
        print("product         | buy_price | clearing | submit_qty | max_fill | total_match | profit")
    else:
        print("buy_price | clearing | submit_qty | max_fill | total_match | profit")
    for row in rows:
        if multiple_products:
            print(
                f"{row['product']:<15} | {row['buy_price']:>9} | {row['clearing_price']:>8} | "
                f"{row['submitted_qty']:>10} | {row['max_filled_volume']:>8} | "
                f"{row['total_matched_volume']:>11} | {row['profit']:>7.2f}"
            )
        else:
            print(
                f"{row['buy_price']:>9} | {row['clearing_price']:>8} | {row['submitted_qty']:>10} | "
                f"{row['max_filled_volume']:>8} | {row['total_matched_volume']:>11} | {row['profit']:>7.2f}"
            )


def best_profit_row(rows: List[Dict]) -> Dict:
    return max(
        rows,
        key=lambda row: (row["profit"], row["max_filled_volume"], -row["submitted_qty"]),
    )


def print_all_scan_summary(best_rows: List[Dict]) -> None:
    combined_profit = sum(row["profit"] for row in best_rows)
    combined_fill = sum(row["max_filled_volume"] for row in best_rows)
    combined_cost = sum(row["auction_cost"] for row in best_rows)
    combined_revenue = sum(row["buyback_revenue"] for row in best_rows)
    combined_fees = sum(row["fees"] for row in best_rows)

    print("=" * 92)
    print("Combined highest-profit setup across ALL products")
    for row in best_rows:
        print(
            f"{row['product']}: buy={row['buy_price']}, submit={row['submitted_qty']}, "
            f"clear={row['clearing_price']}, fill={row['max_filled_volume']}, "
            f"profit={row['profit']:.2f}"
        )
    print("-" * 92)
    print(f"Total filled volume  : {combined_fill}")
    print(f"Total auction cost   : {combined_cost:.2f}")
    print(f"Total buyback revenue: {combined_revenue:.2f}")
    print(f"Total fees           : {combined_fees:.2f}")
    print(f"Combined profit      : {combined_profit:.2f}")
    print("=" * 92)


def main():
    print("Auction calculator")
    print("Running default scan for ALL products...")

    best_rows: List[Dict] = []
    for product in PRODUCT_NAMES:
        rows = scan_buy_price_opportunities(product=product)
        print()
        print_scan_results(rows, title=f"{product} opportunities")
        if rows:
            best_rows.append(best_profit_row(rows))

    if best_rows:
        print()
        print_all_scan_summary(best_rows)


if __name__ == "__main__":
    main()
