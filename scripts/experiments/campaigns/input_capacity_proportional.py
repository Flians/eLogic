"""Uniform input-scaled cumulative e-node capacity, without admission refunds.

The primitive input bound includes both phases of every PI and AND, plus two
constants. Richer optional helpers are not free: their admissions consume the
same total ledger. Physical candidates have a separate retained-bank limit.
"""
from fractions import Fraction

from input_capacity_policy_v2 import input_capacity as primitive_capacity


def input_capacity(path, search_ratio="1", pool_extra=0):
    try:
        ratio = Fraction(str(search_ratio))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("search ratio must be finite and nonnegative") from error
    if ratio < 0:
        raise ValueError("search ratio must be finite and nonnegative")
    if pool_extra != 0:
        raise ValueError("this policy keeps the optional retained bank at 100000")
    result = primitive_capacity(path, node_extra=0, pool_extra=0)
    base = result["base_enode_upper"]
    extra = (base * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    result.update(
        policy="primitive-input-plus-proportional-search-v1",
        search_ratio=str(ratio),
        search_ratio_numerator=ratio.numerator,
        search_ratio_denominator=ratio.denominator,
        search_allowance=extra,
        node_extra=extra,
        node_limit=base + extra,
        fixed_node_floor=None,
        input_bound_scope="primitive PI/AND/NOT/two constants; optional helpers consume the total ledger",
        proportional_search_only=True,
    )
    return result
