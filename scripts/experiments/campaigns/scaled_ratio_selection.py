"""Predeclared whole-suite choice; never pick a different rho per circuit."""
import math
from fractions import Fraction


def geometric(values):
    if not values:
        return None
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid metric")
    if 0 in values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def select(rows, ratios):
    groups = {r: [x for x in rows if x["ratio"] == r] for r in ratios}
    verified = {r: {(x["bench"], x["k"]): x for x in xs if x["status"] == "ok"}
                for r, xs in groups.items()}
    common = set.intersection(*(set(points) for points in verified.values()))
    summary = {}
    for ratio, xs in groups.items():
        summary[ratio] = dict(verified=len(verified[ratio]), common_verified_cases=len(common),
            delay_geomean=geometric([verified[ratio][key]["delay"] for key in common]),
            area_geomean=geometric([verified[ratio][key]["area"] for key in common]),
            mapping_seconds_geomean=geometric([x["mapping_secs"] for x in xs]))
    eligible = [r for r in ratios if summary[r]["verified"] == max(x["verified"] for x in summary.values())]
    for metric in (["delay_geomean", "area_geomean"] if common else []):
        best = min(summary[r][metric] for r in eligible)
        eligible = [r for r in eligible if summary[r][metric] <= best * (1 + 1e-6)]
    return min(eligible, key=Fraction), summary, common
