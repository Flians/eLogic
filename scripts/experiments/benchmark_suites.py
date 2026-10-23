"""Explicit, ordered benchmark coverage; custom subsets remain labelled subsets."""
EPFL = (
    "adder", "bar", "div", "hyp", "log2", "max", "multiplier", "sin", "sqrt", "square",
    "arbiter", "cavlc", "ctrl", "dec", "i2c", "int2float", "mem_ctrl", "priority", "router", "voter",
)
ISCAS = ("c17", "c432", "c499", "c880", "c1355", "c1908", "c2670")


def add_suite_arguments(parser):
    parser.add_argument("--suite", choices=["epfl", "iscas"], default="epfl")
    parser.add_argument("--benches", nargs="+", help="Explicit subset; default is the complete selected suite")


def resolve_suite(args):
    args.benches = args.benches or list(EPFL if args.suite == "epfl" else ISCAS)
