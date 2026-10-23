"""Preserve AIGER scalar port names, including EPFL bit-index symbols."""
import re


def identifier(name):
    if not name or any(c.isspace() or c in "\\;\"" for c in name):
        raise ValueError("unsupported port identifier")
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name) else "\\" + name + " "


def escape_indexed_scalars(text):
    # The mockturtle writer emits scalar AIG names, not Verilog buses.
    # Rewrite only the known indexed-name shape; escaped tokens remain intact.
    return re.sub(r"\\[^\s]+\s|(?<![\w$])([A-Za-z_][A-Za-z0-9_$]*(?:\[[0-9]+\])+)",
                  lambda m: identifier(m[1]) if m[1] else m[0], text)
