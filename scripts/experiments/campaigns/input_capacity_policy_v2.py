"""Input-size capacity v2: include possible complemented PIs as well as ANDs.

The v1 policy and its experiments remain frozen. All admitted nodes count; no refunds.
"""
from pathlib import Path

def input_capacity(path, node_extra=20000, pool_extra=0):
    if node_extra<0 or pool_extra<0:raise ValueError("negative capacity")
    with Path(path).open("rb")as f:fields=f.readline(256).split()
    if len(fields)!=6 or fields[0]not in (b"aig",b"aag"):raise ValueError("unsupported AIGER header")
    maximum,inputs,latches,outputs,ands=map(int,fields[1:])
    if min(maximum,inputs,latches,outputs,ands)<0 or latches or maximum<inputs+ands:raise ValueError("combinational AIGER required")
    # One constant, every PI/live AND, and at most one NOT per variable;
    # an additional constant phase is included. This is an upper, not a lower bound.
    base_upper=2*(inputs+ands)+2
    return dict(policy="input-upper-plus-search-v2",aiger_header=dict(maximum=maximum,inputs=inputs,latches=latches,outputs=outputs,ands=ands),base_enode_upper=base_upper,node_extra=node_extra,node_limit=max(20000,base_upper+node_extra),pool_extra=pool_extra,pool_limit=max(100000,base_upper+pool_extra)if pool_extra else 100000,refund_on_rebuild=False,baseline_algorithms_changed=False)
