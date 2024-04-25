import networkx as nx
import copy
from collections import defaultdict
from typing import Any, Dict, List, Set
from itertools import product
from eggexpr import graph_to_egg_expr, graph_from_egg_expr
import mig_egg


def kcuts_kcones_PIs_POs(graph: nx.DiGraph, K: int, starts: set = {}, computed: Dict[Any, List[Set]] = {}, all_cones: Dict[str, set[int]] = {}) -> tuple[Dict[Any, List[Set]], Dict[str, Set[int]]]:  # type: ignore
    """
    Generate all K-cuts from PIs to POs.

    Parameters
    ----------
    K : int
            Maximum cut width.

    Returns
    -------
    iter of str
            K-cuts.

    """
    flag = False
    for n in nx.topological_sort(graph):
        if starts and n in starts:
            flag = True
        if starts and not flag:
            continue
        if starts and flag and n in computed:
            computed[n].clear()
        it = graph.predecessors(n)
        pre = next(it, None)
        if pre is not None:
            cuts = copy.deepcopy(computed[pre])
            partial_cones: Dict[str, Set[int]] = {}
            for a_cut in cuts:
                la = ','.join(map(str, sorted(a_cut | {pre})))
                lc = ','.join(map(str, sorted(a_cut | {n})))
                partial_cones[lc] = all_cones[la] | {n, pre}
            for pre2 in it:
                merged_cuts = []
                cuts2 = copy.deepcopy(computed[pre2])
                for a_cut, b_cut in product(cuts, cuts2):
                    merged_cut = a_cut | b_cut
                    if len(merged_cut) <= K:
                        merged_cuts.append(merged_cut)
                        la = ','.join(map(str, sorted(a_cut | {n})))
                        lb = ','.join(map(str, sorted(b_cut | {pre2})))
                        lc = ','.join(map(str, sorted(merged_cut | {n})))
                        partial_cones[lc] = partial_cones[la] | all_cones[lb] | {pre, pre2}
                cuts = merged_cuts
                pre = pre2
            for a_cut in cuts:
                lc = ','.join(map(str, sorted(a_cut | {n})))
                all_cones[lc] = partial_cones[lc]
            cuts += [{n}]
        else:
            cuts = [{n}]
        # add cuts
        computed[n] = cuts
        all_cones[str(n)] = {n}

    return computed, all_cones


def rewrite_dp2(graph: nx.DiGraph, K: int):
    all_cuts, all_cones = kcuts_kcones_PIs_POs(graph, K=8)
    dp: Dict[str, tuple] = defaultdict(lambda: (0, 0, 0))
    pt: Dict[str, tuple[set, str]] = {}
    inputs = set()
    outputs = set()
    # initialization
    for n in nx.topological_sort(graph):
        cost = (1, 1, 0) if graph.nodes[n]['type'] == 'M' else ((0, 0, 1) if graph.nodes[n]['type'] == '~' else (0, 0, 0))
        for pre in graph.predecessors(n):
            cost = tuple(map(sum, zip(cost, dp[pre])))
        dp[n] = cost
        pt[n] = ({n}, n)
        if graph.in_degree(n) == 0:
            inputs.add(n)
        if graph.out_degree(n) == 0:
            outputs.add(n)

    for n in nx.topological_sort(graph):
        if n in inputs:
            continue
        cuts = all_cuts[n]
        for cut in cuts:
            lc = ','.join(map(str, sorted(cut | {n})))
            cone = all_cones[lc]
            if len(cone - inputs) <= 1:
                continue  # {y} = M(a,b,c) covered by {y,a,b,c}
            else:
                subgraph = graph.subgraph(cone)
                expr = graph_to_egg_expr(subgraph, cut)
                expr_opt, inital_cost, cost = mig_egg.simplify(expr[0])
                for pre in cut:
                    cost = tuple(map(sum, zip(cost, dp[pre])))
                if cost < dp[n]:
                    dp[n] = cost
                    pt[n] = (cut, expr_opt)
    print(dp[n], pt[n])


def rewrite_dp(graph: nx.DiGraph, K: int):
    all_cuts, all_cones = kcuts_kcones_PIs_POs(graph, K=8)
    dp: Dict[str, tuple] = defaultdict(lambda: (0, 0, 0))
    pt: Dict[str, tuple[set, str]] = {}
    inputs = set()
    outputs = set()
    # initialization
    for n in nx.topological_sort(graph):
        cost = (1, 1, 0) if graph.nodes[n]['type'] == 'M' else ((0, 0, 1) if graph.nodes[n]['type'] == '~' else (0, 0, 0))
        for pre in graph.predecessors(n):
            cost = tuple(map(sum, zip(cost, dp[pre])))
        dp[n] = cost
        pt[n] = ({n}, '')
        if graph.in_degree(n) == 0:
            inputs.add(n)
        if graph.out_degree(n) == 0:
            outputs.add(n)

    for n in nx.topological_sort(graph):
        if n in inputs:
            continue
        cuts = all_cuts[n].copy()
        for cut in cuts:
            lc = ','.join(map(str, sorted(cut | {n})))
            cone = all_cones[lc].copy()
            if len(cone - cut) < 3:
                continue  # {y} = M(a,b,c) covered by {y,a,b,c}
            else:
                subgraph = graph.subgraph(cone)
                expr = graph_to_egg_expr(subgraph, cut)
                expr_opt, inital_cost, cost = mig_egg.simplify(expr[0])
                for pre in cut:
                    cost = tuple(map(sum, zip(cost, dp[pre])))
                if cost < dp[n]:
                    dp[n] = cost
                    pt[n] = (cut, expr_opt)
        if pt[n][1]:
            subgraph_opt = graph_from_egg_expr(pt[n][1])
            # check for name overlaps
            mapping = {}
            new_root = ''
            sc_io = pt[n][0] | {n}
            for sn, attr in subgraph_opt.nodes.data():
                if sn in sc_io:
                    pass
                else:
                    if f"{n}_{sn}" in graph.nodes:
                        raise ValueError(f"name {sn} overlaps with {lc} subcircuit.")
                    if subgraph_opt.nodes[sn]['output']:
                        new_root = f"{n}_{sn}"
                        graph.add_node(f"{n}_{sn}", **graph.nodes[n])
                    else:
                        graph.add_node(f"{n}_{sn}", **attr)
                    mapping[sn] = f"{n}_{sn}"
            # add sub circuit
            g = nx.relabel_nodes(subgraph_opt, mapping)
            graph.add_edges_from(g.edges.data())
            # remove nodes in original cone
            lc = ','.join(map(str, sorted(sc_io)))
            cone = all_cones[lc].copy()
            subgraph = graph.subgraph(cone)
            unloaded = [n]
            while unloaded:
                cur = unloaded.pop()
                for pre in subgraph.predecessors(cur):
                    if cur in sc_io:
                        continue
                    if graph.out_degree(pre) == 1:
                        unloaded.append(pre)
                if cur == n:
                    graph.add_edges_from(((new_root, suc) for suc in graph.successors(cur)))
                graph.remove_node(cur)
                if cur == n:
                    nx.relabel_nodes(graph, {new_root: cur}, copy=False)
            # remove duplicates
            new_nodes = set(mapping.values()) - {new_root}
            subgraph = graph.subgraph(cone | new_nodes)
            record_ins = {}
            for cur in nx.topological_sort(subgraph):
                cur_ins = ','.join(map(str, sorted(subgraph.predecessors(cur))))
                if not cur_ins:
                    continue
                if cur_ins in record_ins:
                    if graph.nodes[cur]['type'] != graph.nodes[record_ins[cur_ins]]['type']:
                        continue
                    graph.add_edges_from(((record_ins[cur_ins], suc) for suc in graph.successors(cur)))
                    graph.remove_node(cur)
                else:
                    record_ins[cur_ins] = cur
            # update K-cuts
            all_cuts, all_cones = kcuts_kcones_PIs_POs(graph, 8, pt[n][0], all_cuts, all_cones)
    pass
