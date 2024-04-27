import networkx as nx
import copy
from collections import defaultdict
from typing import Any, Dict, List, Set
from itertools import product
from eggexpr import graph_to_egg_expr, graph_from_egg_expr
import mig_egg


def transitive_fanin(graph: nx.DiGraph, ns, ntypes=None):
    """
    Compute the transitive fanin of a node.

    Parameters
    ----------
    ns : str or iterable of str
            Node(s) to compute transitive fanin for.

    Returns
    -------
    set of str
            Nodes in transitive fanin.

    """
    if isinstance(ns, str):
        ns = [ns]
    gates = set()
    for n in ns:
        gates |= nx.ancestors(graph, n)
    if ntypes:
        if isinstance(ntypes, str):
            ntypes = [ntypes]
        gates = {n for n in gates if graph.nodes[n]["type"] in ntypes}
    return gates


def distances_from_PIs_PO(graph: nx.DiGraph) -> dict:
    distances = defaultdict(lambda: 0)
    for cur in reversed(list(nx.topological_sort(graph))):
        gap = graph.nodes[cur]["type"] == 'M'
        dist = distances[cur] + gap
        for pre in graph.predecessors(cur):
            distances[pre] = max(distances[pre], dist)
    return distances


def kcuts_kcones_PIs_POs(graph: nx.DiGraph, K: int, starts: set = {}, end: str = None, computed: Dict[Any, List[Set]] = {}, all_cones: Dict[str, set[int]] = {}, fanins: Dict[str, set[str]] = {}) -> tuple[Dict[Any, List[Set]], Dict[str, Set[int]]]:  # type: ignore
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
            if fanins:
                if graph.nodes[n]['type'] == 'M':
                    fanins[n] = {n}
                else:
                    fanins[n] = set()
                fanins[n] |= fanins[pre]
            for pre2 in it:
                merged_cuts = []
                if fanins:
                    fanins[n] |= fanins[pre2]
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
        if end and n == end:
            break

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
    dp: Dict[str, list] = defaultdict(lambda: [0, 0])
    pt: Dict[str, tuple[set, str]] = {}
    inputs = set()
    outputs = set()
    fanins = defaultdict(set)
    # initialization
    for n in nx.topological_sort(graph):
        cost = [0, 0]
        for pre in graph.predecessors(n):
            fanins[n] |= fanins[pre]
            cost[0] = max(cost[0], dp[pre][0])
        if graph.nodes[n]['type'] == 'M':
            fanins[n] |= {n}
            dp[n][0] = 1 + cost[0]
            dp[n][1] = len(fanins[n])
        else:
            dp[n][0] = 0 + cost[0]
            dp[n][1] = len(fanins[n])
        pt[n] = ({n}, '')
        if graph.in_degree(n) == 0:
            inputs.add(n)
        if graph.out_degree(n) == 0:
            outputs.add(n)

    last_n = None
    for n in nx.topological_sort(graph):
        if n in inputs:
            continue
        # update K-cuts
        if last_n is not None:
            all_cuts, all_cones = kcuts_kcones_PIs_POs(graph, 8, pt[last_n][0], n, all_cuts, all_cones, fanins)
        cuts = all_cuts[n].copy()
        for cut in cuts:
            lc = ','.join(map(str, sorted(cut | {n})))
            cone = all_cones[lc].copy()
            nonleaves = cone - cut
            if len(nonleaves) < 3 or len({n for n in nonleaves if graph.nodes[n]["type"] == 'M'}) < 3:
                continue  # {y} = M(a,b,c) covered by {y,a,b,c}
            else:
                subgraph: nx.DiGraph = graph.subgraph(cone)
                # check if inner nodes are used out of the subgraph
                flag = False
                for inn in nonleaves:
                    if subgraph.nodes[inn]['type'] == 'M' and subgraph.in_degree(inn) != 3:
                        raise Exception(f"Maj node {inn} has less than 3 inputs")
                    subgraph.successors(inn)
                    if inn != n and subgraph.out_degree(inn) != graph.out_degree(inn):
                        flag = True
                        break
                if flag:
                    continue
                expr = graph_to_egg_expr(subgraph, cut)
                expr_opt, inital_cost, final_cost = mig_egg.simplify(expr[0])  # type: ignore
                if inital_cost[0] < final_cost[0] or inital_cost[1] <= final_cost[1]:
                    continue
                subgraph_opt = graph_from_egg_expr(expr_opt)
                distances = distances_from_PIs_PO(subgraph_opt)
                cost = [0, set()]
                for pre in cut:
                    cost[0] = max(distances[pre] + dp[pre][0], cost[0])
                    cost[1] |= fanins[pre]
                cost[1] = len(cost[1]) + final_cost[1]
                if cost < dp[n]:
                    dp[n] = cost
                    pt[n] = (cut, expr_opt)
        if pt[n][1]:
            subgraph_opt = graph_from_egg_expr(pt[n][1])
            # add sub circuit
            mapping = {}
            new_root = ''
            sc_io = pt[n][0] | {n}
            for sn, attr in subgraph_opt.nodes.data():
                if sn in sc_io:
                    pass
                else:
                    # check for name overlaps
                    if f"{n}_{sn}" in graph.nodes:
                        raise ValueError(f"name {sn} overlaps with {lc} subcircuit.")
                    if subgraph_opt.nodes[sn]['output']:
                        new_root = f"{n}_{sn}"
                        graph.add_node(f"{n}_{sn}", **graph.nodes[n])
                        graph.nodes[f"{n}_{sn}"]['type'] = subgraph_opt.nodes[sn]['type']
                    else:
                        graph.add_node(f"{n}_{sn}", **attr)
                    mapping[sn] = f"{n}_{sn}"
            g = nx.relabel_nodes(subgraph_opt, mapping)
            graph.add_edges_from(g.edges.data())
            # remove nodes in original cone
            lc = ','.join(map(str, sorted(sc_io)))
            cone = all_cones[lc].copy()
            all_cones[lc] = set(mapping.values()) | sc_io
            subgraph = graph.subgraph(cone)
            unloaded = [n]
            while unloaded:
                cur = unloaded.pop()
                for pre in subgraph.predecessors(cur):
                    if pre in sc_io:
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
                cur_ins += '|' + graph.nodes[cur]['type']
                if cur_ins in record_ins:
                    graph.add_edges_from(((record_ins[cur_ins], suc) for suc in graph.successors(cur)))
                    graph.remove_node(cur)
                else:
                    record_ins[cur_ins] = cur
        last_n = n
    pass
