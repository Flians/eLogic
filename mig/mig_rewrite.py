import networkx as nx
import copy
from collections import defaultdict
from typing import Any, Dict, List, Set
from itertools import product
from eggexpr import graph_to_egg_expr, graph_from_egg_expr
import mig_egg


def add_edges(graph: nx.DiGraph, ebunch_to_add):
    for e in ebunch_to_add:
        ne = len(e)
        if ne == 3:
            u, v, dd = e
        elif ne == 2:
            u, v = e
            dd = {}
        else:
            raise ValueError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
        if graph.has_edge(u, v):
            if 'type' in graph.nodes[v] and graph.nodes[v]['type'] == 'M':
                sucs = set(graph.successors(v))
                graph.remove_node(v)
                add_edges(graph, ((u, suc) for suc in sucs))
                continue
            else:
                print(f"??? Edge ({u},{v}) exists.")
        if u not in graph._succ:
            if u is None:
                raise ValueError("None cannot be a node")
            graph._succ[u] = graph.adjlist_inner_dict_factory()
            graph._pred[u] = graph.adjlist_inner_dict_factory()
            graph._node[u] = graph.node_attr_dict_factory()
        if v not in graph._succ:
            if v is None:
                raise ValueError("None cannot be a node")
            graph._succ[v] = graph.adjlist_inner_dict_factory()
            graph._pred[v] = graph.adjlist_inner_dict_factory()
            graph._node[v] = graph.node_attr_dict_factory()
        datadict = graph._adj[u].get(v, graph.edge_attr_dict_factory())
        datadict.update(dd)
        graph._succ[u][v] = datadict
        graph._pred[v][u] = datadict
    nx._clear_cache(graph)


def extract_subgraph(graph: nx.DiGraph, cone: set, cut: set, root: str) -> nx.DiGraph:
    subgraph: nx.DiGraph = graph.subgraph(cone)
    new_subgraph = nx.DiGraph()
    new_subgraph.update(subgraph)
    remove_edges = []
    for n in cut:
        for pre in new_subgraph.predecessors(n):
            remove_edges.append((pre, n))
    new_subgraph.remove_edges_from(remove_edges)
    unloaded = [n for n in new_subgraph if new_subgraph.out_degree(n) == 0 and n != root]
    while unloaded:
        n = unloaded.pop()
        for fi in new_subgraph.predecessors(n):
            if n in cut:
                continue
            if new_subgraph.out_degree(fi) == 1:
                unloaded.append(fi)
        new_subgraph.remove_node(n)
    return new_subgraph


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


def kcuts_kcones_PIs_POs(graph: nx.DiGraph, K: int) -> tuple[Dict[Any, List[Set]], Dict[str, Set[int]]]:  # type: ignore
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
    computed: Dict[Any, List[Set]] = {}
    all_cones: Dict[str, Set[int]] = {}

    for n in nx.topological_sort(graph):
        it = graph.predecessors(n)
        pre = next(it, None)
        if pre is not None:
            cuts = copy.deepcopy(computed[pre])
            partial_cones: Dict[str, Set[int]] = {}
            for a_cut in cuts:
                la = ','.join(map(str, sorted(a_cut))) + f'|{pre}'
                lc = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                partial_cones[lc] = all_cones[la] | {n, pre}
            for pre2 in it:
                merged_cuts = []
                cuts2 = copy.deepcopy(computed[pre2])
                for a_cut, b_cut in product(cuts, cuts2):
                    merged_cut = a_cut | b_cut
                    if len(merged_cut) <= K:
                        merged_cuts.append(merged_cut)
                        la = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                        lb = ','.join(map(str, sorted(b_cut))) + f'|{pre2}'
                        lc = ','.join(map(str, sorted(merged_cut))) + f'|{n}'
                        partial_cones[lc] = partial_cones[la] | all_cones[lb] | {pre, pre2}
                cuts = merged_cuts
                pre = pre2
            for a_cut in cuts:
                lc = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                all_cones[lc] = partial_cones[lc]
            cuts += [{n}]
        else:
            cuts = [{n}]
        # add cuts
        computed[n] = cuts
        all_cones[f'{n}|{n}'] = {n}

    return computed, all_cones


def update_kcuts_kcones(graph: nx.DiGraph, K: int, starts: set = {}, all_cuts: Dict[Any, List[Set]] = {}, all_cones: Dict[str, set[int]] = {}, fanins: Dict[str, set[str]] = {}) -> tuple[Dict[Any, List[Set]], Dict[str, Set[int]]]:  # type: ignore
    """
    Update all K-cuts from PIs to POs.

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
    stop_times = defaultdict(lambda: 0)
    for n in nx.topological_sort(graph):
        if starts and n in starts:
            flag = True
        if n in all_cuts and ((starts and not flag) or stop_times[n] == graph.in_degree(n)):
            if fanins:
                fanins[n] = {n} if graph.nodes[n]['type'] == 'M' else set()
                for pre in graph.predecessors(n):
                    fanins[n] |= fanins[pre]
            for suc in graph.successors(n):
                stop_times[suc] += 1
            continue
        it = graph.predecessors(n)
        pre = next(it, None)
        is_changed = False
        if pre is not None:
            cuts = copy.deepcopy(all_cuts[pre])
            partial_cones: Dict[str, Set[int]] = {}
            for a_cut in cuts:
                la = ','.join(map(str, sorted(a_cut))) + f'|{pre}'
                lc = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                partial_cones[lc] = all_cones[la] | {n, pre}
            if fanins:
                fanins[n] = {n} if graph.nodes[n]['type'] == 'M' else set()
                fanins[n] |= fanins[pre]
            for pre2 in it:
                merged_cuts = []
                if fanins:
                    fanins[n] |= fanins[pre2]
                cuts2 = copy.deepcopy(all_cuts[pre2])
                for a_cut, b_cut in product(cuts, cuts2):
                    merged_cut = a_cut | b_cut
                    if len(merged_cut) <= K:
                        merged_cuts.append(merged_cut)
                        la = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                        lb = ','.join(map(str, sorted(b_cut))) + f'|{pre2}'
                        lc = ','.join(map(str, sorted(merged_cut))) + f'|{n}'
                        partial_cones[lc] = partial_cones[la] | all_cones[lb] | {pre, pre2}
                cuts = merged_cuts
                pre = pre2
            for a_cut in cuts:
                lc = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                if not is_changed and (lc not in all_cones or all_cones[lc] != partial_cones[lc]):
                    is_changed = True
                all_cones[lc] = partial_cones[lc]
            cuts += [{n}]
        else:
            cuts = [{n}]
        # update cuts
        if is_changed or n not in all_cuts or len(all_cuts[n]) != len(cuts):
            all_cuts[n] = cuts
        else:
            for suc in graph.successors(n):
                stop_times[suc] += 1
        all_cones[f'{n}|{n}'] = {n}


def rewrite_dp(graph: nx.DiGraph, K: int = 8, obj_area=False):
    all_cuts, all_cones = kcuts_kcones_PIs_POs(graph, K=K)
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
            if graph.in_degree(n) != 3:
                pass
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
            update_kcuts_kcones(graph, K, starts=pt[last_n][0], all_cuts=all_cuts, all_cones=all_cones, fanins=fanins)
        cuts = all_cuts[n].copy()
        for cut in cuts:
            lc = ','.join(map(str, sorted(cut))) + f'|{n}'
            cone = all_cones[lc].copy()
            nonleaves = cone - cut
            if len(nonleaves) < 3 or (cone - set(graph.nodes)) or len({nn for nn in nonleaves if graph.nodes[nn]["type"] == 'M'}) < 3:
                continue  # {y} = M(a,b,c) covered by {y,a,b,c}
            else:
                subgraph: nx.DiGraph = extract_subgraph(graph, cone, cut, n)
                '''
                # check if inner nodes are used out of the subgraph
                flag = False
                for inn in subgraph:
                    if subgraph.nodes[inn]['type'] == 'M' and subgraph.in_degree(inn) not in [0, 3]:
                        raise Exception(f"Maj node {inn} has less than 3 inputs")
                    if inn != n and subgraph.out_degree(inn) != graph.out_degree(inn) and graph.in_degree(inn) != 0:
                        flag = True
                        break
                if flag:
                    continue
                '''
                expr = graph_to_egg_expr(subgraph, cut)
                if not expr:
                    print(f"??? {n} {cut} {cone}")
                    continue
                expr_opt, inital_cost, final_cost = mig_egg.simplify(expr[0])  # type: ignore
                if obj_area:
                    if inital_cost[1] < final_cost[1] or (inital_cost[1] == final_cost[1] and inital_cost[0] <= final_cost[0]):
                        continue
                else:
                    if inital_cost[0] < final_cost[0] or (inital_cost[0] == final_cost[0] and inital_cost[1] <= final_cost[1]):
                        continue
                print(f"Simplified {expr} with cost {inital_cost} to {expr_opt} with cost {final_cost}")
                subgraph_opt = graph_from_egg_expr(expr_opt)
                distances = distances_from_PIs_PO(subgraph_opt)
                cost = [0, set()]
                for pre in cut:
                    cost[0] = max(distances[pre] + dp[pre][0], cost[0])
                    cost[1] |= fanins[pre]
                cost[1] = len(cost[1]) + final_cost[1]
                if obj_area:
                    if cost[1] < dp[n][1] or (cost[1] == dp[n][1] and cost[0] < dp[n][0]):
                        dp[n] = cost
                        pt[n] = (cut.copy(), expr_opt)
                else:
                    if cost < dp[n]:
                        dp[n] = cost
                        pt[n] = (cut.copy(), expr_opt)
        if pt[n][1]:
            cut = pt[n][0].copy()
            subgraph_opt = graph_from_egg_expr(pt[n][1])
            # add sub circuit
            mapping = {}
            new_root = pt[n][1]
            sc_io = cut | {n}
            for sn, attr in subgraph_opt.nodes.data():
                if sn in sc_io:
                    pass
                elif sn in ['true', 'false']:
                    sc_io |= {sn}
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
            add_edges(graph, g.edges.data())
            # remove nodes in original cone
            lc = ','.join(map(str, sorted(cut))) + f'|{n}'
            cone = all_cones[lc].copy()
            subgraph: nx.DiGraph = extract_subgraph(graph, cone, cut, n)
            removed = set()
            unloaded = [n]
            while unloaded:
                cur = unloaded.pop()
                for pre in subgraph.predecessors(cur):
                    if pre in sc_io:
                        continue
                    if graph.out_degree(pre) == 1:
                        unloaded.append(pre)
                if cur == n:
                    sucs = set(graph.successors(cur))
                    add_edges(graph, ((new_root, suc) for suc in sucs))
                graph.remove_node(cur)
                if cur == n and new_root not in ['true', 'false']:
                    nx.relabel_nodes(graph, {new_root: cur}, copy=False)
                    removed.add(new_root)
                else:
                    removed.add(cur)
            # remove duplicates
            new_nodes = set(mapping.values()) | set(subgraph.nodes) - removed
            subgraph: nx.DiGraph = graph.subgraph(new_nodes)
            record_ins = {}
            for cur in nx.topological_sort(subgraph):
                cur_ins = ','.join(map(str, sorted(graph.predecessors(cur))))
                if not cur_ins:
                    continue
                cur_ins += '|' + graph.nodes[cur]['type']
                if cur_ins in record_ins:
                    if graph.nodes[cur]['output']:
                        old = record_ins[cur_ins]
                        record_ins[cur_ins] = cur
                        sucs = set(graph.successors(old))
                        add_edges(graph, ((cur, suc) for suc in sucs))
                        graph.remove_node(old)
                    else:
                        sucs = set(graph.successors(cur))
                        add_edges(graph, ((record_ins[cur_ins], suc) for suc in sucs))
                        graph.remove_node(cur)
                else:
                    record_ins[cur_ins] = cur
        last_n = n
    pass
