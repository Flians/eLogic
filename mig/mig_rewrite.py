import networkx as nx
import copy
from collections import defaultdict
from typing import Any, Dict, List, Set
from itertools import product
from eggexpr import graph_to_egg_expr, graph_from_egg_expr
import mig_egg


def add_edges(graph: nx.DiGraph, ebunch_to_add) -> Dict:
    removed = {}
    for e in ebunch_to_add:
        ne = len(e)
        if ne == 3:
            u, v, dd = e
        elif ne == 2:
            u, v = e
            dd = {}
        else:
            raise ValueError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
        if v in removed:
            continue
        if u in removed:
            u = removed[u]
        if graph.has_edge(u, v):
            if 'type' in graph.nodes[v] and graph.nodes[v]['type'] == 'M':
                sucs = set(graph.successors(v))
                graph.remove_node(v)
                removed[v] = u
                removed.update(add_edges(graph, ((u, suc) for suc in sucs)))
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
        if graph.has_edge('false', v) and graph.has_edge('true', v):
            sucs = set(graph.successors(v))
            third_in = set(graph.predecessors(v)) - {'false', 'true'}
            assert len(third_in) == 1
            graph.remove_node(v)
            removed[v] = list(third_in)[0]
            removed.update(add_edges(graph, ((removed[v], suc) for suc in sucs)))
    return removed


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
            if fi in cut:
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


def kcuts_kcones_PIs_POs(graph: nx.DiGraph, K: int) -> tuple[Dict[Any, List[Set]], Dict[str, Set[int]], Dict[str, List], Dict[str, Set], Set, Set, Dict[str, int]]:  # type: ignore
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
    all_cuts: Dict[Any, List[Set]] = {}
    all_cones: Dict[str, Set[int]] = {}
    dp: Dict[str, list] = defaultdict(lambda: [0, 0])
    fanins = defaultdict(set)  # record majorities
    inputs = set()
    outputs = set()
    indegree_map: Dict[str, int] = {}

    for n in nx.topological_sort(graph):
        depth = 0
        it = graph.predecessors(n)
        pre = next(it, None)
        if pre is not None:
            fanins[n] |= fanins[pre]
            depth = max(depth, dp[pre][0])
            cuts = copy.deepcopy(all_cuts[pre])
            partial_cones: Dict[str, Set[int]] = {}
            for a_cut in cuts:
                la = ','.join(map(str, sorted(a_cut))) + f'|{pre}'
                lc = ','.join(map(str, sorted(a_cut))) + f'|{n}'
                partial_cones[lc] = all_cones[la] | {n, pre}
            for pre2 in it:
                fanins[n] |= fanins[pre]
                depth = max(depth, dp[pre][0])
                cuts2 = copy.deepcopy(all_cuts[pre2])
                merged_cuts = []
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
        if graph.nodes[n]['type'] == 'M':
            assert graph.in_degree(n) == 3
            fanins[n] |= {n}
            depth += 1
        # add cuts
        all_cuts[n] = cuts
        all_cones[f'{n}|{n}'] = {n}
        dp[n] = [depth, len(fanins[n])]
        ind = graph.in_degree(n)
        if ind == 0:
            inputs.add(n)
        else:
            indegree_map[n] = ind
        if graph.out_degree(n) == 0:
            outputs.add(n)

    return all_cuts, all_cones, dp, fanins, inputs, outputs, indegree_map


def update_kcuts_kcones(graph: nx.DiGraph, K: int, starts: set = {}, all_cuts: Dict[Any, List[Set]] = {}, all_cones: Dict[str, set[int]] = {}, fanins: Dict[str, set[str]] = {}, dp: Dict[str, list] = {}) -> tuple[Dict[Any, List[Set]], Dict[str, Set[int]]]:  # type: ignore
    flag = False
    stop_times = defaultdict(lambda: 0)
    for n in nx.topological_sort(graph):
        if graph.nodes[n]['type'] == 'M' and graph.in_degree(n) != 3:
            pass
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
    # initialization
    all_cuts, all_cones, dp, fanins, inputs, outputs, indegree_map = kcuts_kcones_PIs_POs(graph, K=K)
    all_nodes_topo = list(nx.topological_sort(graph))

    best_cut = set()
    for n in all_nodes_topo:
        if n in inputs or n not in graph:
            continue
        # update K-cuts, all_cones, dp, fanins
        if best_cut:
            # update_kcuts_kcones(graph, K, starts=best_cut, all_cuts=all_cuts, all_cones=all_cones, fanins=fanins, dp=dp)  # type: ignore
            all_cuts, all_cones, dp, fanins, _, _, _ = kcuts_kcones_PIs_POs(graph, K=K)
        cuts = copy.deepcopy(all_cuts[n])
        best_cut = set()
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
                if (obj_area and (cost[1] < dp[n][1] or (cost[1] == dp[n][1] and cost[0] < dp[n][0]))) or (obj_area is False and cost < dp[n]):
                    cost[1] += len({nn for nn in subgraph if nn != n and graph.nodes[nn]['type'] == 'M' and graph.out_degree(nn) != subgraph.out_degree(nn)})
                    if (obj_area and (cost[1] < dp[n][1] or (cost[1] == dp[n][1] and cost[0] < dp[n][0]))) or (obj_area is False and cost < dp[n]):
                        dp[n] = cost
                        best_cut = cut.copy()
                        best_expr = expr_opt
        if best_cut:
            subgraph_opt = graph_from_egg_expr(best_expr)
            # add sub circuit
            mapping = {}
            new_root = best_expr
            sc_io = best_cut | {n}
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
            lc = ','.join(map(str, sorted(best_cut))) + f'|{n}'
            cone = all_cones[lc].copy()
            subgraph: nx.DiGraph = extract_subgraph(graph, cone, best_cut, n)
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
                    graph.remove_node(cur)
                    add_edges(graph, ((new_root, suc) for suc in sucs))
                else:
                    graph.remove_node(cur)
                del all_cuts[cur]
                del fanins[cur]
                if cur == n and new_root not in sc_io:
                    nx.relabel_nodes(graph, {new_root: cur}, copy=False)
                    cur = new_root
                removed.add(cur)
            # remove duplicates
            all_nodes = set(mapping.values()) | set(subgraph.nodes) - removed
            subgraph: nx.DiGraph = graph.subgraph(all_nodes)
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
                    else:
                        old = cur
                        cur = record_ins[cur_ins]
                    sucs = set(graph.successors(old))
                    graph.remove_node(old)
                    add_edges(graph, ((cur, suc) for suc in sucs))
                else:
                    record_ins[cur_ins] = cur
