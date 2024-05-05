import networkx as nx
import subprocess
import re


def mig_cec(cpath: str, opath: str) -> bool:
    abc_command = f'cec {cpath} {opath}'
    cmd_elements = ['yosys-abc', '-c', abc_command]
    cec = False
    try:
        proc = subprocess.check_output(cmd_elements)
        # read results and extract information
        line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        ob = re.search(r'Networks are equivalent.', line)
        cec = ob is not None
    except Exception:
        print(f'ABC Error: {cmd_elements}')
    return cec


def del_buf(graph: nx.DiGraph, n: str):
    if n not in graph:
        return
    pres = list(graph.predecessors(n))
    assert len(pres) == 1
    pre = pres[0]
    sucs = set(graph.successors(n))
    graph.remove_node(n)
    for suc in sucs:
        if graph.has_edge(pre, suc):
            suc_pres = list(set(graph.predecessors(suc)) - {pre})
            assert len(suc_pres) == 1
            graph.remove_edge(suc_pres[0], suc)
            del_buf(graph, suc)
        else:
            graph.add_edge(pre, suc)


def process_not_buf(graph: nx.DiGraph):
    undriven = set()
    name_mapping = {}
    for node in graph.nodes:
        ntype = graph.nodes[node]["type"]
        if ntype == '~':
            pass
        elif ntype == 'buf':
            if graph.nodes[node]["output"]:
                pres = set(graph.predecessors(node))
                if pres:
                    assert len(pres) == 1
                    for pre in pres:
                        if graph.nodes[pre]["type"] == 'input':
                            continue
                        else:
                            undriven.add(node)
                            name_mapping[pre] = node
                            graph.nodes[pre]["output"] = True
                            graph.add_edges_from([(pre, suc) for suc in graph.successors(node)])
                else:
                    print(f'Warning: the type {ntype} of {node} is not driven.')
                    undriven.add(node)
            else:
                pres = set(graph.predecessors(node))
                if pres:
                    assert len(pres) == 1
                    for pre in pres:
                        graph.add_edges_from([(pre, suc) for suc in graph.successors(node)])
                else:
                    print(f'Warning: the type {ntype} of {node} is not driven.')
                    undriven.add(node)
    graph.remove_nodes_from(undriven)
    nx.relabel_nodes(graph, name_mapping, copy=False)


if __name__ == '__main__':
    K = 8
    benchmarks = ['adder', 'arbiter', 'bar', 'cavlc', 'c432', 'c499', 'c1355', 'c6288', 'ctrl', 'dec', 'div', 'i2c', 'int2float', 'max', 'multiplier', 'priority', 'sin', 'sqrt']
    for case in benchmarks:
        print(f'nohup python3 -u mig/main.py --K {K} --obj 0 --benchmark {case} > egg_depth_K{K}_{case}.log 2>&1 &')
    print()
    for case in benchmarks:
        print(f'nohup python3 -u mig/main.py --K {K} --obj 1 --benchmark {case} > egg_area_K{K}_{case}.log 2>&1 &')
