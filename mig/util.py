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
                            sucs = set(graph.successors(node))
                            graph.add_edges_from([(pre, suc) for suc in sucs])
                else:
                    print(f'Warning: the type {ntype} of {node} is not driven.')
                    undriven.add(node)
            else:
                pass
    graph.remove_nodes_from(undriven)
    nx.relabel_nodes(graph, name_mapping, copy=False)
