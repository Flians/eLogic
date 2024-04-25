import mig_egg
import re
import networkx as nx


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.name: str = ''


def parse_nested_parentheses(text):
    stack = []
    root = Node('')
    stack.append(root)

    flag = 0
    for char in text:
        if char == '(':
            flag = 1
        elif char == ')':
            stack.pop()
        elif char == ' ':
            if flag != 2:
                stack.pop()
            node = Node('')
            stack[-1].children.append(node)
            stack.append(node)
            flag = 3
        else:
            if flag == 1 and char in ['M', '~']:
                stack[-1].value = char
                flag = 2
            else:
                stack[-1].value += char

    return root


def graph_from_egg_expr(egg_seq):
    graph = nx.DiGraph()
    nid = 0
    root = parse_nested_parentheses(egg_seq)
    root.name = f'g{nid}' if root.value in ['M', '~'] else root.value
    graph.add_node(root.name)
    cur_nodes = [root]
    while cur_nodes:
        cur = cur_nodes.pop(0)
        if cur.value in ['M', '~']:
            graph.nodes[cur.name]["type"] = cur.value
        else:
            graph.nodes[cur.name]["type"] = 'input'
        graph.nodes[cur.name]["output"] = nid == 0
        for child in cur.children:
            nid += 1
            cur_nodes.append(child)
            child.name = f'g{nid}' if child.value in ['M', '~'] else child.value
            graph.add_node(child.name)
            graph.add_edges_from([(child.name, cur.name)])
    return graph


def graph_to_egg_expr(graph: nx.DiGraph, inputs: set = None) -> list[str]:
    exprs: dict[str, str] = {}
    outputs = set()
    for node in nx.topological_sort(graph):
        pres = set(graph.predecessors(node))
        if pres and (not inputs or node not in inputs):
            exprs[node] = f"({graph.nodes[node]['type']} {' '.join([exprs[din] for din in pres])})"
        else:
            exprs[node] = node
        if graph.out_degree(node) == 0:
            outputs.add(node)
    return [exprs[po] for po in outputs]


if __name__ == '__main__':
    egg_seq = "(M x3 (M x3 x4 (M x5 x6 x7)) x1)"
    res, inital_cost, best_cost = mig_egg.simplify(egg_seq)

    pattern = r"^\(M (\(?[\w ]+\)?) (\(?[\w ]+\)?) (\(?[\w ]+\)?)\)"
    match = re.search(pattern, res)

    graph0 = graph_from_egg_expr(egg_seq)
    graph = graph_from_egg_expr(res)

    res0 = graph_to_egg_expr(graph0)
    res2 = graph_to_egg_expr(graph)
    print(res0)
    print(res2)

    # import matplotlib.pyplot as plt
    # nx.draw(graph, with_labels=True)
    # plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')
    # plt.show()
    nx.drawing.nx_agraph.write_dot(graph, 'plotgraph.dot')

    # pip install py-aiger
    import aiger

    # Parser for ascii AIGER format.
    aig1 = aiger.load('tests/adder.aag')
    aig2 = aiger.load('tests/adder1.aag')
    aig3 = aig1 >> aig2
    aig4 = aig1 | aig2
    pass
