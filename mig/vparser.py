import networkx as nx
import circuitgraph as cg


def parser(vpath):
    const1 = 'CONST1'
    cir_graph = cg.from_file(vpath)
    cir_graph.add(const1, '1')
    mig_graph = cir_graph.graph
    exist_const = set()
    for node in nx.topological_sort(mig_graph):
        ntype = mig_graph.nodes[node]["type"]
        if ntype == 'and':
            mig_graph.add_edges_from([(const1, node)], inv=True)
        elif ntype == 'or':
            mig_graph.add_edges_from([(const1, node)], inv=False)
        elif ntype in ['0', '1'] and node != const1:
            exist_const.add(node)
            for suc in mig_graph.successors(node):
                mig_graph.add_edges_from([(const1, suc)], inv=ntype == '0')
        else:
            continue
        mig_graph.nodes[node]["type"] = 'M'
    # clean other constants
    mig_graph.remove_nodes_from(exist_const)
    return cir_graph


def circuit_to_verilog(c, behavioral=False):
    """
    Generate a `str` of Verilog code from a `CircuitGraph`.

    Parameters
    ----------
    c: Circuit
            the circuit to turn into Verilog.
    behavioral: bool
            if True, use assign statements instead of primitive gates.

    Returns
    -------
    str
        Verilog code.

    """
    c = cg.Circuit(graph=c.graph.copy(), name=c.name, blackboxes=c.blackboxes.copy())
    # sanitize escaped nets
    for node in c.nodes():
        if node.startswith("\\"):
            c.relabel({node: node + " "})

    inputs = list(c.inputs())
    outputs = list(c.outputs())
    insts = []
    wires = []

    # blackboxes
    for name, bb in c.blackboxes.items():
        io = []
        for n in bb.inputs():
            try:
                driver = c.fanin(f"{name}.{n}").pop()
                io += [f".{n}({driver})"]
            except KeyError:
                io += [f".{n}()"]

        for n in bb.outputs():
            try:
                driven = c.fanout(f"{name}.{n}").pop()
                # Disconnect so no buffer is created
                c.disconnect(f"{name}.{n}", driven)
                io += [f".{n}({driven})"]
            except KeyError:
                io += [f".{n}()"]

        io_def = ", ".join(io)
        insts.append(f"{bb.name} {name} ({io_def})")

    # gates
    for n in c.nodes():
        if c.type(n) in ["xor", "xnor", "buf", "not", "nor", "or", "and", "nand", "M"]:
            wires.append(n)
            fanin = list(c.fanin(n))
            if not fanin:
                continue
            if behavioral:
                if c.type(n) == "buf":
                    insts.append(f"assign {n} = {fanin[0]}")
                elif c.type(n) == "not":
                    insts.append(f"assign {n} = ~{fanin[0]}")
                elif c.type(n) == "M":
                    insts.append(f"assign {n} = M({', '.join(fanin)})")
                else:
                    if c.type(n) in ["xor", "xnor"]:
                        symbol = "^"
                    elif c.type(n) in ["and", "nand"]:
                        symbol = "&"
                    elif c.type(n) in ["nor", "or"]:
                        symbol = "|"
                    fanin = f" {symbol} ".join(fanin)
                    if c.type(n) in ["xnor", "nor", "nand"]:
                        insts.append(f"assign {n} = ~({fanin})")
                    else:
                        insts.append(f"assign {n} = {fanin}")
            else:
                fanin = ", ".join(fanin)
                gate_name = c.uid(f"g_{len(insts)}")
                insts.append(f"{c.type(n)} {gate_name}({n}, {fanin})")
        elif c.type(n) in ["0", "1", "x"]:
            insts.append(f"assign {n} = 1'b{c.type(n)}")
            wires.append(n)
        elif c.type(n) in ["input", "bb_input", "bb_output"]:
            pass
        else:
            raise ValueError(f"unknown gate type: {c.type(n)}")

    verilog = f"module {c.name} ("
    verilog += ", ".join(inputs + outputs)
    verilog += ");\n"
    verilog += "".join(f"  input {inp};\n" for inp in inputs)
    verilog += "\n"
    verilog += "".join(f"  output {out};\n" for out in outputs)
    verilog += "\n"
    verilog += "".join(f"  wire {wire};\n" for wire in wires)
    verilog += "\n"
    verilog += "".join(f"  {inst};\n" for inst in insts)
    verilog += "endmodule\n"

    return verilog
