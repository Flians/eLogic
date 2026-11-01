"""Verification-only export of physical choices already recorded during mapping.

No cuts, matching, cell choice, rewriting or optimization is performed here.
The resulting reference is never supplied to the mapper as new search evidence.
"""
import json
from pathlib import Path
from aig_io import identifier


def bundle_verilog(bundle, library_json):
    if bundle.get('schema') != 1 or len(bundle['input_names']) != bundle['input_count']:
        raise ValueError('invalid recorded source bundle')
    cells = {}
    for c in json.loads(Path(library_json).read_text()):
        if c['name'] in cells:
            raise ValueError('recorded source requires unique single-output cell names')
        cells[c['name']] = c
    ins, outs = bundle['input_names'], bundle['output_names']
    if len(outs) != len(bundle['roots']) or len(set(ins+outs)) != len(ins+outs):
        raise ValueError('invalid recorded named ports')
    prefix = '_egm_reference_'
    while any(n.startswith(prefix) for n in ins+outs):
        prefix += 'x'

    def net(literal):
        if not isinstance(literal, int) or literal < 0:
            raise ValueError('invalid recorded literal')
        if literal < 2:
            return "1'b1" if literal else "1'b0"
        if literal % 2 == 0 and literal//2 <= len(ins):
            return identifier(ins[literal//2-1])
        return prefix+str(literal)

    lines = ['module mapped('+', '.join(['input '+identifier(n) for n in ins]+
                                       ['output '+identifier(n) for n in outs])+');']
    seen = set()
    for i, choice in enumerate(bundle['instances']):
        cell, output = cells[choice['cell']], choice['output']
        if output < 2 or (output % 2 == 0 and output//2 <= len(ins)) or output in seen:
            raise ValueError('recorded physical output is constant, PI or multiply driven')
        seen.add(output)
        if len(choice['inputs']) != len(cell['vars']):
            raise ValueError('recorded physical pin count mismatch')
        lines.append('wire '+net(output)+';')
        pins = [f'.{identifier(pin)}({net(value)})' for pin, value in zip(cell['vars'], choice['inputs'])]
        pins.append(f'.{identifier(cell["out_pin"])}({net(output)})')
        lines.append(f'{identifier(cell["name"])} {prefix}cell{i}('+', '.join(pins)+');')
    for name, literal in zip(outs, bundle['roots']):
        lines.append(f'assign {identifier(name)} = {net(literal)};')
    return '\n'.join(lines+['endmodule'])+'\n'
