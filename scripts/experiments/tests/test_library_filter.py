#!/usr/bin/env python3
"""Differential-test the current C++ filter against the frozen pre-bucketing reference.
The current method body is extracted, compiled and exercised; the test does not reimplement it.
"""
from pathlib import Path
import subprocess, tempfile
root=Path(__file__).resolve().parents[3]
header='include/mockturtle/utils/tech_library.hpp'
current=(root/'tools/mockturtle'/header).read_text()
reference=Path(__file__).with_name('filter_reference.hpp').read_text()
def method(text,name):
 start=text.index('  '+name)
 brace=text.index('{',start); depth=1; end=brace+1
 while depth:
  depth += (text[end]=='{')-(text[end]=='}'); end+=1
 return text[start:end]
def cls(name,text):
 return 'struct '+name+' { static constexpr unsigned NInputs=9; struct { bool load_minimum_size_only=false; } _ps;\n'+method(text,'bool compare_sizes(')+'\n'+method(text,'void filter_gates(')+'\n};\n'
with tempfile.TemporaryDirectory(prefix='epm-filter-') as tmp:
 tmp=Path(tmp); (tmp/'filter_methods.hpp').write_text(cls('Actual',current)+reference)
 cmd=['g++','-std=c++17','-O2','-I'+str(tmp),'-I'+str(root/'tools/mockturtle/lib/kitty'),str(Path(__file__).with_name('library_filter.cpp')),'-o',str(tmp/'test')]
 subprocess.run(cmd,check=True); subprocess.run([str(tmp/'test')],check=True)
