#!/usr/bin/env python3
"""Check the actual backward timing recurrence on reconvergent, asymmetric pin paths."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
header = (root/'epmapping/epmapping.hpp').read_text()
start = header.index('  double required_time(')
end, depth = header.index('{', start)+1, 1
while depth:
    depth += (header[end] == '{') - (header[end] == '}')
    end += 1
method = header[start:end]
source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>
struct Signal { unsigned node; bool complement=false; };
struct Gate { std::vector<double> delays; };
struct Cell { std::vector<Gate> gates; };
struct Network {
  std::vector<std::vector<Signal>> inputs;
  std::vector<Cell> cells;
  std::vector<Signal> outputs;
  unsigned get_node(Signal f) const { return f.node; }
  unsigned node_to_index(unsigned n) const { return n; }
  unsigned index_to_node(unsigned n) const { return n; }
  bool is_dead(unsigned) const { return false; }
  bool has_cell(unsigned n) const { return !cells[n].gates.empty(); }
  Cell const& get_cell(unsigned n) const { return cells[n]; }
  template<class F> void foreach_po(F f) const { for(auto s:outputs) f(s); }
  template<class F> void foreach_fanin(unsigned n,F f) const { for(auto s:inputs[n]) f(s); }
};
struct Actual {
  Network ntk_;
  std::vector<std::vector<unsigned>> fanouts_;
  mutable std::unordered_map<unsigned,double> required_cache_;
  double delay_bound_=100, inv_delay_=2, live_delay_target_=80;
  struct { bool path_objective=false; } ps_;
  double pin_delay_of(Gate const& g,unsigned p) const {return g.delays[p];}
''' + method + r'''
};
int main() {
  Actual m;
  // 0 -> 1 -> 3, and 0 -> 2 -> 3. Pins at node 3 have different delays.
  m.ntk_.inputs={{},{{0}},{{0}},{{1},{2,true}}};
  m.ntk_.cells={ {},{{{{10}}}},{{{{20}}}},{{{{5,30}}}} };
  m.ntk_.outputs={{3,true},{1}};
  m.fanouts_={{1,2},{3},{3},{}};
  assert(m.required_time(3)==98); // complemented output costs 2
  assert(m.required_time(1)==93); // tighter than the direct PO budget 100
  assert(m.required_time(2)==66); // physical pin 1 delay 30, complemented edge 2
  assert(m.required_time(0)==46); // minimum over both reconvergent paths
  m.ntk_.cells[3].gates[0].delays[1]=40;
  m.required_cache_.clear(); // substitutions invalidate cached fanout requirements
  assert(m.required_time(0)==36);
  m.ntk_.inputs[3]={{1}}; // stale fanout adjacency must not impose a removed edge
  m.required_cache_.clear();
  assert(m.required_time(0)==83);
  m.ps_.path_objective=true;
  m.required_cache_.clear();
  assert(m.required_time(0)==63); // tightening the live critical path updates every requirement
}
'''
with tempfile.TemporaryDirectory(prefix='epm-required-times-') as directory:
    tmp = Path(directory)
    (tmp/'test.cpp').write_text(source)
    subprocess.run(['g++','-std=c++17','-O2',str(tmp/'test.cpp'),'-o',str(tmp/'test')],check=True)
    subprocess.run([str(tmp/'test')],check=True)
print('Live required times: reconvergence, physical pin delays, inversions and invalidation passed')
