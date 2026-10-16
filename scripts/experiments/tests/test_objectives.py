#!/usr/bin/env python3
"""Exercise the real candidate comparator and the shared lexicographic key."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
header = (root / 'epmapping/epmapping.hpp').read_text()
start = header.index('  bool candidate_is_better(')
brace = header.index('{', start)
end, depth = brace + 1, 1
while depth:
    depth += (header[end] == '{') - (header[end] == '}')
    end += 1
method = header[start:end]
selector_start=header.index('  uint8_t context_objective(')
selector_end=header.index('\n  }',selector_start)+4
selector=header[selector_start:selector_end]
start=header.index('  std::pair<int64_t, int64_t> objective_metric() const')
brace=header.index('{',start);end=brace+1;depth=1
while depth:
    depth += (header[end]=='{')-(header[end]=='}');end+=1
metric=header[start:end]
source = r'''
#include "epmapping/objective.hpp"
#include <cassert>
#include <tuple>
namespace epmapping {
struct CCost { double area, delay; };
struct cut_context { unsigned level; double original_area=0; bool critical=false; };
struct Actual {
  struct { unsigned objective=1; bool rank_area_gain=false; bool path_objective=false; } ps_;
  struct { double area=2, delay=3; double compute_area() const {return area;} double compute_worst_delay() const {return delay;} } ntk_;
  unsigned candidate_level(CCost const&, cut_context const& c) const { return c.level; }
''' + selector + method + metric + r'''
}; }
int main() {
  using namespace epmapping;
  assert(parse_objective("delay_area")==0 && parse_objective("(delay,area)")==0);
  assert(parse_objective("area_delay")==1 && parse_objective("(area,delay)")==1);
  assert(parse_objective("0")==0 && parse_objective("1")==1);
  assert(!parse_objective("2") && !parse_objective("unknown"));
  assert(objective_key(0,10,2)<objective_key(0,1,3)); // faster, larger
  assert(objective_key(1,1,3)<objective_key(1,10,2)); // smaller, slower
  assert(objective_key(0,1,3)<objective_key(0,2,3)); // secondary-only progress
  assert(objective_key(1,1,2)<objective_key(1,1,3));
  assert(objective_key(1,1.000001,2.000001)==objective_key(1,1,2));
  Actual m;
  assert(m.candidate_is_better({1,2},{9},{1,3},{1})); // delay precedes depth
  m.ps_.objective=0;
  assert(m.candidate_is_better({1,2},{9},{2,2},{1})); // area precedes depth
  assert(!m.candidate_is_better({1,3},{1},{2,2},{9}));
  m.ps_.objective=1;
  // A larger replacement can save more area when it deletes a larger MFFC.
  assert(!m.candidate_is_better({3,2},{1,10},{1,2},{1,2}));
  m.ps_.rank_area_gain=true;
  assert(m.candidate_is_better({3,2},{1,10},{1,2},{1,2}));
  m.ps_.objective=0;
  assert(m.candidate_is_better({3,2},{1,10},{1,2},{1,2}));
  m.ps_.path_objective=true;
  // Critical cones choose speed despite area growth; noncritical cones choose area.
  assert(m.candidate_is_better({3,2},{1,2,true},{1,3},{1,2,true}));
  assert(!m.candidate_is_better({3,2},{1,2,false},{1,3},{1,2,false}));
  assert(m.context_objective({1,2,true})==0);
  assert(m.context_objective({1,2,false})==1);
  m.ps_.path_objective=false;
  auto const before=m.objective_metric();
  m.ntk_.area=1;
  assert(m.objective_metric()<before); // controller must continue on secondary-only improvement
  m.ntk_.area=4;
  assert(m.objective_metric()>before); // round guard must roll back worsening secondary ties
}
'''
with tempfile.TemporaryDirectory(prefix='epm-objective-') as directory:
    tmp = Path(directory)
    cpp, binary = tmp / 'test.cpp', tmp / 'test'
    cpp.write_text(source)
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(root), str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print('Objective aliases, tradeoffs, secondary progress and real candidate ordering passed')
