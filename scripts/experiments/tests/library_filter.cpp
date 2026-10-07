#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <kitty/dynamic_truth_table.hpp>
#include <kitty/hash.hpp>
#include <kitty/operators.hpp>
struct Gate { std::string name; };
template<unsigned N> struct composed_gate {
 Gate const* root{}; uint32_t num_vars{}; kitty::dynamic_truth_table function;
 double area{}; std::array<float,N> tdelay{};
};
// Methods of the actual header and pre-optimization reference are injected by the runner.
#include "filter_methods.hpp"
int main() {
 std::mt19937 rng(42); std::array<Gate,8> roots;
 for(unsigned i=0;i<roots.size();++i) roots[i].name=std::to_string(i);
 for(bool minimum : {false,true}) for(unsigned trial=0;trial<500;++trial) {
  std::deque<composed_gate<9>> gs; std::vector<bool> skipped;
  auto count=2+rng()%180;
  for(unsigned i=0;i<count;++i) {
   composed_gate<9> g; g.num_vars=1+rng()%4; g.function=kitty::dynamic_truth_table(g.num_vars);
   g.function._bits[0]=rng()%8; g.root=&roots[rng()%roots.size()];
   g.area=(rng()%5)/10.0;
   for(auto& d:g.tdelay) d=float(rng()%4);
   // Null roots are valid inputs only in their own function groups in the library.
   if(rng()%12==0) { g.root=nullptr; g.function=kitty::dynamic_truth_table(0); }
   gs.push_back(g); skipped.push_back(rng()%8==0);
  }
  auto a=skipped,b=skipped; Actual actual; Reference reference;
  actual._ps.load_minimum_size_only=reference._ps.load_minimum_size_only=minimum;
  actual.filter_gates(gs,a); reference.filter_gates(gs,b);
  if(a!=b) { std::cerr<<"mismatch in trial "<<trial<<"\n"; return 1; }
 }
 std::cout<<"1000 randomized dominance-filter comparisons passed\n";
}
