#include "epmapping/exact_aig_key.hpp"
#include "epmapping/bounded_cache.hpp"
#include <mockturtle/networks/aig.hpp>
#include <cassert>
int main() {
  auto make=[](){mockturtle::aig_network a;auto x=a.create_pi(),y=a.create_pi();a.create_po(a.create_and(x,y));return a;};
  auto a=make(),b=make();
  auto key=epmapping::exact_aig_key(a,8,1);
  assert(key==epmapping::exact_aig_key(b,8,1));
  assert(key!=epmapping::exact_aig_key(b,8,0));
  assert(key!=epmapping::exact_aig_key(b,7,1));
  epmapping::bounded_cache<unsigned> maps;
  unsigned mapping_calls=0;
  auto build=[&](auto const& net){auto k=epmapping::exact_aig_key(net,8,1);if(auto hit=maps.find(k))return *hit;auto v=++mapping_calls;maps.insert(k,v,2,true);return v;};
  assert(build(a)==build(b) && mapping_calls==1); // Different decomposition requests, same mapper state.
  auto extra=make();extra.create_pi();
  assert(key!=epmapping::exact_aig_key(extra,8,1));
  auto phase=make();phase.create_po(!phase.make_signal(phase.index_to_node(3)));
  assert(key!=epmapping::exact_aig_key(phase,8,1));
  auto dangling=make();auto x=dangling.make_signal(dangling.index_to_node(1));auto y=dangling.make_signal(dangling.index_to_node(2));dangling.create_and(x,!y);
  assert(key!=epmapping::exact_aig_key(dangling,8,1));
  auto refs=make();refs.incr_fanout_size(refs.index_to_node(3));
  assert(key!=epmapping::exact_aig_key(refs,8,1));
  auto marks=make();marks.set_visited(marks.index_to_node(3),1);
  assert(key!=epmapping::exact_aig_key(marks,8,1));
  auto values=make();values.set_value(values.index_to_node(3),1);
  assert(key!=epmapping::exact_aig_key(values,8,1));
}
