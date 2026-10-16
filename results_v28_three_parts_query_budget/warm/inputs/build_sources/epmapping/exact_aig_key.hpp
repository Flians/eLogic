#pragma once
#include <cstdint>
#include <string>

namespace epmapping {
// Exact pre-mapping state, not functional/NPN canonicalization. Unreachable
// nodes, PI order, edge phases and reference counts can affect mapper ties.
// The cache lives inside one optimizer with an immutable technology library.
template<class Aig>
std::string exact_aig_key(Aig const& aig, uint64_t truth, uint32_t objective)
{
  std::string key;
  auto put=[&](uint64_t v){for(unsigned i=0;i<8;++i)key.push_back(static_cast<char>(v>>(8*i)));};
  put(truth);put(objective);put(aig.size());put(aig.num_pis());put(aig.num_pos());put(aig.trav_id());
  aig.foreach_pi([&](auto n){put(aig.node_to_index(n));});
  for(uint32_t i=0;i<aig.size();++i) {
    auto n=aig.index_to_node(i);put(aig.is_dead(n));
    if(aig.is_dead(n))continue;
    put(aig.is_constant(n));put(aig.is_pi(n));put(aig.fanout_size(n));
    put(aig.value(n));put(aig.visited(n));
    if(!aig.is_constant(n) && !aig.is_pi(n))
      aig.foreach_fanin(n,[&](auto f){put(aig.node_to_index(aig.get_node(f)));put(aig.is_complemented(f));});
  }
  aig.foreach_po([&](auto f){put(aig.node_to_index(aig.get_node(f)));put(aig.is_complemented(f));});
  return key;
}
}
