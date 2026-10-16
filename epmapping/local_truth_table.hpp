#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
namespace epmapping {
struct local_truth_fanin {uint32_t index;bool negated;bool constant;bool value;};
struct local_truth_node {uint32_t index;std::vector<local_truth_fanin> fanins;std::vector<uint64_t> function;};
// A closed, topologically ordered cut writes every nonconstant slot before reading it.
// Scratch may retain unrelated values from previous cuts and may grow after substitutions.
inline uint64_t simulate_local_truth(std::vector<uint32_t> const& leaves,
 std::vector<local_truth_node> const& nodes,uint32_t root,uint32_t slots,std::vector<uint8_t>& scratch) {
 assert(leaves.size()<=6);scratch.resize(slots);
#ifndef NDEBUG
 std::vector<bool> defined(slots,false);
 for(auto l:leaves){assert(l<slots);defined[l]=true;}
 for(auto const& n:nodes){assert(n.index<slots);for(auto const& f:n.fanins)assert(f.constant || (f.index<slots && defined[f.index]));defined[n.index]=true;}
 assert(root<slots && defined[root]);
#endif
 uint64_t result=0;
 for(uint32_t m=0;m<(1u<<leaves.size());++m){
  for(uint32_t i=0;i<leaves.size();++i)scratch[leaves[i]]=(m>>i)&1u;
  for(auto const& n:nodes){
   uint32_t assignment=0;
   for(uint32_t i=0;i<n.fanins.size();++i){auto const& f=n.fanins[i];bool v=f.constant?f.value:scratch[f.index];assignment|=uint32_t(v ^ f.negated)<<i;}
   assert(assignment/64<n.function.size());scratch[n.index]=(n.function[assignment/64]>>(assignment%64))&1u;
  }
  if(scratch[root])result|=uint64_t{1}<<m;
 }
 return result;
}
}
