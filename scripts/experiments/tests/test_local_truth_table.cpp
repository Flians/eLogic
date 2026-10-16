#include "epmapping/local_truth_table.hpp"
#include <random>
#include <unordered_map>
#include <functional>
int main(){
 std::mt19937_64 rng(42);std::vector<uint8_t> scratch;
 for(unsigned leaves=0;leaves<=6;++leaves)for(unsigned trial=0;trial<200;++trial){
  std::vector<uint32_t> inputs;for(unsigned i=0;i<leaves;++i)inputs.push_back(2+i);
  std::vector<epmapping::local_truth_node> nodes;
  std::vector<uint32_t> known=inputs;
  for(unsigned j=0;j<12;++j){
   epmapping::local_truth_node node;node.index=50+3*j;
   unsigned arity=rng()%10;
   for(unsigned k=0;k<arity;++k){bool constant=known.empty() || rng()%4==0;node.fanins.push_back({constant?0:known[rng()%known.size()],bool(rng()%2),constant,bool(rng()%2)});}
   node.function.resize(std::max(1u,(1u<<arity)/64));for(auto& word:node.function)word=rng();nodes.push_back(node);known.push_back(node.index);
  }
  uint64_t expected=0;
  for(unsigned m=0;m<(1u<<leaves);++m){
   std::unordered_map<uint32_t,bool> values;for(unsigned i=0;i<leaves;++i)values[inputs[i]]=(m>>i)&1;
   for(auto const& node:nodes){unsigned idx=0;for(unsigned k=0;k<node.fanins.size();++k){auto f=node.fanins[k];bool v=f.constant?f.value:values.at(f.index);idx|=unsigned(v!=f.negated)<<k;}values[node.index]=(node.function[idx/64]>>(idx%64))&1;}
   if(values.at(nodes.back().index))expected|=uint64_t{1}<<m;
  }
  // Polluted scratch and sparse global indices exercise cross-cut reuse.
  scratch.assign(1000,255);auto actual=epmapping::simulate_local_truth(inputs,nodes,nodes.back().index,1000,scratch);assert(actual==expected);
  assert(epmapping::simulate_local_truth(inputs,nodes,nodes.back().index,1100,scratch)==expected);
 }
}
