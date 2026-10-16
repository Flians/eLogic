#!/usr/bin/env python3
"""Exercise the actual dependency selector against exhaustive window truth tables."""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[3]
s=(root/'epmapping/epmapping.hpp').read_text()
a=s.index('  std::vector<cut_context> dependency_contexts');b=s.index('{',a)+1;d=1
while d:
    d+=(s[b]=='{')-(s[b]=='}');b+=1
method=s[a:b]
a=s.index('  std::vector<uint32_t> dependency_removable_cone');b=s.index('{',a)+1;d=1
while d:
    d+=(s[b]=='{')-(s[b]=='}');b+=1
method=s[a:b]+method
source=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace kitty {struct dynamic_truth_table {uint64_t bits;}; bool get_bit(dynamic_truth_table t,unsigned m){return (t.bits>>m)&1;}}
struct Network {std::unordered_map<unsigned,std::vector<unsigned>> inputs;unsigned get_node(unsigned f)const{return f;} template<class F> void foreach_fanin(unsigned n,F f)const{auto it=inputs.find(n);if(it!=inputs.end())for(auto v:it->second)f(v);} unsigned index_to_node(unsigned i)const{return i;}unsigned node_to_index(unsigned i)const{return i;}bool is_dead(unsigned)const{return false;}};
namespace mockturtle {
struct reconvergence_driven_cut_parameters {unsigned max_leaves;};struct reconvergence_driven_cut_statistics {};
namespace detail {template<class N,bool A,bool B> struct reconvergence_driven_cut_impl {
reconvergence_driven_cut_impl(N const&,reconvergence_driven_cut_parameters const&,reconvergence_driven_cut_statistics&){}
std::pair<std::vector<unsigned>,int> run(std::vector<unsigned>) {return {{1,2,3},0};}
};}}
struct Actual {
using network=Network;using node=unsigned;Network ntk_;
struct {unsigned dependency_window_pis=10;bool dependency_internal_divisors=false;unsigned dependency_min_removed=1;} ps_;
std::vector<unsigned> levels_={0,0,0,0,1,3,2,4};
std::vector<double> arrivals_={0,0,0,0,1,3,2,4};
std::vector<std::vector<unsigned>> fanouts_={{},{4,5},{4,5},{5},{6,7},{},{},{}};
std::unordered_map<unsigned,kitty::dynamic_truth_table> tables;
struct cut_context {bool dependency=false,tt_ready=false,has_dont_cares=false;
unsigned root_index=0,original_level=0;double original_area=0,original_delay=0;
std::vector<unsigned> leaves,cone_nodes,leaf_levels,leaf_arrivals;uint64_t care_mask=0,tt_cache=0;};
std::vector<unsigned> build_mffw(std::vector<unsigned> const&) const{return {1,2,3,4,5,6,7};}
bool reaches_node(unsigned a,unsigned) const{return a==7;} // descendant must not become a divisor
std::optional<std::unordered_map<unsigned,kitty::dynamic_truth_table>> simulate_window(std::vector<unsigned> const&,std::vector<unsigned> const&)const{return tables;}
double mffc_area(std::vector<unsigned> const&,unsigned)const{return 3;}
''' + method + r'''
};
int main(){
Actual model;
model.ntk_.inputs={{5,{1,2}},{6,{5}},{7,{6,3}}};
assert(model.dependency_removable_cone({5,6,7},{6})==std::vector<unsigned>{7});
assert(model.dependency_removable_cone({5,6,7},{4})==(std::vector<unsigned>{5,6,7}));
assert(model.dependency_removable_cone({5,6,7},{5})==(std::vector<unsigned>{6,7}));
model.ntk_.inputs.clear();
// x=a&(b|c); divisor d=b|c is a side branch outside the root's MFFC.
for(unsigned m=0;m<8;++m){
 bool a=m&1,b=m&2,c=m&4,d=b||c,x=a&&d;
 for(auto [i,value]:std::vector<std::pair<unsigned,bool>>{{1,a},{2,b},{3,c},{4,d},{5,x},{6,x},{7,x}})
  if(value)model.tables[i].bits|=uint64_t(1)<<m;
}
auto contexts=model.dependency_contexts(5,{5,6});assert(!contexts.empty());
bool reused=false;
for(auto const& ctx:contexts){
 assert(ctx.dependency&&ctx.tt_ready&&ctx.original_area==3);
 for(auto i:ctx.leaves){assert(i!=5&&i!=6&&i!=7);if(i==4) reused=true;}
 for(unsigned m=0;m<8;++m){unsigned tuple=0;
  for(unsigned k=0;k<ctx.leaves.size();++k)tuple|=unsigned(kitty::get_bit(model.tables.at(ctx.leaves[k]),m))<<k;
  assert((ctx.care_mask>>tuple)&1);
  assert(((ctx.tt_cache>>tuple)&1)==kitty::get_bit(model.tables.at(5),m));
 }
}
assert(reused);
// Identity and opposite phase are legitimate one-divisor dependencies.
model.tables[5]=model.tables[4];
auto identity=model.dependency_contexts(5,{5,6});
assert(std::any_of(identity.begin(),identity.end(),[](auto const& ctx){return ctx.leaves==std::vector<unsigned>{4} && ctx.tt_cache==2 && ctx.care_mask==3;}));
model.tables[5].bits=(~model.tables[4].bits)&255;
auto inverse=model.dependency_contexts(5,{5,6});
assert(std::any_of(inverse.begin(),inverse.end(),[](auto const& ctx){return ctx.leaves==std::vector<unsigned>{4} && ctx.tt_cache==1 && ctx.care_mask==3;}));
// A free hidden input toggles target while all eligible divisors are fixed:
// no dependency proof may succeed, even if the known minterms look favourable.
for(auto& [i,tt]:model.tables)tt.bits=0;
model.tables[5].bits=0xAA;
assert(model.dependency_contexts(5,{5,6}).empty());
}
'''
with tempfile.TemporaryDirectory(prefix='epm-dependency-') as folder:
 p=Path(folder);(p/'test.cpp').write_text(source)
 subprocess.run(['g++','-std=c++17','-O2',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('Actual dependency selector: side-branch reuse, exhaustive truth/care, MFFC/cycle exclusions, conflicting target passed')
