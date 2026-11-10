#include "epmapping/pivot_features.hpp"
#include <cassert>
#include <numeric>
#include <random>
#include <iostream>
int main() {
  std::mt19937_64 rng(20261023);
  for(uint32_t n=3;n<=6;++n)for(int trial=0;trial<250;++trial) {
    kitty::dynamic_truth_table tt(n);tt._bits[0]=rng();tt.mask_bits();
    auto f=epmapping::analyze_pivots(tt);
    for(uint32_t p=0;p<n;++p)assert(kitty::has_var(tt,p)==(kitty::cofactor0(tt,p)!=kitty::cofactor1(tt,p)));
    for(uint32_t kind=1;kind<=3;++kind) {
      auto old_score=[&](uint32_t p) {
        auto lo=kitty::cofactor0(tt,p),hi=kitty::cofactor1(tt,p),delta=lo^hi;
        uint32_t l=0,h=0,d=0;
        for(uint32_t i=0;i<n;++i){l+=kitty::has_var(lo,i);h+=kitty::has_var(hi,i);d+=kitty::has_var(delta,i);}
        return kind==1?l+h:d+(kind==2?l:h);
      };
      for(uint32_t budget=0;budget<=n+1;++budget) {
        std::vector<uint32_t> old(n);std::iota(old.begin(),old.end(),0);
        if(budget && budget<n){std::stable_sort(old.begin(),old.end(),[&](auto a,auto b){return old_score(a)<old_score(b);});old.resize(budget);std::sort(old.begin(),old.end());}
        assert(epmapping::select_pivots(f,kind,budget,false)==old);
        auto extra=epmapping::select_pivots(f,kind,budget,true);
        assert(std::includes(extra.begin(),extra.end(),old.begin(),old.end()));
        assert(extra.size()<=old.size()+1);
        for(auto p:extra)if(!std::binary_search(old.begin(),old.end(),p))assert(f[p].compact_split(kind));
      }
    }
  }
  // Low-support mux: tied omitted pivots really exercise the bounded recheck.
  bool exercised=false;
  for(uint64_t bits=0;bits<256;++bits){kitty::dynamic_truth_table tt(3);tt._bits[0]=bits;auto f=epmapping::analyze_pivots(tt);
    for(uint32_t k=1;k<=3;++k)exercised|=epmapping::select_pivots(f,k,2,true).size()==3;}
  assert(exercised);
  std::cout << "1000 random functions: legacy ranking identical, anchors retained, <=1 compact recheck\n";
}
