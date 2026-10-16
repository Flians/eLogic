#include "epmapping/operator_policy.hpp"
#include <cassert>
int main(){
  epmapping::operator_policy p(2);
  for(unsigned i=0;i<128;++i)assert(p.allow(2,12,false));
  p.elapsed(2,12,1.0);p.build_elapsed(2,12,1.0);
  assert(!p.allow(2,12,false));
  assert(p.allow(2,12,true));
  assert(p.allow(2,12,false,true));
  bool probe=false;for(unsigned i=0;i<16;++i)probe|=p.allow(2,12,false);assert(probe);
  p.accepted(2,12,2.0);assert(p.allow(2,12,false));
  p.finish_round(1.0,true);
  auto const& r=p.records()[2*epmapping::operator_policy::buckets+12];assert(r.committed==1 && r.credited_area==1.0);
  assert(p.allow(2,12,false));
  epmapping::operator_policy rolled(2);
  rolled.accepted(1,0,10);rolled.finish_round(10,false);
  assert(rolled.records()[epmapping::operator_policy::buckets].committed==0 && rolled.records()[epmapping::operator_policy::buckets].credited_area==0);
  epmapping::operator_policy profile(1);
  for(unsigned i=0;i<1000;++i)assert(profile.allow(0,0,false));
  epmapping::operator_policy off;assert(off.allow(0,0,false));assert(off.records()[0].calls==0);
  epmapping::operator_policy mixed(2);
  for(unsigned i=0;i<128;++i)assert(mixed.allow(0,0,false));
  mixed.build_elapsed(0,0,1.0);assert(!mixed.allow(0,0,false));
  bool sampled=false;
  for(unsigned i=0;i<16;++i){
    for(unsigned hit=0;hit<15;++hit)assert(mixed.allow(0,0,true));
    sampled|=mixed.allow(0,0,false);
  }
  assert(sampled && "cache traffic must not starve exploration of new structures");
  epmapping::operator_policy recovery(3);
  unsigned const bucket=511;
  for(unsigned i=0;i<128;++i)assert(recovery.allow(4,bucket,false));
  recovery.build_elapsed(4,bucket,1.0);
  for(unsigned i=0;i<32;++i)assert(recovery.allow(4,bucket,false));
  auto const& rr=recovery.records()[4*epmapping::operator_policy::buckets+bucket];
  assert(rr.deferred>0 && rr.deferred==rr.rechecked && rr.skipped==0);
  assert(rr.builds==160);
  recovery.dominated(4,bucket);
  assert(rr.dominated==1);
  assert(recovery.records()[4*epmapping::operator_policy::buckets].calls==0);
}
