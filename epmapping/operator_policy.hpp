#pragma once

#include <array>
#include <algorithm>
#include <cstdint>

namespace epmapping
{
/// Experimental scheduling of NEW structural builds. Cached realizations remain eligible.
/// Credits are conservative allocations of globally validated round gain, not causal ablations.
class operator_policy
{
public:
  static constexpr uint32_t sources=6, buckets=512;
  struct record {
    uint64_t calls=0, misses=0, builds=0, cache_hits=0, skipped=0, proposals=0, committed=0, delay_rejects=0, area_rejects=0;
    uint64_t deferred=0, rechecked=0, dominated=0;
    double seconds=0.0, build_seconds=0.0, credited_area=0.0;
    uint64_t pending=0;
    double pending_gain=0.0;
  };
  explicit operator_policy(uint32_t mode=0):mode_(mode){}
  bool enabled() const {return mode_!=0;}
  bool allow(uint32_t source,uint32_t bucket,bool cached,bool promising=false)
  {
    if(!enabled())return true;
    auto& r=at(source,bucket);++r.calls;
    if(cached){++r.cache_hits;return true;}
    ++r.misses;
    // Warm-up and deterministic exploration prevent permanent exclusion. A
    // structural signature or successful proposal keeps an operator active.
    bool const explore=(r.misses%16u)==0u;
    if(mode_>=2 && !promising && r.builds>=128 && r.build_seconds>=0.02 &&
       r.pending==0 && r.committed==0 && !explore){
      if(mode_==2){++r.skipped;return false;}
      // A zero-reward bucket is evidence for deferral, never a proof of
      // dominance. Mode 3 rechecks every such request before root commitment.
      // Preserve generation order so equal-cost ties follow full coverage.
      ++r.deferred;++r.rechecked;
    }
    ++r.builds;return true;
  }
  void dominated(uint32_t source,uint32_t bucket)
  {if(enabled())++at(source,bucket).dominated;}
  void elapsed(uint32_t source,uint32_t bucket,double seconds)
  {if(enabled())at(source,bucket).seconds+=seconds;}
  void build_elapsed(uint32_t source,uint32_t bucket,double seconds)
  {if(enabled())at(source,bucket).build_seconds+=seconds;}
  void reject(uint32_t source,uint32_t bucket,uint32_t reason)
  {if(enabled()){auto& r=at(source,bucket);if(reason==2)++r.delay_rejects;if(reason==3)++r.area_rejects;}}
  void accepted(uint32_t source,uint32_t bucket,double estimated_gain)
  {if(enabled()){auto& r=at(source,bucket);++r.proposals;++r.pending;r.pending_gain+=std::max(0.0,estimated_gain);}}
  void finish_round(double actual_area_gain,bool legal)
  {
    if(!enabled())return;
    double sum=0.0;for(auto const& r:records_)sum+=r.pending_gain;
    double const scale=legal && sum>0 ? std::min(1.0,std::max(0.0,actual_area_gain)/sum):0.0;
    for(auto& r:records_){
      if(legal){r.committed+=r.pending;r.credited_area+=r.pending_gain*scale;}
      r.pending=0;r.pending_gain=0.0;
    }
  }
  std::array<record,sources*buckets> const& records() const{return records_;}
private:
  record& at(uint32_t source,uint32_t bucket){return records_[source*buckets+bucket];}
  uint32_t mode_;
  std::array<record,sources*buckets> records_{};
};
}
