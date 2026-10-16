#pragma once
#include <cstdint>
#include <unordered_map>
#include <utility>
namespace epmapping {
// Values (including negative results) are valid only in one unchanged root context.
template<class Value> class live_query_cache {
public:
 Value const* find(uint64_t revision,uint64_t slots,uint32_t root,uint32_t index) const {
  if(revision!=revision_ || slots!=slots_ || root!=root_) {
   entries_.clear();revision_=revision;slots_=slots;root_=root;
  }
  auto it=entries_.find(index);return it==entries_.end()?nullptr:&it->second;
 }
 void insert(uint32_t index,Value value) const {entries_.insert_or_assign(index,std::move(value));}
private:
 mutable uint64_t revision_=~uint64_t{0},slots_=~uint64_t{0};
 mutable uint32_t root_=~uint32_t{0};
 mutable std::unordered_map<uint32_t,Value> entries_;
};
}
