#pragma once

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>

namespace epmapping
{
/// A pure structural memo. Replacement changes work, never the meaning of an entry.
template<typename Value>
class bounded_cache
{
public:
  Value const* find( std::string const& key )
  {
    auto it = entries_.find( key );
    if ( it == entries_.end() ) return nullptr;
    order_.splice( order_.begin(), order_, it->second.position );
    return &it->second.value;
  }

  void insert( std::string const& key, Value const& value, size_t capacity, bool replace )
  {
    if ( capacity == 0 || entries_.count( key ) ) return;
    if ( entries_.size() >= capacity )
    {
      if ( !replace ) return;
      entries_.erase( order_.back() );
      order_.pop_back();
    }
    order_.push_front( key );
    entries_.emplace( key, entry{ value, order_.begin() } );
  }

private:
  struct entry { Value value; std::list<std::string>::iterator position; };
  std::list<std::string> order_;
  std::unordered_map<std::string, entry> entries_;
};
}
