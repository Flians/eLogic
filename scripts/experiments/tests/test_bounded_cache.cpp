#include "epmapping/bounded_cache.hpp"
#include <cassert>
#include <optional>
int main() {
  epmapping::bounded_cache<std::optional<int>> cache;
  cache.insert("old", 1, 2, true); cache.insert("hot", 2, 2, true);
  assert(cache.find("old") && **cache.find("old") == 1);
  cache.insert("new", 3, 2, true);
  assert(!cache.find("hot") && **cache.find("new") == 3);
  cache.insert("failure", std::nullopt, 2, true);
  assert(cache.find("failure") && !*cache.find("failure"));
  assert(!cache.find("old"));
  epmapping::bounded_cache<int> frozen;
  frozen.insert("old", 1, 1, false); frozen.insert("new", 2, 1, false);
  assert(frozen.find("old") && !frozen.find("new"));
  epmapping::bounded_cache<int> disabled;
  disabled.insert("key", 1, 0, true); assert(!disabled.find("key"));
}
