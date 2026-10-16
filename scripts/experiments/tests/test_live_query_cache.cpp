#include "epmapping/live_query_cache.hpp"
#include <cassert>
#include <optional>
int main(){
 epmapping::live_query_cache<std::optional<unsigned>> c;
 assert(!c.find(1,100,20,4));c.insert(4,std::nullopt);
 auto missing=c.find(1,100,20,4);assert(missing && !*missing);
 c.insert(5,9);assert(**c.find(1,100,20,5)==9);
 assert(!c.find(2,100,20,5)); // same node slots after substitution: revision matters
 c.insert(5,10);assert(!c.find(2,101,20,5)); // speculative append changes fanouts
 c.insert(5,11);assert(!c.find(2,101,21,5)); // different replacement root, different cycle predicate
 c.insert(5,12);assert(**c.find(2,101,21,5)==12);
}
