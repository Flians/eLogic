#include <mockturtle/networks/block.hpp>
#include <mockturtle/views/cell_view.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
int main(){
  using view=mockturtle::cell_view<mockturtle::block_network>;
  std::vector<mockturtle::standard_cell> cells(47);
  for(unsigned i=0;i<cells.size();++i){cells[i].name="cell_"+std::to_string(i);cells[i].area=i+1;mockturtle::gate g;g.name=cells[i].name;cells[i].gates.push_back(g);}
  auto shared=std::make_shared<std::vector<mockturtle::standard_cell> const>(cells);
  view fast(shared);auto copy=fast;view assigned(cells);assigned=fast;
  assert(fast.get_library().data()==shared->data());
  assert(copy.get_library().data()==shared->data());
  assert(assigned.get_library().data()==shared->data());
  shared.reset(); // Results own the immutable library even after the caller releases it.
  assert(copy.get_library()[46].area==47 && assigned.get_library()[0].name=="cell_0");
  view owned(cells);assert(owned.get_library().data()!=cells.data());
  assigned=owned;assert(assigned.get_library().data()!=fast.get_library().data());
  assert(assigned.get_library()[46].area==47);
  auto start=std::chrono::steady_clock::now();size_t total=0;
  for(unsigned i=0;i<10000;++i){view v(cells);total+=v.get_library().size();}
  auto middle=std::chrono::steady_clock::now();
  auto pool=std::make_shared<std::vector<mockturtle::standard_cell> const>(cells);
  for(unsigned i=0;i<10000;++i){view v(pool);total+=v.get_library().size();}
  auto end=std::chrono::steady_clock::now();assert(total==940000);
  std::cout<<"owned_seconds="<<std::chrono::duration<double>(middle-start).count()<<" shared_seconds="<<std::chrono::duration<double>(end-middle).count()<<'\n';
}
