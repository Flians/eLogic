#!/usr/bin/env python3
"""Exercise the production dependency miter with the actual ABC SAT backend."""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[3]
s=(root/'epmapping/epmapping.hpp').read_text();a=s.index('  bool prove_global_dependency(');b=s.index('{',a)+1;d=1
while d:
 d+=(s[b]=='{')-(s[b]=='}');b+=1
method=s[a:b]
source=r'''
#include <bill/sat/interface/abc_bsat2.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>
struct cut_context {std::vector<uint32_t> leaves;uint32_t root_index;uint64_t tt_cache;};
struct stats {unsigned global_sat_queries=0,global_unknown=0,global_proved=0;};
struct Actual {
 using solver=bill::solver<bill::solvers::bsat2>;
 std::unique_ptr<solver> global_solver_=std::make_unique<solver>();
 struct {unsigned global_sat_conflicts=2000;} ps_;
''' + method+r'''
};
int main(){
 Actual m;for(int i=0;i<4;++i)m.global_solver_->add_variable();
 auto lit=[](unsigned i,bool v){return bill::lit_type(i,v?bill::lit_type::polarities::positive:bill::lit_type::polarities::negative);};
 // Variables: 0=a, 1=b, 2=d=a|b, 3=r=a&d.
 m.global_solver_->add_clause(std::vector<bill::lit_type>{lit(0,false),lit(2,true)});
 m.global_solver_->add_clause(std::vector<bill::lit_type>{lit(1,false),lit(2,true)});
 m.global_solver_->add_clause(std::vector<bill::lit_type>{lit(0,true),lit(1,true),lit(2,false)});
 m.global_solver_->add_clause(std::vector<bill::lit_type>{lit(3,false),lit(0,true)});
 m.global_solver_->add_clause(std::vector<bill::lit_type>{lit(3,false),lit(2,true)});
 m.global_solver_->add_clause(std::vector<bill::lit_type>{lit(0,false),lit(2,false),lit(3,true)});
 stats st;
 assert(m.prove_global_dependency({{0,2},3,8},st)); // conjunction: correct for all PI assignments
 assert(m.prove_global_dependency({{0},3,2},st));   // identity via globally correlated nodes
 assert(!m.prove_global_dependency({{1},3,2},st)); // sampled similarity could not justify this
 assert(!m.prove_global_dependency({{0,2},3,14},st)); // wrong OR despite some unreachable support tuples
 assert(st.global_proved==2 && st.global_sat_queries>=6 && st.global_unknown==0);
}
'''
with tempfile.TemporaryDirectory(prefix='epm-global-proof-') as folder:
 p=Path(folder);(p/'test.cpp').write_text(source)
 subprocess.run(['g++','-std=c++17','-O2','-I'+str(root/'tools/mockturtle/lib/bill'),'-I'+str(root/'tools/mockturtle/lib/abcsat'),str(p/'test.cpp'),str(root/'build/tools/mockturtle/lib/abcsat/liblibabcsat.a'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('Actual SAT miter proves globally correlated supports and rejects counterexamples')
