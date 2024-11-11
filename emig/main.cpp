#include "mig_egg/src/lib.rs.h"
#include "rewrite_egg.hpp"
#include "rust/cxx.h"

#include <stdio.h>
#include <string>
#include <vector>

#include <experiments.hpp>
// #include <mockturtle/algorithms/rewrite.hpp>
#include <mockturtle/mockturtle.hpp>

#define EXPERIMENTS_PATH "tools/mockturtle/experiments/"

const uint32_t CutSize = 8u;

void main_aig() {

  experiments::experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool>
      exp(fmt::format("rewrite_elo_aig_k{}", CutSize), "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  for (auto const &benchmark : experiments::epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    std::string benchmark_path = fmt::format("{}benchmarks/{}.aig", EXPERIMENTS_PATH, benchmark);

    mockturtle::aig_network aig;
    if (lorina::read_aiger(benchmark_path, mockturtle::aiger_reader(aig)) != lorina::return_code::success) {
      continue;
    }

    uint32_t const size_before = aig.num_gates();
    uint32_t const depth_before = mockturtle::depth_view(aig).depth();

    mockturtle::rewrite_params ps;
    mockturtle::rewrite_stats st;
    ps.use_egg = true;
    ps.cut_enumeration_ps.cut_size = CutSize;
    ps.cut_enumeration_ps.cut_limit = 8u;
    mockturtle::rewrite<mockturtle::aig_network, CutSize>(aig, ps, &st);

    bool const cec = experiments::abc_cec_impl(aig, benchmark_path);
    uint32_t const size_after = aig.num_gates();
    uint32_t const depth_after = mockturtle::depth_view(aig).depth();
    exp(benchmark, size_before, aig.num_gates(), depth_before, depth_after, mockturtle::to_seconds(st.time_total), cec);

    std::cout << "size_before = " << size_before << ", depth_before = " << depth_before << std::endl;
    std::cout << "size_after = " << size_after << ", depth_after = " << depth_after << std::endl;
  }

  exp.save();
  exp.table();
}

void main_mig() {

  experiments::experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool>
      exp(fmt::format("rewrite_elo_mig_k{}", CutSize), "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  for (auto const &benchmark : experiments::epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    std::string benchmark_path = fmt::format("{}benchmarks/{}.aig", EXPERIMENTS_PATH, benchmark);

    mockturtle::mig_network mig;
    if (lorina::read_aiger(benchmark_path, mockturtle::aiger_reader(mig)) != lorina::return_code::success) {
      continue;
    }

    uint32_t const size_before = mig.num_gates();
    uint32_t const depth_before = mockturtle::depth_view(mig).depth();

    mockturtle::rewrite_params ps;
    mockturtle::rewrite_stats st;
    ps.use_egg = true;
    ps.cut_enumeration_ps.cut_size = CutSize;
    ps.cut_enumeration_ps.cut_limit = 8u;
    mockturtle::rewrite<mockturtle::mig_network, CutSize>(mig, ps, &st);

    bool const cec = experiments::abc_cec_impl(mig, benchmark_path);
    uint32_t const size_after = mig.num_gates();
    uint32_t const depth_after = mockturtle::depth_view(mig).depth();
    exp(benchmark, size_before, mig.num_gates(), depth_before, depth_after, mockturtle::to_seconds(st.time_total), cec);

    std::cout << "size_before = " << size_before << ", depth_before = " << depth_before << std::endl;
    std::cout << "size_after = " << size_after << ", depth_after = " << depth_after << std::endl;
  }

  exp.save();
  exp.table();
}

int main(int argc, char *argv[]) {
  int op = 0;
  if (argc >= 2)
    op = atoi(argv[1]);
  if (op == 0) {
    main_aig();
  } else if (op == 1) {
    main_mig();
  } else {
    auto cost = std::unique_ptr<CCost, decltype(&free_ccost)>(simplify_size("(M (~ 0) (M 0 (M 0 a c) (~ (M 0 (M (~ 0) b d) (~ (M 0 b d))))) (M 0 (~ (M 0 a c)) (M 0 (M (~ 0) b d) (~ (M 0 b d)))))", nullptr, 0), free_ccost);
    std::cout << "Size: " << cost->aft_size << ", Depth: " << cost->aft_dep << ", Expr: " << cost->aft_expr << std::endl;
  }
  return 0;
}