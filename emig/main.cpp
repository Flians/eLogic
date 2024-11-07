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

void main_aig() {

  experiments::experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool>
      exp("rewrite_aig_dc", "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  mockturtle::xag_npn_resynthesis<mockturtle::aig_network, mockturtle::xag_network, mockturtle::xag_npn_db_kind::aig_complete> resyn;
  mockturtle::exact_library_params ps_exact;
  ps_exact.compute_dc_classes = true;
  mockturtle::exact_library<mockturtle::aig_network> exact_lib(resyn, ps_exact);

  for (auto const &benchmark : experiments::all_benchmarks()) {
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
    ps.window_size = 12;
    ps.cut_enumeration_ps.cut_size = 8;
    ps.cut_enumeration_ps.cut_limit = 15;
    mockturtle::rewrite(aig, exact_lib, ps, &st);

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
      exp("rewrite_mig", "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  mockturtle::mig_npn_resynthesis resyn;
  mockturtle::exact_library_params ps_exact;
  ps_exact.compute_dc_classes = false;
  mockturtle::exact_library<mockturtle::mig_network> exact_lib(resyn, ps_exact);

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
    ps.window_size = 12;
    ps.cut_enumeration_ps.cut_size = 16;
    ps.cut_enumeration_ps.cut_limit = 15;
    mockturtle::rewrite(mig, exact_lib, ps, &st);

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
  int op = 1;
  if (argc >= 2)
    op = atoi(argv[1]);
  if (op == 0) {
    main_aig();
  } else if (op == 1) {
    main_mig();
  } else {
  }
  return 0;
}