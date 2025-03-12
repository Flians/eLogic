#include "baseline.hpp"
#include "mig_egg/src/lib.rs.h"
#include "rewrite_egg.hpp"
#include "rust/cxx.h"

#include <stdio.h>
#include <string>
#include <vector>

#include <experiments.hpp>
// #include <mockturtle/algorithms/rewrite.hpp>
#include <mockturtle/mockturtle.hpp>

#define EXPERIMENTS_PATH "./tools/mockturtle/experiments/"

const uint32_t CutSize = 8u;

void main_aig() {

  experiments::experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool> exp(fmt::format("rewrite_elo_aig_k{}", CutSize), "benchmark", "size_before", "depth_before", "size_after", "depth_after", "runtime", "equivalent");

  mockturtle::xag_npn_resynthesis<mockturtle::aig_network, mockturtle::xag_network, mockturtle::xag_npn_db_kind::aig_complete> resyn;
  mockturtle::exact_library_params ps_exact;
  ps_exact.compute_dc_classes = true;
  mockturtle::exact_library<mockturtle::aig_network, 4u> exact_lib(resyn, ps_exact);

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
    ps.cut_enumeration_ps.cut_limit = 15u;
    mockturtle::rewrite<mockturtle::aig_network, mockturtle::exact_library<mockturtle::aig_network, 4u>, CutSize>(aig, exact_lib, ps, &st);

    bool const cec = experiments::abc_cec_impl(aig, benchmark_path);
    uint32_t const size_after = aig.num_gates();
    uint32_t const depth_after = mockturtle::depth_view(aig).depth();
    exp(benchmark, size_before, depth_before, aig.num_gates(), depth_after, mockturtle::to_seconds(st.time_total), cec);

    std::cout << "size_before = " << size_before << ", depth_before = " << depth_before << std::endl;
    std::cout << "size_after = " << size_after << ", depth_after = " << depth_after << std::endl;
  }

  exp.save();
  exp.table();
}

void main_mig() {
  experiments::experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t, float, uint32_t, uint32_t, float, bool> exp(
      fmt::format("rewrite_elo_mig_k{}_post2", CutSize), "benchmark", "size_before", "depth_before", "size_after", "depth_after", "runtime", "size_post_rw", "depth_post_rw", "runtime_post_rw", "size_post_resub", "depth_post_resub", "runtime_post_resub", "equivalent");

  mockturtle::mig_npn_resynthesis resyn;
  mockturtle::exact_library_params ps_exact;
  ps_exact.compute_dc_classes = true;
  mockturtle::exact_library<mockturtle::mig_network, 4u> exact_lib(resyn, ps_exact);

  for (auto const &benchmark : experiments::epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    std::string benchmark_path = fmt::format("{}benchmarks/{}.aig", EXPERIMENTS_PATH, benchmark);

    mockturtle::mig_network mig;
    if (lorina::read_aiger(benchmark_path, mockturtle::aiger_reader(mig)) != lorina::return_code::success) {
      continue;
    }

    uint32_t const size_before = mig.num_gates();
    uint32_t const depth_before = mockturtle::depth_view(mig).depth();
    std::cout << "size_before = " << size_before << ", depth_before = " << depth_before << std::endl;

    mockturtle::rewrite_params ps;
    mockturtle::rewrite_stats st;
    ps.use_egg = true;
    ps.window_size = 8u;
    ps.cut_enumeration_ps.cut_size = CutSize;
    ps.cut_enumeration_ps.cut_limit = 15u;
    mockturtle::rewrite<mockturtle::mig_network, mockturtle::exact_library<mockturtle::mig_network, 4u>, CutSize>(mig, exact_lib, ps, &st);

    uint32_t const size_after = mig.num_gates();
    mockturtle::depth_view depth_mig{mig};
    uint32_t const depth_after = depth_mig.depth();
    std::cout << "size_after = " << size_after << ", depth_after = " << depth_after << std::endl;
    mockturtle::exp_map.flush_cost_table();

    double runtime_after_post_rw = 0;
    mockturtle::mig_network mig_post_rw = mig.clone();
    {
      // post optimizaiton using rewrite
      baseline::rewrite_params ps_size;
      baseline::rewrite_stats st_size;
      ps_size.use_dont_cares = true;
      ps_size.window_size = 8u;
      ps_size.cut_enumeration_ps.cut_size = 4u;
      ps_size.cut_enumeration_ps.cut_limit = 15u;
      baseline::rewrite(mig_post_rw, exact_lib, ps_size, &st_size);
      runtime_after_post_rw = mockturtle::to_seconds(st_size.time_total);
    }
    uint32_t const size_after_post_rw = mig_post_rw.num_gates();
    uint32_t const depth_after_post_rw = mockturtle::depth_view(mig_post_rw).depth();
    std::cout << "size_after_post_rw = " << size_after_post_rw << ", depth_after_post_rw = " << depth_after_post_rw << std::endl;

    double runtime_after_post_resub = 0;
    {
      // post optimizaiton using resubstitution
      mockturtle::resubstitution_params ps_size;
      mockturtle::resubstitution_stats st_size;
      ps_size.max_pis = 8u;
      ps_size.max_inserts = 1u;
      ps_size.progress = false;
      ps_size.window_size = 12u;
      ps_size.use_dont_cares = true;
      mockturtle::fanout_view fanout_mig{depth_mig};
      mockturtle::mig_resubstitution2(fanout_mig, ps_size, &st_size);
      mig = cleanup_dangling(mig);
      runtime_after_post_resub = mockturtle::to_seconds(st_size.time_total);
    }

    bool const cec = experiments::abc_cec_impl(mig, benchmark_path);

    uint32_t const size_after_post_resub = mig.num_gates();
    uint32_t const depth_after_post_resub = mockturtle::depth_view(mig).depth();
    std::cout << "size_after_post_resub = " << size_after_post_resub << ", depth_after_post_resub = " << depth_after_post_resub << std::endl;

    exp(benchmark, size_before, depth_before, size_after, depth_after, mockturtle::to_seconds(st.time_total), size_after_post_rw, depth_after_post_rw, runtime_after_post_rw, size_after_post_resub, depth_after_post_resub, runtime_after_post_resub, cec);
  }

  exp.save();
  exp.table();
}

int main(int argc, char *argv[]) {
  int op = 0;
  if (argc >= 2) op = atoi(argv[1]);
  if (op == 0) {
    main_aig();
  } else if (op == 1) {
    main_mig();
  } else if (op == 2) {
    std::vector<uint32_t> leaf_levels = {0, 0, 2, 2, 4, 6, 5, 7};
    auto cost = std::make_unique<CCost>(simplify_size("(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))", leaf_levels.data(), leaf_levels.size()));
    std::cout << "Size: " << cost->aft_size << ", Depth: " << cost->aft_dep << ", Expr: ";
    print_rust_vec_string(cost->aft_expr);

    std::vector<uint32_t> leaf_levels2 = {0, 0, 3, 4, 2, 4};
    cost = std::make_unique<CCost>(simplify_size("(M (~ 0) b (M (~ (M a (~ c) e)) f (M 0 d f)))", leaf_levels2.data(), leaf_levels2.size()));
    std::cout << "Size: " << cost->aft_size << ", Depth: " << cost->aft_dep << ", Expr: ";
    print_rust_vec_string(cost->aft_expr);
  } else {
    // clean the library of expr2cost
    mockturtle::exp_map.merge_cost_table("mig_egg/lib_expr2cost.json");
  }
  return 0;
}