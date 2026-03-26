#include "emap_egg.hpp"
// #include "map_egg/src/lib.rs.h"
#include "rust/cxx.h"

#include <stdio.h>
#include <string>
#include <vector>

#include <experiments.hpp>
#include <mockturtle/mockturtle.hpp>

#define TECHLIB "asap7"

int main_mapper() {
  using namespace experiments;
  using namespace mockturtle;

  experiment<std::string, uint32_t, uint32_t, double, uint32_t, uint32_t, double, float, bool> exp(
      "mapper", "benchmark", "size", "size_after", "area_after", "depth", "depth_after", "delay_after", "runtime", "cec");

  fmt::print("[i] processing technology library for mapper\n");

  /* library to map to technology */
  std::vector<gate> gates;
  std::ifstream in(cell_libraries_path(TECHLIB));

  if (lorina::read_genlib(in, genlib_reader(gates)) != lorina::return_code::success) {
    return 1;
  }

  tech_library_params tps;
  tech_library<9, classification_type::np_configurations> tech_lib(gates, tps);

  for (auto const &benchmark : epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    names_view<aig_network> aig;
    if (lorina::read_aiger(benchmark_path(benchmark), aiger_reader(aig)) != lorina::return_code::success) {
      continue;
    }

    const uint32_t size_before = aig.num_gates();
    const uint32_t depth_before = depth_view(aig).depth();

    map_params ps2;
    ps2.cut_enumeration_ps.minimize_truth_table = true;
    ps2.cut_enumeration_ps.cut_limit = 24;
    map_stats st2;

    binding_view<klut_network> res2 = map(aig, tech_lib, ps2, &st2);

    const auto cec2 = benchmark == "hyp" ? true : abc_cec(res2, benchmark);

    exp(benchmark, size_before, res2.num_gates(), st2.area, depth_before, depth_view(res2).depth(), st2.delay, to_seconds(st2.time_total), cec2);
  }

  exp.save();
  exp.table();
  return 0;
}

int main_emapper() {
  using namespace experiments;
  using namespace mockturtle;

  experiment<std::string, uint32_t, uint32_t, double, uint32_t, uint32_t, double, float, bool> exp(
      "main_emapper", "benchmark", "size", "size_after", "area_after", "depth", "depth_after", "delay_after", "runtime", "cec");

  fmt::print("[i] processing technology library for emapper\n");

  /* library to map to technology */
  std::vector<gate> gates;
  std::ifstream in(cell_libraries_path(TECHLIB));

  if (lorina::read_genlib(in, genlib_reader(gates)) != lorina::return_code::success) {
    return 1;
  }

  tech_library_params tps;
  tps.load_multioutput_gates = false;
  tech_library<9, classification_type::np_configurations> tech_lib(gates, tps);

  for (auto const &benchmark : epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    names_view<aig_network> aig;
    if (lorina::read_aiger(benchmark_path(benchmark), aiger_reader(aig)) != lorina::return_code::success) {
      continue;
    }

    const uint32_t size_before = aig.num_gates();
    const uint32_t depth_before = depth_view(aig).depth();

    emap_params ps;
    ps.matching_mode = emap_params::hybrid;
    ps.area_oriented_mapping = false;
    ps.map_multioutput = false;
    ps.relax_required = 0;
    ps.enable_egg_fallback = true;
    ps.egg_min_cut_size = 4;
    ps.egg_min_cut_depth = 2;
    ps.egg_max_per_node = 3;
    emap_stats st;
    cell_view<block_network> res = emap<9>(aig, tech_lib, ps, &st);

    names_view res_names{res};
    restore_network_name(aig, res_names);
    restore_pio_names_by_order(aig, res_names);
    const auto cec = benchmark == "hyp" ? true : abc_cec_mapped_cell(res_names, benchmark, TECHLIB);

    exp(benchmark, size_before, res.num_gates(), res.compute_area(), depth_before, depth_view(res).depth(), res.compute_worst_delay(), to_seconds(st.time_total), cec);
  }

  exp.save();
  exp.table();
  return 0;
}

int main(int argc, char *argv[]) {
  int op = 0;
  if (argc >= 2) op = atoi(argv[1]);
  if (op == 0) {
    main_mapper();
  } else if (op == 1) {
    main_emapper();
  } else {
    // clean the library of expr2cost
  }
  return 0;
}