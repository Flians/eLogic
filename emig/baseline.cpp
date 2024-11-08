/* mockturtle: C++ logic network library
 * Copyright (C) 2018-2023  EPFL
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include <experiments.hpp>
#include <lorina/aiger.hpp>
#include <mockturtle/algorithms/aig_balancing.hpp>
#include <mockturtle/algorithms/balancing.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/node_resynthesis/exact.hpp>
#include <mockturtle/algorithms/node_resynthesis/mig_npn.hpp>
#include <mockturtle/algorithms/node_resynthesis/xag_npn.hpp>
#include <mockturtle/algorithms/xag_balancing.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/utils/tech_library.hpp>
#include <mockturtle/views/depth_view.hpp>

#include <fmt/format.h>
#include <string>

#include "baseline.hpp"
// #include "partial_simulation.hpp"

#define EXPERIMENTS_PATH "tools/mockturtle/experiments/"
#define CUT_SIZE 4u

void main_aig(const bool use_dc) {
  using namespace experiments;
  using namespace mockturtle;

  experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool>
      exp(fmt::format("rewrite_aig_{}", use_dc ? "dc" : "nodc"), "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  xag_npn_resynthesis<aig_network, xag_network, xag_npn_db_kind::aig_complete> resyn;
  exact_library_params ps_exact;
  ps_exact.compute_dc_classes = use_dc;
  exact_library<aig_network, CUT_SIZE> exact_lib(resyn, ps_exact);

  for (auto const &benchmark : epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    std::string benchmark_path = fmt::format("{}benchmarks/{}.aig", EXPERIMENTS_PATH, benchmark);

    aig_network aig;
    if (lorina::read_aiger(benchmark_path, aiger_reader(aig)) != lorina::return_code::success) {
      continue;
    }

    uint32_t const size_before = aig.num_gates();
    uint32_t const depth_before = depth_view(aig).depth();

    baseline::rewrite_params ps;
    baseline::rewrite_stats st;
    ps.use_dont_cares = use_dc;
    ps.window_size = 8u;
    ps.cut_enumeration_ps.cut_size = CUT_SIZE;
    ps.cut_enumeration_ps.cut_limit = 8u;
    baseline::rewrite(aig, exact_lib, ps, &st);

    bool const cec = true; // abc_cec_impl(aig, benchmark_path);
    exp(benchmark, size_before, aig.num_gates(), depth_before, depth_view(aig).depth(), to_seconds(st.time_total), cec);
  }

  exp.save();
  exp.table();
}

void main_mig(const bool use_dc) {

  using namespace experiments;
  using namespace mockturtle;

  experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool>
      exp(fmt::format("rewrite_mig_{}", use_dc ? "dc" : "nodc"), "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  mig_npn_resynthesis resyn;
  exact_library_params ps_exact;
  ps_exact.compute_dc_classes = use_dc;
  exact_library<mig_network, CUT_SIZE> exact_lib(resyn, ps_exact);

  for (auto const &benchmark : epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    std::string benchmark_path = fmt::format("{}benchmarks/{}.aig", EXPERIMENTS_PATH, benchmark);

    mig_network mig;
    if (lorina::read_aiger(benchmark_path, aiger_reader(mig)) != lorina::return_code::success) {
      continue;
    }

    uint32_t const size_before = mig.num_gates();
    uint32_t const depth_before = depth_view(mig).depth();

    baseline::rewrite_params ps;
    baseline::rewrite_stats st;
    ps.use_dont_cares = use_dc;
    ps.window_size = 8u;
    ps.cut_enumeration_ps.cut_size = CUT_SIZE;
    ps.cut_enumeration_ps.cut_limit = 8u;
    baseline::rewrite(mig, exact_lib, ps, &st);

    bool const cec = true; // abc_cec_impl(aig, benchmark_path);
    exp(benchmark, size_before, mig.num_gates(), depth_before, depth_view(mig).depth(), to_seconds(st.time_total), cec);
  }

  exp.save();
  exp.table();
}

void main_xag(const bool use_dc) {
  using namespace experiments;
  using namespace mockturtle;

  experiment<std::string, uint32_t, uint32_t, uint32_t, uint32_t, float, bool>
      exp(fmt::format("rewrite_xag_{}", use_dc ? "dc" : "nodc"), "benchmark", "size_before", "size_after", "depth_before", "depth_after", "runtime", "equivalent");

  xag_npn_resynthesis<xag_network, xag_network, xag_npn_db_kind::xag_complete> resyn;
  exact_library_params ps_exact;
  ps_exact.compute_dc_classes = use_dc;
  exact_library<xag_network, CUT_SIZE> exact_lib(resyn, ps_exact);

  for (auto const &benchmark : epfl_benchmarks()) {
    fmt::print("[i] processing {}\n", benchmark);
    std::string benchmark_path = fmt::format("{}benchmarks/{}.aig", EXPERIMENTS_PATH, benchmark);

    xag_network xag;
    if (lorina::read_aiger(benchmark_path, aiger_reader(xag)) != lorina::return_code::success) {
      continue;
    }

    uint32_t const size_before = xag.num_gates();
    uint32_t const depth_before = depth_view(xag).depth();

    baseline::rewrite_params ps;
    baseline::rewrite_stats st;
    ps.use_dont_cares = use_dc;
    ps.window_size = 8u;
    ps.cut_enumeration_ps.cut_size = CUT_SIZE;
    ps.cut_enumeration_ps.cut_limit = 8u;
    baseline::rewrite(xag, exact_lib, ps, &st);

    bool const cec = true; // abc_cec_impl(aig, benchmark_path);
    exp(benchmark, size_before, xag.num_gates(), depth_before, depth_view(xag).depth(), to_seconds(st.time_total), cec);
  }

  exp.save();
  exp.table();
}

int main(int argc, char *argv[]) {
  int op = 0;
  if (argc > 1)
    op = atoi(argv[1]);
  bool use_dc = false;
  if (argc > 2)
    use_dc = atoi(argv[2]);
  if (op == 0) {
    main_aig(use_dc);
  } else if (op == 1) {
    main_mig(use_dc);
  } else {
    main_xag(use_dc);
  }
  return 0;
}