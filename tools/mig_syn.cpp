
#include <fmt/format.h>
#include <lorina/pla.hpp>
#include <mockturtle/mockturtle.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "db_string.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define NULLL_PATH "NUL"
#define RM_CMD "del"
#define PATH_SEP "\\"
#else
#define NULLL_PATH "/dev/null"
#define RM_CMD "rm"
#define PATH_SEP "/"
#endif

template <class Ntk>
inline bool abc_cec_impl(Ntk const &ntk, std::string const &benchmark_fullpath) {
  mockturtle::write_bench(ntk, "/tmp/test.bench");
  std::string command = fmt::format("yosys-abc -q \"cec -n {} /tmp/test.bench\"", benchmark_fullpath);

  std::array<char, 128> buffer;
  std::string result;
#if WIN32
  std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
#else
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
#endif
  if (!pipe) {
    throw std::runtime_error("popen() failed");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  /* search for one line which says "Networks are equivalent" and ignore all other debug output from ABC */
  std::stringstream ss(result);
  std::string line;
  while (std::getline(ss, line, '\n')) {
    if (line.size() >= 23u && line.substr(0u, 23u) == "Networks are equivalent") {
      return true;
    }
  }

  return false;
}

struct opt_params_t {
  uint32_t optimization_rounds{1};
  uint32_t max_remapping_rounds{3};
  uint32_t max_resynthesis_rounds{10};
  std::unordered_map<uint32_t, double> gate_costs{{3u, 6.0}, {5u, 10.0}};
  std::unordered_map<uint32_t, double> splitters{{1u, 2.0}, {4u, 2.0}};
  mutable mockturtle::aqfp_db<> db{gate_costs, splitters};
  mutable mockturtle::aqfp_db<> db_last{gate_costs, splitters};
  mockturtle::aqfp_node_resyn_strategy strategy{mockturtle::aqfp_node_resyn_strategy::area};
  std::string lutmap{"abc"};
  mockturtle::aqfp_assumptions assume{true, true, true, 4u};
};

struct opt_stats_t {
  uint32_t maj3_after_remapping;
  uint32_t level_after_remapping;
  uint32_t maj3_after_exact;
  uint32_t maj5_after_exact;
  uint32_t jj_after_exact;
  uint32_t jj_level_after_exact;
};

std::string strRand(uint32_t length) {
  char tmp;
  std::string buffer;
  std::random_device rd;
  std::default_random_engine random(rd());
  for (uint32_t i = 0u; i < length; i++) {
    tmp = random() % 36u;
    if (tmp < 10) {
      tmp += '0';
    } else {
      tmp -= 10;
      tmp += 'A';
    }
    buffer += tmp;
  }
  return buffer;
}

template <typename Ntk>
mockturtle::klut_network lut_map_abc(Ntk const &ntk, uint32_t k = 4, std::string name = {}) {
  std::string tempfile1 = "/tmp/" + strRand(10) + name + "_1.bench";
  std::string tempfile2 = "/tmp/" + strRand(10) + name + "_2.bench";

  mockturtle::write_bench(ntk, tempfile1);

  if (-1 == system(fmt::format("yosys-abc -q \"{}; &get; &if -K {}; &put; write_bench {}\" >> {} 2>&1", tempfile1, k, tempfile2, NULLL_PATH).c_str())) {
    std::cout << "yosys &if -K: error" << std::endl;
  }

  mockturtle::klut_network klut;
  if (lorina::read_bench(tempfile2, mockturtle::bench_reader(klut)) != lorina::return_code::success) {
    std::cout << "FATAL NEW LUT MAP - Reading mapped network failed! " << tempfile1 << " " << tempfile2 << std::endl;
    std::abort();
    return klut;
  }

  if (-1 == system(fmt::format("{} {}", RM_CMD, tempfile1).c_str())) {
    std::cout << "rm error" << std::endl;
  }
  if (-1 == system(fmt::format("{} {}", RM_CMD, tempfile2).c_str())) {
    std::cout << "rm error" << std::endl;
  }
  return klut;
}

mockturtle::mig_network remapping_round(mockturtle::mig_network const &ntk, mockturtle::exact_library<mockturtle::mig_network> const &exact_lib, opt_params_t const &opt_params, opt_stats_t &stats) {
  mockturtle::map_params psm;
  psm.skip_delay_round = false;
  mockturtle::map_stats stm;

  mockturtle::mig_network mig = mockturtle::cleanup_dangling(ntk);

  /* initial mig mapping, depth-oriented */
  for (auto i = 0u; i < opt_params.max_remapping_rounds; ++i) {
    uint32_t old_mig_depth = mockturtle::depth_view(ntk).depth();
    uint32_t old_mig_size = ntk.num_gates();

    mockturtle::mig_network mig_map = mockturtle::map(mig, exact_lib, psm, &stm);

    if (mockturtle::depth_view(mig_map).depth() > old_mig_depth ||
        (mockturtle::depth_view(mig_map).depth() == old_mig_depth && mig_map.num_gates() >= old_mig_size)) {
      break;
    }
    mig = mockturtle::cleanup_dangling(mig_map);
  }

  stats.maj3_after_remapping = mig.num_gates();
  stats.level_after_remapping = mockturtle::depth_view(mig).depth();

  return mig;
}

/* Supplementary functions for AQFP resynthesis */
template <typename Result>
bool has_better_cost(Result &current, Result &previous) {
  if (current.first < previous.first)
    return true;

  if (current.first > previous.first)
    return false;

  return current.second < previous.second;
}

template <typename Result>
bool has_better_level(Result &current, Result &previous) {
  if (current.second < previous.second)
    return true;

  if (current.second > previous.second)
    return false;

  return current.first < previous.first;
}

template <typename T>
auto count_majorities(T &ntk) {
  std::unordered_map<uint32_t, uint32_t> counts;
  ntk.foreach_gate([&](auto n) { counts[ntk.fanin_size(n)]++; });
  return counts;
}

template <typename Ntk>
mockturtle::aqfp_network aqfp_exact_resynthesis(Ntk &ntk, opt_params_t const &params, opt_stats_t &stats) {
  mockturtle::aqfp_network_cost cost_fn(params.assume, params.gate_costs, params.splitters);
  mockturtle::aqfp_node_resyn n_resyn(params.db, {params.assume, params.splitters, params.strategy});
  mockturtle::aqfp_node_resyn n_resyn_last(params.db_last, {params.assume, params.splitters, params.strategy});
  mockturtle::aqfp_fanout_resyn fo_resyn(params.assume);

  mockturtle::klut_network klut;

  if (params.lutmap == "abc") {
    klut = lut_map_abc(ntk, 4);
  } else {
    assert(false);
  }

  mockturtle::aqfp_network aqfp;
  mockturtle::aqfp_network aqfp_last;

  auto res = mockturtle::aqfp_resynthesis(aqfp, klut, n_resyn, fo_resyn);
  auto res_last = mockturtle::aqfp_resynthesis(aqfp_last, klut, n_resyn_last, fo_resyn);
  std::pair<double, uint32_t> cost_level = {cost_fn(aqfp_last, res_last.node_level, res_last.po_level), res_last.critical_po_level()};

  mockturtle::aqfp_network best_aqfp = aqfp_last;
  auto best_res = res_last;
  auto best_cost_level = cost_level;

  for (auto i = 2u; i <= params.max_resynthesis_rounds; i++) {

    if (params.lutmap == "abc") {
      klut = lut_map_abc(aqfp, 4);
    } else {
      assert(false);
    }

    aqfp = mockturtle::aqfp_network();
    aqfp_last = mockturtle::aqfp_network();
    res = mockturtle::aqfp_resynthesis(aqfp, klut, n_resyn, fo_resyn);
    res_last = mockturtle::aqfp_resynthesis(aqfp_last, klut, n_resyn_last, fo_resyn);
    cost_level = {cost_fn(aqfp_last, res_last.node_level, res_last.po_level), res_last.critical_po_level()};

    if (params.strategy == mockturtle::aqfp_node_resyn_strategy::area) {
      if (has_better_cost(cost_level, best_cost_level)) {
        best_aqfp = aqfp_last;
        best_res = res_last;
        best_cost_level = cost_level;
      }
    } else {
      assert(params.strategy == mockturtle::aqfp_node_resyn_strategy::delay);
      if (has_better_level(cost_level, best_cost_level)) {
        best_aqfp = aqfp_last;
        best_res = res_last;
        best_cost_level = cost_level;
      }
    }
  }

  auto maj_counts = count_majorities(best_aqfp);
  stats.maj3_after_exact = maj_counts[3];
  stats.maj5_after_exact = maj_counts[5];
  stats.jj_after_exact = static_cast<uint32_t>(best_cost_level.first);
  stats.jj_level_after_exact = best_cost_level.second;

  return best_aqfp;
}

void MIGAQFPReSyn(char *pla_path, char *mig_path) {
  std::string tmpf = "/tmp/" + strRand(10) + ".aig";
  if (-1 == system(fmt::format("yosys-abc -c \"read {}; strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; write_aiger {};\"", pla_path, tmpf).c_str())) {
    std::cout << "ABC error!" << std::endl;
  }

  mockturtle::mig_network mig;
  if (lorina::read_aiger(tmpf, mockturtle::aiger_reader(mig)) != lorina::return_code::success) {
    printf("Parsing the pla %s failed.", pla_path);
  }

  /* library to map to MIGs */
  mockturtle::mig_npn_resynthesis resyn{true};
  mockturtle::exact_library_params eps;
  eps.np_classification = true;
  mockturtle::exact_library<mockturtle::mig_network> exact_lib(resyn, eps);
  std::stringstream db3_str(mockturtle::aqfp_db3_str);
  opt_params_t opt_params;
  opt_params.db.load_db(db3_str);
  std::stringstream db5_str(mockturtle::aqfp_db3_str);
  opt_params.db_last.load_db(db5_str);
  opt_stats_t opt_stats;

  mockturtle::aqfp_network aqfp;
  /* main optimization loop */
  for (auto i = 0u; i < opt_params.optimization_rounds; ++i) {
    auto mig_opt = remapping_round(mig, exact_lib, opt_params, opt_stats);
    aqfp = aqfp_exact_resynthesis(mig_opt, opt_params, opt_stats);
  }
  mig = mockturtle::cleanup_dangling<mockturtle::aqfp_network, mockturtle::mig_network>(aqfp);

  assert(abc_cec_impl(mig, pla_path));
  mockturtle::write_verilog(mig, mig_path);
}

void MIGReSyn(char *pla_path, char *mig_path) {
  std::string tmpf = "/tmp/" + strRand(10) + ".aig";
  if (-1 == system(fmt::format("yosys-abc -c \"read {}; strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; write_aiger {};\"", pla_path, tmpf).c_str())) {
    std::cout << "ABC error!" << std::endl;
  }

  mockturtle::mig_network mig;
  if (lorina::read_aiger(tmpf, mockturtle::aiger_reader(mig)) != lorina::return_code::success) {
    printf("Parsing the pla %s failed.", pla_path);
  }

  mockturtle::lut_mapping_params ps;
  ps.cut_enumeration_ps.cut_size = 4u;
  mockturtle::lut_mapping_stats st;
  mockturtle::mapping_view<mockturtle::mig_network, true> mapped_mig{mig};
  mockturtle::lut_mapping<decltype(mapped_mig), true>(mapped_mig, ps, &st);
  const auto klut = *mockturtle::collapse_mapped_network<mockturtle::klut_network>(mapped_mig);

  mockturtle::node_resynthesis_stats nrst;
  mockturtle::mig_npn_resynthesis resyn_mig;
  mockturtle::mig_network mig2 = mockturtle::node_resynthesis<mockturtle::mig_network>(klut, resyn_mig, {}, &nrst);
  mig = cleanup_dangling(mig);

  assert(abc_cec_impl(mig, pla_path));
  mockturtle::write_verilog(mig, mig_path);
}

void MIGReSub(char *pla_path, char *mig_path) {
  std::string tmpf = "/tmp/" + strRand(10) + ".aig";
  if (-1 == system(fmt::format("yosys-abc -c \"read {}; strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; write_aiger {};\"", pla_path, tmpf).c_str())) {
    std::cout << "ABC error!" << std::endl;
  }

  mockturtle::mig_network mig;
  if (lorina::read_aiger(tmpf, mockturtle::aiger_reader(mig)) != lorina::return_code::success) {
    printf("Parsing the pla %s failed.", pla_path);
  }

  mockturtle::resubstitution_params ps;
  mockturtle::resubstitution_stats st;
  ps.max_pis = 8u;
  ps.max_inserts = 1u;
  ps.progress = false;

  mockturtle::depth_view depth_mig{mig};
  mockturtle::fanout_view fanout_mig{depth_mig};
  mig_resubstitution(fanout_mig, ps, &st);
  mig = cleanup_dangling(mig);

  assert(abc_cec_impl(mig, pla_path));
  mockturtle::write_verilog(mig, mig_path);
}

void MIGLSOracle(char *pla_path, char *mig_path, char *LSOracle = nullptr) {
  std::string tmpf = "/tmp/" + strRand(10) + ".v";
  if (-1 == system(fmt::format("yosys-abc -c \"read {}; strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; write_verilog {};\"", pla_path, tmpf).c_str())) {
    std::cout << "ABC error!" << std::endl;
  }

  if (-1 == system(fmt::format("{} -c \"read_verilog {} -m; migscript; write_verilog {} -m;\"", LSOracle ? LSOracle : "./third-party/lsoracle", tmpf, mig_path).c_str())) {
    std::cout << "LSOracle error!" << std::endl;
  }
}

PYBIND11_MODULE(MIGPy, m) {
  m.doc() = R"pbdoc(
        MIGPy
        -----------------------
        .. pla_path:: input pla file path
        .. mig_path:: output verilog file path
        .. return:: list
    )pbdoc";

  m.def("MIGAQFPReSyn", &MIGAQFPReSyn, R"pbdoc( generate the initial MIG netlist using mockturtle with AQFP resynthesis )pbdoc", pybind11::arg("pla_path"), pybind11::arg("mig_path"));
  m.def("MIGReSyn", &MIGReSyn, R"pbdoc( generate the initial MIG netlist using mockturtle with node resynthesis )pbdoc", pybind11::arg("pla_path"), pybind11::arg("mig_path"));
  m.def("MIGReSub", &MIGReSub, R"pbdoc( generate the initial MIG netlist using mockturtle with MIG resubstitution )pbdoc", pybind11::arg("pla_path"), pybind11::arg("mig_path"));
  m.def("MIGLSOracle", &MIGLSOracle, R"pbdoc( generate the initial MIG netlist using LSOracle )pbdoc", pybind11::arg("pla_path"), pybind11::arg("mig_path"), pybind11::arg("LSOracle") = nullptr);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}