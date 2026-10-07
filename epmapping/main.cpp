#include "epmapping.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <experiments.hpp>
#include <lorina/aiger.hpp>
#include <lorina/genlib.hpp>
#include <mockturtle/algorithms/emap.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <memory>
#include <optional>

#include <lorina/super.hpp>
#include <mockturtle/io/genlib_reader.hpp>
#include <mockturtle/io/super_reader.hpp>
#include <mockturtle/io/write_verilog.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/block.hpp>
#include <mockturtle/utils/tech_library.hpp>
#include <mockturtle/views/cell_view.hpp>

namespace
{

std::string shell_quote(std::string const& value)
{
  std::string quoted{"'"};
  for (char c : value)
  {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted += c;
  }
  quoted += '\'';
  return quoted;
}

uint32_t env_u32(char const* name, uint32_t fallback)
{
  if (auto const* value = std::getenv(name))
  {
    try
    {
      return static_cast<uint32_t>(std::stoul(value));
    }
    catch (...)
    {
      return fallback;
    }
  }
  return fallback;
}

double env_f64(char const* name, double fallback)
{
  if (auto const* value = std::getenv(name))
  {
    try
    {
      return std::stod(value);
    }
    catch (...)
    {
      return fallback;
    }
  }
  return fallback;
}

bool run_abc_preprocess(std::filesystem::path const& input,
                        std::string const& script,
                        std::filesystem::path const& compressed)
{
  auto const abc = std::getenv("ABC") != nullptr ? std::getenv("ABC") : "abc";
  std::string command = shell_quote(abc);
  command += " -c \"read_aiger " + shell_quote(input.string()) + "; " + script +
             "; write_aiger " + shell_quote(compressed.string()) + "; quit\"";
  return std::system(command.c_str()) == 0;
}

/* Number of AND nodes in an AIGER file. The header (`aig M I L O A`) is ASCII even in binary AIGER
   files, so this is a cheap way to measure how much a script reduced the subject graph. */
std::optional<uint64_t> aiger_and_count(std::filesystem::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  uint64_t m = 0, i = 0, l = 0, o = 0, a = 0;
  if (!(in >> magic >> m >> i >> l >> o >> a) || magic != "aig")
    return std::nullopt;
  return a;
}

/* High-effort baseline subject graph (ASPDAC'25 "A_e"): apply `fraig; resyn2rs; fraig; compress2rs`
   repeatedly and keep the last graph that still reduced the AIG, i.e. iterate until the subject
   graph cannot be reduced any further. `EPMAPPING_ABC_ROUND` overrides the per-round script and
   `EPMAPPING_ABC_ITERS` the iteration cap. */
bool run_abc_high_effort(std::filesystem::path const& input,
                         std::filesystem::path const& compressed)
{
  char const* round_env = std::getenv("EPMAPPING_ABC_ROUND");
  std::string const round = (round_env != nullptr && *round_env != '\0')
                                ? round_env
                                : "fraig; resyn2rs; fraig; compress2rs";
  auto const max_iters = env_u32("EPMAPPING_ABC_ITERS", 20u);

  auto best = aiger_and_count(input);
  if (!best)
  {
    std::cerr << "[e] cannot read AIGER header: " << input << "\n";
    return false;
  }
  std::cout << "[i] ABC baseline: high_effort, round=\"" << round << "\", ands=" << *best << std::endl;

  if (!std::filesystem::copy_file(input, compressed,
                                  std::filesystem::copy_options::overwrite_existing))
  {
    std::cerr << "[e] cannot copy " << input << "\n";
    return false;
  }

  auto const probe = compressed.parent_path() / (compressed.stem().string() + "_probe.aig");
  bool fixpoint = max_iters == 0u;
  for (uint32_t iter = 0; iter < max_iters; ++iter)
  {
    if (!run_abc_preprocess(compressed, round, probe))
    {
      std::cerr << "[e] ABC preprocessing failed (round " << (iter + 1) << ")\n";
      return false;
    }
    auto const next = aiger_and_count(probe);
    if (!next)
    {
      std::cerr << "[e] cannot read AIGER header: " << probe << "\n";
      return false;
    }
    if (*next >= *best)
    {
      std::filesystem::remove(probe);
      std::cout << "[i] ABC round " << (iter + 1) << ": " << *best << " -> " << *next
                << " ands (no reduction); fixpoint after " << iter << " round(s)" << std::endl;
      fixpoint = true;
      break;
    }
    std::error_code ec;
    std::filesystem::rename(probe, compressed, ec);
    if (ec)
    {
      std::cerr << "[e] cannot replace " << compressed << ": " << ec.message() << "\n";
      return false;
    }
    std::cout << "[i] ABC round " << (iter + 1) << ": " << *best << " -> " << *next
              << " ands" << std::endl;
    best = next;
  }
  /* A cap hit is *not* a fixpoint: report it, so a comparison is never reported against a subject
     graph that a larger `EPMAPPING_ABC_ITERS` could still have improved. */
  if (!fixpoint)
    std::cerr << "[w] ABC high-effort baseline stopped at the iteration cap (" << max_iters
              << " rounds, ands=" << *best << "); raise EPMAPPING_ABC_ITERS for a real fixpoint\n";
  std::cout << "[i] ABC baseline ands=" << *best << std::endl;
  return true;
}

bool abc_cec_mapped_cell_unique(mockturtle::cell_view<mockturtle::block_network> const& ntk,
                                std::filesystem::path const& benchmark,
                                std::filesystem::path const& genlib,
                                std::filesystem::path const& tmp_v)
{
  mockturtle::write_verilog_with_cell(ntk, tmp_v.string());
  auto const abc = std::getenv("ABC") != nullptr ? std::getenv("ABC") : "abc";
  std::string command = shell_quote(abc) + " -q \"read_genlib " + shell_quote(genlib.string()) +
                        "; read -m " + shell_quote(tmp_v.string()) + "; cec -n " +
                        shell_quote(benchmark.string()) + "\" 2>&1";
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr)
    return false;
  std::string result;
  char buffer[512];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    result += buffer;
  pclose(pipe);
  return result.find("Networks are equivalent") != std::string::npos;
}

/* Debug helper: report what drives every primary output. `EPMAPPING_DEBUG_PO=1` enables it. */
template <typename Ntk>
void debug_po_drivers(Ntk const& ntk, char const* tag)
{
  if (env_u32("EPMAPPING_DEBUG_PO", 0u) == 0u)
    return;
  ntk.foreach_po( [&]( auto const& f, auto index ) {
    auto const n = ntk.get_node( f );
    std::cout << "[d] " << tag << " po[" << index << "] node=" << ntk.node_to_index( n )
              << " constant=" << (ntk.is_constant( n ) ? 1 : 0)
              << " pi=" << (ntk.is_pi( n ) ? 1 : 0)
              << " cell=" << (ntk.has_cell( n ) ? ntk.get_cell_index( n ) : 0u )
              << " has_cell=" << (ntk.has_cell( n ) ? 1 : 0)
              << " complemented=" << (ntk.is_complemented( f ) ? 1 : 0) << "\n";
  } );
}

} // namespace

int main(int argc, char** argv)
{
  auto const process_start = std::chrono::steady_clock::now();
  auto const elapsed = []( auto const& start ) {
    return std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
  };
  if (argc < 3)
  {
    std::cerr << "usage: epmapping_check <genlib> <input.aig> [workdir]\n";
    return 2;
  }

  std::filesystem::path const genlib_path = argv[1];
  std::filesystem::path const input_path = argv[2];
  std::filesystem::path workdir = argc >= 4
                                      ? std::filesystem::path{argv[3]}
                                      : std::filesystem::temp_directory_path() / "epmapping";
  std::filesystem::create_directories(workdir);

  /* map_egg loads its own TechLib through `MAP_EGG_TECHLIB_PATH`; keep it in sync. */
  setenv("MAP_EGG_TECHLIB_PATH", genlib_path.string().c_str(), 1);

  auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  auto const compressed_path = workdir / ("abc_" + std::to_string(stamp) + ".aig");

  /* Initial (subject) netlist. Two presets, `EPMAPPING_BASELINE`:
       - `direct`      (A_i): `read_aiger` + area-oriented `emap`, no ABC optimization at all;
       - `high_effort` (A_e, default): `read_aiger` + `fraig; resyn2rs; fraig; compress2rs` iterated
                        until the AIG cannot be reduced any further, then area-oriented `emap`;
       - `custom`: a single pass of `EPMAPPING_ABC_SCRIPT` (this reproduces the tables measured
         before the presets existed, e.g. the IWLS'24 graph `"fraig; compress2rs; resyn2rs;
         compress2rs"` or the one-pass ASPDAC'25 graph `"fraig; rw; rs; rf; resyn2rs; compress2rs"`).
     An explicit `EPMAPPING_ABC_SCRIPT` implies `custom` (the empty string implies `direct`); an
     explicit `EPMAPPING_BASELINE` always wins. */
  std::string baseline = "high_effort";
  if (char const* env = std::getenv("EPMAPPING_BASELINE"); env != nullptr && *env != '\0')
    baseline = env;
  else if (char const* env = std::getenv("EPMAPPING_ABC_SCRIPT"); env != nullptr)
    baseline = *env == '\0' ? "direct" : "custom";

  std::filesystem::path subject_path = input_path;
  if (baseline == "direct")
  {
    std::cerr << "[i] ABC baseline: direct (no ABC preprocessing)\n";
  }
  else if (baseline == "high_effort")
  {
    if (!run_abc_high_effort(input_path, compressed_path))
    {
      std::cerr << "[e] ABC preprocessing failed\n";
      return 1;
    }
    subject_path = compressed_path;
  }
  else if (baseline == "custom")
  {
    std::string const abc_script = std::getenv("EPMAPPING_ABC_SCRIPT") != nullptr
                                       ? std::getenv("EPMAPPING_ABC_SCRIPT")
                                       : "";
    std::cerr << "[i] ABC baseline: custom, script=\"" << abc_script << "\"\n";
    if (!run_abc_preprocess(input_path, abc_script, compressed_path))
    {
      std::cerr << "[e] ABC preprocessing failed\n";
      return 1;
    }
    subject_path = compressed_path;
  }
  else
  {
    std::cerr << "[e] unknown EPMAPPING_BASELINE=\"" << baseline
              << "\" (expected direct|high_effort|custom)\n";
    return 2;
  }

  double const preprocess_secs = elapsed( process_start );
  auto const library_start = std::chrono::steady_clock::now();
  std::vector<mockturtle::gate> gates;
  {
    std::ifstream input(genlib_path);
    if (lorina::read_genlib(input, mockturtle::genlib_reader(gates)) != lorina::return_code::success)
    {
      std::cerr << "[e] failed to read genlib: " << genlib_path << "\n";
      return 1;
    }
  }

  mockturtle::tech_library_params tech_params;
  tech_params.ignore_symmetries = false;
  /* A supergate library (ABC: `super -I k -L l <genlib>`) turns the technology library into a
     database of pre-mapped multi-cell sub-networks, which is what the paper's database matching
     uses. Without it `tech_library` only knows single gates. */
  mockturtle::super_lib super_spec;
  bool const have_super = [&]() {
    char const* path = std::getenv("EPMAPPING_SUPER");
    if (path == nullptr || *path == '\0')
      return false;
    /* The 16 MB database takes tens of seconds to parse, and every invocation pays it once: report
       it separately so the mapping runtime can be compared with published numbers that way too. */
    auto const t0 = std::chrono::steady_clock::now();
    std::ifstream super_in(path);
    if (lorina::read_super(super_in, mockturtle::super_reader(super_spec)) != lorina::return_code::success)
    {
      std::cerr << "[e] failed to read supergate library: " << path << "\n";
      return false;
    }
    std::cout << "[i] super library load = "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << " s" << std::endl;
    return true;
  }();
  if (have_super)
    std::cout << "[i] super library: " << super_spec.supergates.size() << " gates, max_vars="
              << super_spec.max_num_vars << std::endl;
  /* The baseline mapper always uses the plain library: adding supergates changes `emap`'s own
     mapping, and the comparison against the paper requires the baseline to stay fixed. */
  mockturtle::tech_library<9> tech_lib(gates, tech_params);
  std::unique_ptr<mockturtle::tech_library<9>> tech_lib_super;
  if (have_super)
    tech_lib_super = std::make_unique<mockturtle::tech_library<9>>(gates, super_spec, tech_params);

  /* How complete is the database over 4-input functions? (Every function that has no entry can never
     be matched, no matter how good the rest of the engine is.) */
  if ( tech_lib_super && env_u32("EPMAPPING_DB_STATS", 0u) != 0u )
  {
    uint32_t covered = 0u, with_multi = 0u;
    double area_sum = 0.0, area_max = 0.0;
    for ( uint32_t t = 0u; t < 65536u; ++t )
    {
      kitty::dynamic_truth_table tt( 4u );
      uint64_t word = t;
      kitty::create_from_words( tt, &word, &word + 1 );
      auto const fe = kitty::extend_to<6>( tt );
      auto const* list = tech_lib_super->get_supergates( fe );
      if ( list == nullptr || list->empty() )
        continue;
      ++covered;
      auto const best = ( *list )[0].area;
      area_sum += best;
      area_max = std::max( area_max, best );
      if ( ( *list )[0].area > 0.2 )
        ++with_multi; /* larger than any single cell of this library */
    }
    std::cout << "[i] db4 coverage=" << covered << "/65536"
              << " with_multi_cell=" << with_multi
              << " avg_best_area=" << ( covered ? area_sum / covered : 0.0 )
              << " max_best_area=" << area_max << std::endl;
  }

  double const library_secs = elapsed( library_start );
  auto const mapping_start = std::chrono::steady_clock::now();
  mockturtle::aig_network aig;
  if (lorina::read_aiger(subject_path, mockturtle::aiger_reader(aig)) != lorina::return_code::success)
  {
    std::cerr << "[e] failed to read subject graph: " << subject_path << "\n";
    return 1;
  }

  mockturtle::emap_params map_params;
  map_params.matching_mode = mockturtle::emap_params::hybrid;
  /* ASPDAC'25 maps the subject graph with **area-oriented** `emap` (that is the `A_e` its numbers
     are reported on), so it is the default; `EPMAPPING_AREA_MAPPING=0` switches back to the
     delay-oriented mapping. Mockturtle's area-oriented mode ignores `required_time` entirely, so
     "the best possible delay as the constraint" (the paper's wording) is nominal here — see
     `EPMAPPING_REQUIRED_FROM_DELAY`. */
  map_params.area_oriented_mapping = env_u32("EPMAPPING_AREA_MAPPING", 1u) != 0u;
  map_params.map_multioutput = env_u32("EPMAPPING_MAP_MULTIOUTPUT", 0u) != 0u;
  map_params.cut_enumeration_ps.cut_limit = static_cast<uint32_t>( env_u32("EPMAPPING_MAP_CUT_LIMIT", map_params.cut_enumeration_ps.cut_limit) );
  /* `relax_required` is a percentage relaxation of the required times; mockturtle emap can assert
     on area-oriented mapping of some designs unless it is positive. */
  /* Area-oriented mapping can assert in mockturtle unless the required times are relaxed; with
     `area_oriented_mapping` mockturtle ignores `required_time` anyway, so "fully relaxed" (100%) is
     the natural default there. */
  map_params.relax_required = env_f64("EPMAPPING_RELAX_REQUIRED",
                                      map_params.area_oriented_mapping ? 100.0 : map_params.relax_required);
  /* The paper's baseline is *area-oriented mapping under the tightest delay constraint*: "we map the
     network using area-oriented mapping and optimize the network with the most stringent delay
     constraints, i.e. we do not allow for any delay increase on the critical path." The critical path
     to hold is the one of the delay-oriented mapping, so map once for delay, then recover area
     without exceeding it. */
  if ( map_params.area_oriented_mapping && env_u32("EPMAPPING_REQUIRED_FROM_DELAY", 0u) != 0u )
  {
    auto delay_params = map_params;
    delay_params.area_oriented_mapping = false;
    auto const delay_mapped = mockturtle::emap<9>(aig, tech_lib, delay_params);
    map_params.required_time = delay_mapped.compute_worst_delay();
    map_params.relax_required = env_f64("EPMAPPING_RELAX_REQUIRED", 0.0);
    std::cout << "[i] required_time from delay-oriented mapping = " << map_params.required_time
              << " (relax " << map_params.relax_required << "%)" << std::endl;
  }

  mockturtle::emap_stats map_stats;
  auto mapped = mockturtle::emap<9>(aig, tech_lib, map_params, &map_stats);

  std::cout << "[i] emap area=" << mapped.compute_area()
            << " delay=" << mapped.compute_worst_delay()
            << " gates=" << mapped.num_gates()
            << " area_oriented=" << (map_params.area_oriented_mapping ? 1 : 0)
            << " multioutput=" << map_stats.multioutput_gates << std::endl;

  debug_po_drivers(mapped, "emap");
  double const mapping_secs = elapsed( mapping_start );

  if (env_u32("EPMAPPING_SKIP", 0u) != 0u)
  {
    std::cout << "[i] EPMAPPING_SKIP set; skipping optimizer" << std::endl;
    /* `EPMAPPING_WRITE_BASELINE=1` also writes the mapped netlist. The comparison against ABC's
       `mfs3 -a` (paper Table `tab:mfs3`) needs exactly this netlist: both optimizers start from the
       area-oriented `emap` result, and `mfs3` rewrites a *mapped* network. Off by default so the
       base configuration keeps behaving as before. */
    if ( env_u32("EPMAPPING_WRITE_BASELINE", 0u) != 0u )
    {
      auto const baseline_path = workdir / "epmapping_mapped.v";
      mockturtle::write_verilog_with_cell( mapped, baseline_path.string() );
      std::cout << "[i] wrote " << baseline_path << std::endl;
    }
    if (env_u32("EPMAPPING_BASELINE_CEC", 0u) != 0u)
    {
      bool const base_cec = abc_cec_mapped_cell_unique(
          mapped, input_path, genlib_path, workdir / ("cec_base_" + std::to_string(stamp) + ".v"));
      std::cout << "[i] baseline_cec=" << (base_cec ? "1" : "0") << std::endl;
    }
    return 0;
  }

  epmapping::params ps;
  ps.cut_size = env_u32("EPMAPPING_CUT_SIZE", ps.cut_size);
  /* `map_egg` represents a cut function as a 9-input truth table (`Tt9`) and names cut leaves
     `a..i`, so 9 is the hard ceiling of this pipeline. */
  constexpr uint32_t max_cut_size = 9u;
  if (ps.cut_size > max_cut_size)
  {
    std::cerr << "[w] EPMAPPING_CUT_SIZE=" << ps.cut_size << " exceeds the supported maximum "
              << max_cut_size << " (map_egg truth tables are 9-input); clamping\n";
    ps.cut_size = max_cut_size;
  }
  ps.cut_limit = env_u32("EPMAPPING_CUT_LIMIT", ps.cut_limit);
  ps.objective = static_cast<uint8_t>(env_u32("EPMAPPING_OBJECTIVE", ps.objective));
  ps.preserve_depth = env_u32("EPMAPPING_PRESERVE_DEPTH", ps.preserve_depth ? 1u : 0u) != 0u;
  ps.preserve_delay = env_u32("EPMAPPING_PRESERVE_DELAY", ps.preserve_delay ? 1u : 0u) != 0u;
  ps.delay_slack = env_f64("EPMAPPING_DELAY_SLACK", ps.delay_slack);
  ps.max_rounds = env_u32("EPMAPPING_ROUNDS", ps.max_rounds);
  ps.use_egg = env_u32("EPMAPPING_EGG", ps.use_egg ? 1u : 0u) != 0u;
  ps.validate_candidates = env_u32("EPMAPPING_VALIDATE", ps.validate_candidates ? 1u : 0u) != 0u;
  ps.use_dont_cares = env_u32("EPMAPPING_DC", ps.use_dont_cares ? 1u : 0u) != 0u;
  ps.synth_cache_entries = env_u32("EPMAPPING_SYN_CACHE", ps.synth_cache_entries);
  ps.egg_required = env_u32("EPMAPPING_EGG_REQ", ps.egg_required ? 1u : 0u) != 0u;
  if ( char const* min_mffc = std::getenv("EPMAPPING_EGG_MIN_MFFC") )
    ps.egg_min_mffc = std::atof( min_mffc );
  /* Structural-synthesis candidate source and the SAT don't-care refinement (see epmapping.hpp). */
  ps.synth = env_u32("EPMAPPING_SYN", ps.synth);
  ps.synth_min_leaves = env_u32("EPMAPPING_SYN_MIN_LEAVES", ps.synth_min_leaves);
  ps.synth_modes = env_u32("EPMAPPING_SYN_MODE", ps.synth_modes);
  ps.synth_exact_leaves = env_u32("EPMAPPING_SYN_EXACT_LEAVES", ps.synth_exact_leaves);
  ps.synth_exact_conflict = env_u32("EPMAPPING_SYN_EXACT_CONFLICT", ps.synth_exact_conflict);
  ps.synth_exact_mffc = env_u32("EPMAPPING_SYN_EXACT_MFFC", ps.synth_exact_mffc ? 1u : 0u) != 0u;
  /* Whole-window structural synthesis over the MFFW (see epmapping.hpp). */
  ps.window_synth = env_u32("EPMAPPING_WINDOW_SYN", ps.window_synth);
  if ( char const* g = std::getenv("EPMAPPING_WINDOW_SYN_MIN_GAIN") )
    ps.window_synth_min_gain = std::atof(g);
  ps.window_synth_max_leaves = env_u32("EPMAPPING_WINDOW_SYN_LEAVES", ps.window_synth_max_leaves);
  ps.window_synth_max_gates = env_u32("EPMAPPING_WINDOW_SYN_GATES", ps.window_synth_max_gates);
  ps.synth_use_dc = env_u32("EPMAPPING_SYN_NO_DC", 0u) == 0u;
  ps.sat_dc = env_u32("EPMAPPING_SAT_DC", ps.sat_dc ? 1u : 0u) != 0u;
  ps.sat_dc_calls = env_u32("EPMAPPING_SAT_DC_CALLS", ps.sat_dc_calls);
  ps.sat_dc_max_nodes = env_u32("EPMAPPING_SAT_DC_MAX_NODES", ps.sat_dc_max_nodes);
  if ( char const* w = std::getenv( "EPMAPPING_DC_WINDOW" ) )
    ps.dc_window = ( std::string{ w } == "per_node" ) ? epmapping::params::dc_window_mode::per_node
                                                     : epmapping::params::dc_window_mode::per_cut;
  ps.no_delay_regression = env_u32("EPMAPPING_NO_DELAY_REGRESSION", ps.no_delay_regression ? 1u : 0u) != 0u;
  if ( ps.no_delay_regression )
    ps.delay_slack = 0.0;
  if ( ps.no_delay_regression )
    ps.preserve_delay = true;
  ps.dc_max_free = env_u32("EPMAPPING_DC_FREE", ps.dc_max_free);
  ps.dc_entry_limit = env_u32("EPMAPPING_DC_ENTRIES", ps.dc_entry_limit);
  if ( char const* sm = std::getenv("EPMAPPING_SINGLE_MATCHER") )
  {
    auto const m = std::string( sm );
    if ( m == "index" )
      ps.single = epmapping::params::single_matcher::index;
    else if ( m == "brute" )
      ps.single = epmapping::params::single_matcher::brute;
    else if ( m == "hybrid" )
      ps.single = epmapping::params::single_matcher::hybrid;
    else
      std::cerr << "[w] unknown EPMAPPING_SINGLE_MATCHER=" << m << " (index|brute|hybrid), keeping default\n";
  }
  ps.single_entries = env_u32("EPMAPPING_SINGLE_ENTRIES", ps.single_entries);
  ps.single_free = env_u32("EPMAPPING_SINGLE_FREE", ps.single_free);
  if ( char const* dc_budget = std::getenv("EPMAPPING_DC_TIME_BUDGET") )
    ps.dc_time_budget = std::atof(dc_budget);
  ps.verbose = env_u32("EPMAPPING_VERBOSE", 0u) != 0u;

  /* Instantiate up to the 9-input ceiling so `EPMAPPING_CUT_SIZE` can be raised; the default
     `params::cut_size` stays 6, so default behaviour is unchanged. */
  auto const optimizer_start = std::chrono::steady_clock::now();
  epmapping::optimizer<max_cut_size> optimizer{mapped, ps, tech_lib_super ? tech_lib_super.get() : &tech_lib};
  std::cout << "[i] epmapping running (objective=" << static_cast<unsigned>(ps.objective)
            << ", preserve_depth=" << (ps.preserve_depth ? 1 : 0)
            << ", preserve_delay=" << (ps.preserve_delay ? 1 : 0)
            << ", delay_slack=" << ps.delay_slack
            << ", max_rounds=" << ps.max_rounds
            << ", egg=" << (ps.use_egg ? 1 : 0)
            << ", dc=" << (ps.use_dont_cares ? 1 : 0)
            << ", synth_cache=" << ps.synth_cache_entries
            << ", dc_window="
            << ( ps.dc_window == epmapping::params::dc_window_mode::per_node ? "per_node" : "per_cut" )
            << ", single_matcher="
            << ( ps.single == epmapping::params::single_matcher::brute
                     ? "brute"
                     : ( ps.single == epmapping::params::single_matcher::hybrid ? "hybrid" : "index" ) )
            << ")" << std::endl;
  reset_mapping_counters();
  auto const result = optimizer.run();
  double const optimizer_secs = elapsed( optimizer_start );
  std::cout << "[i] epmapping synth_cache lookups=" << result.synth_cache_lookups
            << " hits=" << result.synth_cache_hits << std::endl;
  std::cout << "[i] map_egg paths library_hits=" << mapping_library_hits()
            << " egg_calls=" << mapping_egg_calls()
            << " egg_cache_hits=" << mapping_egg_cache_hits() << std::endl;
  std::cout << "[i] epmapping rounds=" << result.rounds << " replacements_per_round=";
  for (size_t i = 0u; i < result.replacements_per_round.size(); ++i)
    std::cout << (i ? "," : "") << result.replacements_per_round[i];
  std::cout << std::endl;

  std::cout << "[i] epmapping roots=" << result.roots
            << " cuts=" << result.cuts
            << " dc_cuts=" << result.dc_cuts
            << " dc_cuts_over_free=" << result.dc_cuts_over_free
            << " dc_free_hist=";
  for ( size_t i = 0u; i < result.dc_free_hist.size(); ++i )
    std::cout << ( i == 0u ? "" : "," ) << result.dc_free_hist[i];
  std::cout << " rejected=" << result.rejected_expressions << "\n";
  std::cout << "[i] epmapping replacements=" << result.replacements
            << " (egg=" << result.large_replacements
            << ", single=" << result.local_replacements
            << ", multi=" << result.multi_cell_replacements
            << ", synth=" << result.synth_replacements
            << ", window=" << result.window_synth_replacements
            << ")\n";
  std::cout << "[i] epmapping phase: replacements with input negation="
            << result.replacements_with_input_negation << " with inverted output="
            << result.replacements_with_output_inversion << "\n";
  std::cout << "[i] epmapping functional_rejections=" << result.functional_rejections
            << " build_rejections=" << result.build_rejections
            << " cycle_rejections=" << result.cycle_rejections << "\n";
  std::cout << "[i] epmapping staged library_cuts=" << result.staged_library_hits
            << " mffc_cut_egg_roots=" << result.mffc_cut_egg_roots
            << " egg_covers=" << result.egg_covers
            << " reject_depth=" << result.staged_depth_rejections
            << " reject_delay=" << result.staged_delay_rejections
            << " reject_objective=" << result.staged_objective_rejections
            << " reject_objective_by_source=";
  for ( int i = 0; i < 11; ++i )
    std::cout << ( i == 0 ? "" : "," ) << result.staged_objective_by_source[i];
  std::cout << "\n";
  std::cout << "[i] epmapping staged_secs cut_enum=" << result.secs_cut_enum
            << " care=" << result.secs_care
            << " stage1=" << result.secs_stage1
            << " stage2=" << result.secs_stage2
            << " egg=" << result.secs_egg
            << " mffc=" << result.secs_mffc
            << " care_cache=" << result.care_hits << "/" << result.care_lookups
            << " care_mismatch=" << result.care_mismatches
            << " dc_cut=" << result.secs_dc_cut
            << " dc_collect=" << result.secs_dc_collect
            << " dc_sim=" << result.secs_dc_sim
            << " dc_window=" << result.secs_dc_window
            << " dc_cone=" << result.secs_dc_cone
            << " win_calls=" << result.dc_window_calls
            << " win_gates_avg=" << ( result.dc_window_calls ? result.dc_window_nodes / result.dc_window_calls : 0u )
            << " win_gates_max=" << result.dc_window_nodes_max
            << " node_windows=" << result.dc_node_windows
            << " node_window_cuts=" << result.dc_node_window_cuts
            << " node_window_no_dc=" << result.dc_node_window_no_dc
            << " node_window_pis_max=" << result.dc_node_window_pis_max << "\n";
  std::cout << "[i] epmapping window_synth windows=" << result.window_synth_windows
            << " applied=" << result.window_synth_applied << " gain=" << result.window_synth_gain
            << " actual=" << result.window_synth_actual << " gates=" << result.window_synth_replacements
            << " abandoned=" << result.window_synth_abandoned
            << " delay_rejected=" << result.window_synth_delay_rejected
            << " secs=" << result.secs_window_synth << "\n";
  std::cout << "[i] epmapping synth roots=" << result.synth_roots << " covers=" << result.synth_covers
            << " wins=" << result.synth_replacements << " secs=" << result.secs_synth << "\n";
  std::cout << "[i] epmapping sat_dc cuts=" << result.dc_sat_cuts << " calls=" << result.dc_sat_calls
            << " cleared=" << result.dc_sat_cleared << " budget_hits=" << result.dc_sat_budget_hits
            << " secs=" << result.secs_dc_sat
            << " dc_time_budget_stops=" << result.dc_budget_stops << "\n";
  std::cout << "[i] epmapping area " << result.area_before << " -> " << result.area_after
            << " delay " << result.delay_before << " -> " << result.delay_after << "\n";
  for (auto const& [reason, count] : result.failure_reasons)
    std::cout << "[i] epmapping_failure_reason " << reason << "=" << count << "\n";
  if (ps.verbose)
  {
    for (auto const& expression : result.failure_examples)
      std::cout << "[i] epmapping_failure_example " << expression << "\n";
  }

  auto const validation_start = std::chrono::steady_clock::now();
  auto const verilog_path = workdir / "epmapping_mapped.v";
  debug_po_drivers(optimizer.ntk(), "opt");
  mockturtle::write_verilog_with_cell(optimizer.ntk(), verilog_path.string());
  std::cout << "[i] wrote " << verilog_path << "\n";

  bool const cec = abc_cec_mapped_cell_unique(optimizer.ntk(), input_path, genlib_path,
                                              workdir / ("cec_" + std::to_string(stamp) + ".v"));
  std::cout << "[i] cec=" << (cec ? "1" : "0") << "\n";
  std::cout << "[i] epmapping timing preprocess=" << preprocess_secs
            << " library=" << library_secs << " mapping=" << mapping_secs
            << " optimizer=" << optimizer_secs << " validation=" << elapsed( validation_start )
            << " total=" << elapsed( process_start ) << std::endl;
  return cec ? 0 : 3;
}
