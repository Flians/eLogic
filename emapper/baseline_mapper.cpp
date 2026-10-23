// Standalone, stock mockturtle mapper; no map_egg path or preprocessing.
#include <mockturtle/algorithms/mapper.hpp>
#include <mockturtle/algorithms/emap.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/io/genlib_reader.hpp>
#include <mockturtle/io/write_verilog.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/utils/name_utils.hpp>
#include <mockturtle/views/names_view.hpp>
#include <lorina/aiger.hpp>
#include <lorina/genlib.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "Usage: baseline_mapper <genlib> <aig> <mapped.v> <area|delay> [map|emap]\n";
    return 2;
  }
  using namespace mockturtle;
  std::vector<gate> gates;
  std::ifstream library(argv[1]);
  if (lorina::read_genlib(library, genlib_reader(gates)) != lorina::return_code::success) return 1;
  names_view<aig_network> aig;
  if (lorina::read_aiger(argv[2], aiger_reader(aig)) != lorina::return_code::success) return 1;
  auto start = std::chrono::steady_clock::now();
  std::string objective(argv[4]);
  if (objective != "area" && objective != "delay") return 2;
  if (argc == 6 && std::string(argv[5]) == "emap") {
    tech_library_params library_params;
    library_params.load_multioutput_gates = true;
    tech_library<6, classification_type::np_configurations> tech(gates, library_params);
    emap_params params;
    params.area_oriented_mapping = objective == "area";
    params.map_multioutput = true;
    emap_stats stats;
    auto result = emap(aig, tech, params, &stats);
    if (stats.mapping_error) return 1;
    names_view named{result};
    named.set_network_name("mapped");
    restore_pio_names_by_order(aig, named);
    write_verilog_with_cell(named, std::string(argv[3]));
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"area\":" << stats.area << ",\"delay\":" << stats.delay
              << ",\"mapping_secs\":" << seconds << "}\n";
    return 0;
  }
  if (argc == 6 && std::string(argv[5]) != "map") return 2;
  tech_library<6> tech(gates);
  map_params params;
  if (objective == "area") {
    params.skip_delay_round = true;
    params.required_time = std::numeric_limits<double>::max();
  } else if (objective != "delay") return 2;
  map_stats stats;
  auto result = map(aig, tech, params, &stats);
  if (stats.mapping_error) return 1;
  names_view named{result};
  named.set_network_name("mapped");
  restore_pio_names_by_order(aig, named);
  write_verilog_with_binding(named, std::string(argv[3]));
  double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "{\"area\":" << stats.area << ",\"delay\":" << stats.delay
            << ",\"mapping_secs\":" << seconds << "}\n";
}
