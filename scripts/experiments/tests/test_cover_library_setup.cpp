// Minimal real fallback-path probe. This is not a circuit-quality experiment.
#include "map_egg/src/lib.rs.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <limits>
int main() {
  auto wall = std::chrono::steady_clock::now();
  auto cpu = std::clock();
  // The malformed term deliberately reaches fallback but has no cover to
  // search.
  auto covers = cut_mapping_egg_arr(
      "(% a b)", nullptr, 0, 1, std::numeric_limits<float>::infinity(), 1, 0);
  auto stats = mapping_profile_stats();
  std::cout << "wall_seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             wall)
                   .count()
            << " cpu_seconds=" << double(std::clock() - cpu) / CLOCKS_PER_SEC
            << " returned_covers=" << covers.size() << '\n';
  std::cout.write(stats.data(), stats.size());
  std::cout << '\n';
}
