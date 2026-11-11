#include "epmapping/mapped_io.hpp"
#include "epmapping/shared_sum_factor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <lorina/genlib.hpp>
#include <mockturtle/io/genlib_reader.hpp>
#include <mockturtle/networks/block.hpp>
#include <mockturtle/views/cell_view.hpp>
#include <mockturtle/views/topo_view.hpp>
#include <random>
using Network = mockturtle::cell_view<mockturtle::block_network>;

std::vector<uint64_t> evaluate(Network const &n,
                               std::vector<uint64_t> const &inputs) {
  std::vector<uint64_t> values(n.size());
  n.foreach_pi([&](auto node, uint32_t i) {
    values[n.node_to_index(node)] = inputs[i];
  });
  values[n.node_to_index(n.get_node(n.get_constant(true)))] = ~uint64_t{0};
  mockturtle::topo_view<Network> topo{n};
  topo.foreach_node([&](auto node) {
    if (n.is_constant(node) || n.is_pi(node))
      return;
    std::vector<uint64_t> args;
    n.foreach_fanin(node, [&](auto s) {
      auto v = values[n.node_to_index(n.get_node(s))];
      args.push_back(s.complement ? ~v : v);
    });
    uint64_t out = 0;
    for (uint32_t bit = 0; bit < 64; ++bit) {
      uint32_t m = 0;
      for (uint32_t p = 0; p < args.size(); ++p)
        m |= ((args[p] >> bit) & 1u) << p;
      if ((n.node_function(node)._bits[0] >> m) & 1u)
        out |= uint64_t{1} << bit;
    }
    values[n.node_to_index(node)] = out;
  });
  std::vector<uint64_t> out;
  n.foreach_po([&](auto s) {
    auto v = values[n.node_to_index(n.get_node(s))];
    out.push_back(s.complement ? ~v : v);
  });
  return out;
}
double area(Network const &n) {
  double a = 0;
  mockturtle::topo_view<Network> topo{n};
  topo.foreach_node([&](auto node) {
    if (n.has_cell(node))
      a += n.get_cell(node).area;
  });
  return a;
}
int main(int argc, char **argv) {
  assert(argc == 2 || argc == 4);
  std::vector<mockturtle::gate> gates;
  assert(lorina::read_genlib(argv[1], mockturtle::genlib_reader(gates)) ==
         lorina::return_code::success);
  auto cells = mockturtle::get_standard_cells(gates);
  auto find = [&](std::string name) {
    for (uint32_t i = 0; i < cells.size(); ++i)
      if (cells[i].name == name)
        return i;
    throw std::runtime_error(name);
  };
  auto delay = [](auto const &g, size_t p) {
    return p < g.pins.size() ? std::max(g.pins[p].rise_block_delay,
                                        g.pins[p].fall_block_delay)
                             : 0.;
  };
  if (argc == 4) {
    auto n = epmapping::read_mapped_verilog<Network>(argv[2], cells);
    auto before = area(n), d0 = n.compute_worst_delay();
    auto report = epmapping::factor_shared_sums(n, delay);
    std::cout << "PROPOSAL groups=" << report.groups
              << " roots=" << report.roots << " factors=" << report.factors
              << " before=" << before << " after=" << area(n) << " D0=" << d0
              << " D=" << n.compute_worst_delay() << "\n";
    epmapping::write_mapped_verilog(n, argv[3]);
    return 0;
  }
  for (uint32_t count : {1u, 11u}) {
    Network n{cells};
    auto create = [&](std::string name, std::vector<Network::signal> args) {
      auto i = find(name);
      auto s = n.create_node(args, cells[i].gates.front().function);
      n.add_cell(n.get_node(s), i);
      return s;
    };
    auto q = n.create_pi(), s = n.create_pi();
    std::vector<Network::signal> a, d;
    for (uint32_t i = 0; i < count; ++i) {
      a.push_back(n.create_pi());
      d.push_back(n.create_pi());
    }
    auto iq = create("INVx1_ASAP7_75t_R", {q});
    auto r = create("AND2x2_ASAP7_75t_R", {iq, s});
    auto nor = create("NOR2x1_ASAP7_75t_R", {q, s});
    for (uint32_t i = 0; i < count; ++i)
      n.create_po(create("AO222x2_ASAP7_75t_R", {a[i], q, a[i], r, d[i], nor}));
    n.create_po(iq);
    n.create_po(r);
    n.create_po(nor); // Retain the context's shared original gates.
    std::mt19937_64 rng(17);
    std::vector<std::vector<uint64_t>> samples, expected;
    std::vector<uint64_t> exhaustive(2 + 2 * count);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      uint32_t m = lane % 16;
      for (uint32_t i = 0; i < exhaustive.size(); ++i) {
        uint32_t variable = i < 2 ? i : 2 + (i % 2);
        if ((m >> variable) & 1u)
          exhaustive[i] |= uint64_t{1} << lane;
      }
    }
    samples.push_back(exhaustive);
    expected.push_back(evaluate(n, exhaustive));
    for (uint32_t trial = 0; trial < 64; ++trial) {
      std::vector<uint64_t> in(2 + 2 * count);
      for (auto &v : in)
        v = rng();
      samples.push_back(in);
      expected.push_back(evaluate(n, in));
    }
    auto before = area(n);
    auto report = epmapping::factor_shared_sums(n, delay);
    for (uint32_t trial = 0; trial < samples.size(); ++trial)
      assert(evaluate(n, samples[trial]) == expected[trial]);
    if (count == 1) {
      assert(report.roots == 0);
      assert(std::abs(area(n) - before) < 1e-8);
    } else {
      assert(report.roots == 11 && report.factors == 1);
      assert(std::abs((before - area(n)) - .40) < 1e-8);
    }
    std::cout << "PASS roots=" << count << " rewritten=" << report.roots
              << " area_gain=" << before - area(n)
              << " function_vectors=4160\n";
  }
}
