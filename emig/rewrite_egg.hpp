#pragma once

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_enumeration/rewrite_cut.hpp>
#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/algorithms/simulation.hpp>
#include <mockturtle/traits.hpp>
#include <mockturtle/utils/cost_functions.hpp>
#include <mockturtle/utils/node_map.hpp>
#include <mockturtle/utils/stopwatch.hpp>
#include <mockturtle/views/color_view.hpp>
#include <mockturtle/views/cut_view.hpp>
#include <mockturtle/views/depth_view.hpp>
#include <mockturtle/views/fanout_view.hpp>
#include <mockturtle/views/window_view.hpp>

#include <fmt/format.h>
#include <kitty/dynamic_truth_table.hpp>
#include <kitty/npn.hpp>
#include <kitty/operations.hpp>
#include <kitty/static_truth_table.hpp>

#include "egg_view.hpp"
#include "mig_egg/src/lib.rs.h"
#include "rust/cxx.h"
#include <experiments.hpp>

namespace mockturtle {

  /* forward declarations */
  /*! \cond PRIVATE */
  template <typename Ntk, uint32_t NumVars, bool ComputeTruth, typename CutData>
  struct dynamic_network_cuts2;

  namespace detail2 {
    template <typename Ntk, uint32_t NumVars, bool ComputeTruth, typename CutData>
    class dynamic_cut_enumeration_impl;
  }
  /*! \endcond */

  /*! \brief Dynamic cut database for a network.
   *
   * Struct `dynamic_network_cuts2` contains a cut database and can be queried
   * to return all cuts of a node, or the function of a cut (if it was computed).
   *
   * Comparing to `network_cuts`, it supports dynamic allocation of cuts for
   * networks in expansion. Moreover, it uses static truth tables instead of
   * dynamic truth tables to speed-up the truth table computation.
   *
   * An instance of type `dynamic_network_cuts2` can only be constructed from the
   * `dynamic_cut_enumeration_impl` algorithm.
   */
  template <typename Ntk, uint32_t NumVars, bool ComputeTruth, typename CutData>
  struct dynamic_network_cuts2 {
  public:
    static constexpr uint32_t max_cut_num = 16u;
    using cut_t = cut_type<ComputeTruth, CutData>;
    using cut_set_t = cut_set<cut_t, max_cut_num>;
    static constexpr bool compute_truth = ComputeTruth;

  public:
    explicit dynamic_network_cuts2(uint32_t size) : _cuts(size) {
      kitty::static_truth_table<NumVars> zero, proj;
      kitty::create_nth_var(proj, 0u);

      _truth_tables.insert(zero);
      _truth_tables.insert(proj);
    }

  public:
    /*! \brief Returns the cut set of a node */
    cut_set_t &cuts(uint32_t node_index) {
      if (node_index >= _cuts.size()) _cuts.resize(node_index + 1);

      return _cuts[node_index];
    }

    /*! \brief Returns the cut set of a node */
    cut_set_t const &cuts(uint32_t node_index) const {
      assert(node_index < _cuts.size());
      return _cuts[node_index];
    }

    /*! \brief Returns the truth table of a cut */
    template <bool enabled = ComputeTruth, typename = std::enable_if_t<std::is_same_v<Ntk, Ntk> && enabled>>
    auto truth_table(cut_t const &cut) const {
      return _truth_tables[cut->func_id];
    }

    /*! \brief Returns the total number of tuples that were tried to be merged */
    auto total_tuples() const { return _total_tuples; }

    /*! \brief Returns the total number of cuts in the database. */
    auto total_cuts() const { return _total_cuts; }

    /*! \brief Returns the number of nodes for which cuts are computed */
    auto nodes_size() const { return _cuts.size(); }

    /* compute positions of leave indices in cut `sub` (subset) with respect to
     * leaves in cut `sup` (super set).
     *
     * Example:
     *   compute_truth_table_support( {1, 3, 6}, {0, 1, 2, 3, 6, 7} ) = {1, 3, 4}
     */
    std::vector<uint8_t> compute_truth_table_support(cut_t const &sub, cut_t const &sup) const {
      std::vector<uint8_t> support;
      support.reserve(sub.size());

      auto itp = sup.begin();
      for (auto i : sub) {
        itp = std::find(itp, sup.end(), i);
        support.push_back(static_cast<uint8_t>(std::distance(sup.begin(), itp)));
      }

      return support;
    }

    /*! \brief Inserts a truth table into the truth table cache.
     *
     * This message can be used when manually adding or modifying cuts from the
     * cut sets.
     *
     * \param tt Truth table to add
     * \return Literal id from the truth table store
     */
    uint32_t insert_truth_table(kitty::static_truth_table<NumVars> const &tt) { return _truth_tables.insert(tt); }

  private:
    template <typename _Ntk, uint32_t _NumVars, bool _ComputeTruth, typename _CutData>
    friend class detail2::dynamic_cut_enumeration_impl;

  private:
    void add_zero_cut(uint32_t index) {
      auto &cut = _cuts[index].add_cut(&index, &index); /* fake iterator for emptyness */

      if constexpr (ComputeTruth) {
        cut->func_id = 0;
      }
    }

    void add_unit_cut(uint32_t index) {
      auto &cut = _cuts[index].add_cut(&index, &index + 1);

      if constexpr (ComputeTruth) {
        cut->func_id = 2;
      }
    }

    void clear_cut_set(uint32_t index) { _cuts[index].clear(); }

  private:
    /* compressed representation of cuts */
    std::deque<cut_set_t> _cuts;

    /* cut truth tables */
    truth_table_cache<kitty::static_truth_table<NumVars>> _truth_tables;

    /* statistics */
    uint32_t _total_tuples{};
    std::size_t _total_cuts{};
  };

  /*! \cond PRIVATE */
  namespace detail2 {
    template <typename Ntk, uint32_t NumVars, bool ComputeTruth, typename CutData>
    class dynamic_cut_enumeration_impl {
    public:
      using cut_t = typename dynamic_network_cuts2<Ntk, NumVars, ComputeTruth, CutData>::cut_t;
      using cut_set_t = typename dynamic_network_cuts2<Ntk, NumVars, ComputeTruth, CutData>::cut_set_t;

      explicit dynamic_cut_enumeration_impl(Ntk const &ntk, cut_enumeration_params const &ps, cut_enumeration_stats &st, dynamic_network_cuts2<Ntk, NumVars, ComputeTruth, CutData> &cuts) : ntk(ntk), ps(ps), st(st), cuts(cuts) {
        assert(ps.cut_limit < cuts.max_cut_num && "cut_limit exceeds the compile-time limit for the maximum number of cuts");
      }

    public:
      void run() {
        stopwatch t(st.time_total);

        ntk.foreach_node([this](auto node) {
          const auto index = ntk.node_to_index(node);

          if (ps.very_verbose) {
            std::cout << fmt::format("[i] compute cut for node at index {}\n", index);
          }

          if (ntk.is_dead(node)) { // no cut for a dead node
            cuts.add_zero_cut(index);
          } else if (ntk.is_constant(node)) {
            cuts.add_zero_cut(index);
          } else if (ntk.is_ci(node)) {
            cuts.add_unit_cut(index);
          } else {
            if constexpr (Ntk::min_fanin_size == 2 && Ntk::max_fanin_size == 2) {
              merge_cuts2(index);
            } else {
              merge_cuts(index);
            }
          }
        });
      }

      void compute_cuts(node<Ntk> const &n) {
        const auto index = ntk.node_to_index(n);

        if (ntk.is_dead(n)) { // stop at a dead node
          cuts.clear_cut_set(index);
          return;
        }

        if (cuts.cuts(index).size() > 0) return;

        ntk.foreach_fanin(n, [&](auto const &f) { compute_cuts(ntk.get_node(f)); });

        if constexpr (Ntk::min_fanin_size == 2 && Ntk::max_fanin_size == 2) {
          merge_cuts2(index);
        } else {
          merge_cuts(index);
        }
      }

      void init_cuts() {
        cuts.add_zero_cut(ntk.node_to_index(ntk.get_node(ntk.get_constant(false))));
        if (ntk.get_node(ntk.get_constant(false)) != ntk.get_node(ntk.get_constant(true))) cuts.add_zero_cut(ntk.node_to_index(ntk.get_node(ntk.get_constant(true))));
        ntk.foreach_ci([&](auto const &n) { cuts.add_unit_cut(ntk.node_to_index(n)); });
      }

      void clear_cuts(node<Ntk> const &n) {
        const auto index = ntk.node_to_index(n);
        if (cuts.cuts(index).size() == 0) return;

        cuts.clear_cut_set(index);
      }

    private:
      inline bool fast_support_minimization(kitty::static_truth_table<NumVars> const &tt, cut_t &res) {
        uint32_t support = 0u;
        uint32_t support_size = 0u;
        for (uint32_t i = 0u; i < tt.num_vars(); ++i) {
          if (kitty::has_var(tt, i)) {
            support |= 1u << i;
            ++support_size;
          }
        }

        /* has not minimized support? */
        if ((support & (support + 1u)) != 0u) {
          return false;
        }

        /* variables not in the support are the most significative */
        if (support_size != res.size()) {
          std::vector<uint32_t> leaves(res.begin(), res.begin() + support_size);
          res.set_leaves(leaves.begin(), leaves.end());
        }

        return true;
      }

      uint32_t compute_truth_table(uint32_t index, std::vector<cut_t const *> const &vcuts, cut_t &res) {
        stopwatch t(st.time_truth_table);

        std::vector<kitty::static_truth_table<NumVars>> tt(vcuts.size());
        auto i = 0;
        for (auto const &cut : vcuts) {
          tt[i] = cuts._truth_tables[(*cut)->func_id];
          const auto supp = cuts.compute_truth_table_support(*cut, res);
          kitty::expand_inplace(tt[i], supp);
          ++i;
        }

        auto tt_res = ntk.compute(ntk.index_to_node(index), tt.begin(), tt.end());

        if (ps.minimize_truth_table && !fast_support_minimization(tt_res, res)) {
          const auto support = kitty::min_base_inplace(tt_res);
          if (support.size() != res.size()) {
            std::vector<uint32_t> leaves_before(res.begin(), res.end());
            std::vector<uint32_t> leaves_after(support.size());

            auto it_support = support.begin();
            auto it_leaves = leaves_after.begin();
            while (it_support != support.end()) {
              *it_leaves++ = leaves_before[*it_support++];
            }
            res.set_leaves(leaves_after.begin(), leaves_after.end());
          }
        }

        return cuts._truth_tables.insert(tt_res);
      }

      void merge_cuts2(uint32_t index) {
        const auto fanin = 2;

        uint32_t pairs{1};
        ntk.foreach_fanin(ntk.index_to_node(index), [this, &pairs](auto child, auto i) {
          lcuts[i] = &cuts.cuts(ntk.node_to_index(ntk.get_node(child)));
          pairs *= static_cast<uint32_t>(lcuts[i]->size());
        });
        lcuts[2] = &cuts.cuts(index);
        auto &rcuts = *lcuts[fanin];
        rcuts.clear();

        cut_t new_cut;

        std::vector<cut_t const *> vcuts(fanin);

        cuts._total_tuples += pairs;
        for (auto const &c1 : *lcuts[0]) {
          for (auto const &c2 : *lcuts[1]) {
            if (!c1->merge(*c2, new_cut, NumVars)) {
              continue;
            }

            if (rcuts.is_dominated(new_cut)) {
              continue;
            }

            if constexpr (ComputeTruth) {
              vcuts[0] = c1;
              vcuts[1] = c2;
              new_cut->func_id = compute_truth_table(index, vcuts, new_cut);
            }

            cut_enumeration_update_cut<CutData>::apply(new_cut, cuts, ntk, index);

            rcuts.insert(new_cut);
          }
        }

        /* limit the maximum number of cuts */
        rcuts.limit(ps.cut_limit);

        cuts._total_cuts += rcuts.size();

        if (rcuts.size() > 1 || (*rcuts.begin())->size() > 1) {
          cuts.add_unit_cut(index);
        }
      }

      void merge_cuts(uint32_t index) {
        uint32_t pairs{1};
        std::vector<uint32_t> cut_sizes;
        ntk.foreach_fanin(ntk.index_to_node(index), [this, &pairs, &cut_sizes](auto child, auto i) {
          lcuts[i] = &cuts.cuts(ntk.node_to_index(ntk.get_node(child)));
          cut_sizes.push_back(static_cast<uint32_t>(lcuts[i]->size()));
          pairs *= cut_sizes.back();
        });

        const auto fanin = cut_sizes.size();
        lcuts[fanin] = &cuts.cuts(index);

        auto &rcuts = *lcuts[fanin];

        if (fanin > 1 && fanin <= ps.fanin_limit) {
          rcuts.clear();

          cut_t new_cut, tmp_cut;

          std::vector<cut_t const *> vcuts(fanin);

          cuts._total_tuples += pairs;
          foreach_mixed_radix_tuple(cut_sizes.begin(), cut_sizes.end(), [&](auto begin, auto end) {
            auto it = vcuts.begin();
            auto i = 0u;
            while (begin != end) {
              *it++ = &((*lcuts[i++])[*begin++]);
            }

            if (!vcuts[0]->merge(*vcuts[1], new_cut, NumVars)) {
              return true; /* continue */
            }

            for (i = 2; i < fanin; ++i) {
              tmp_cut = new_cut;
              if (!vcuts[i]->merge(tmp_cut, new_cut, NumVars)) {
                return true; /* continue */
              }
            }

            if (rcuts.is_dominated(new_cut)) {
              return true; /* continue */
            }

            if constexpr (ComputeTruth) {
              new_cut->func_id = compute_truth_table(index, vcuts, new_cut);
            }

            cut_enumeration_update_cut<CutData>::apply(new_cut, cuts, ntk, ntk.index_to_node(index));

            rcuts.insert(new_cut);

            return true;
          });

          /* limit the maximum number of cuts */
          rcuts.limit(ps.cut_limit);
        } else if (fanin == 1) {
          rcuts.clear();

          for (auto const &cut : *lcuts[0]) {
            cut_t new_cut = *cut;

            if constexpr (ComputeTruth) {
              new_cut->func_id = compute_truth_table(index, {cut}, new_cut);
            }

            cut_enumeration_update_cut<CutData>::apply(new_cut, cuts, ntk, ntk.index_to_node(index));

            rcuts.insert(new_cut);
          }

          /* limit the maximum number of cuts */
          rcuts.limit(ps.cut_limit);
        }

        cuts._total_cuts += static_cast<uint32_t>(rcuts.size());

        cuts.add_unit_cut(index);
      }

    private:
      Ntk const &ntk;
      cut_enumeration_params const &ps;
      cut_enumeration_stats &st;
      dynamic_network_cuts2<Ntk, NumVars, ComputeTruth, CutData> &cuts;

      std::array<cut_set_t *, Ntk::max_fanin_size + 1> lcuts;
    };
  } // namespace detail2
  /*! \endcond */

} // namespace mockturtle

namespace mockturtle {

  /*! \brief Parameters for Rewrite.
   *
   * The data structure `rewrite_params` holds configurable parameters with
   * default arguments for `rewrite`.
   */
  struct rewrite_params {
    rewrite_params() {
      /* 0 < Cut limit < 16 */
      cut_enumeration_ps.cut_limit = 8;
      cut_enumeration_ps.minimize_truth_table = true;
    }

    /*! \brief Cut enumeration parameters. */
    cut_enumeration_params cut_enumeration_ps{};

    /*! \brief If true, candidates are only accepted if they do not increase logic depth. */
    bool preserve_depth{false};

    /*! \brief Allow rewrite with multiple structures */
    bool allow_multiple_structures{true};

    /*! \brief Allow zero-gain substitutions */
    bool allow_zero_gain{false};

    /*! \brief Use satisfiability don't cares for optimization. */
    bool use_dont_cares{false};

    /*! \brief Use egg for optimization. */
    bool use_egg{true};

    /*! \brief Window size for don't cares calculation. */
    uint32_t window_size{8u};

    /*! \brief Be verbose. */
    bool verbose{false};
  };

  /*! \brief Statistics for rewrite.
   *
   * The data structure `rewrite_stats` provides data collected by running
   * `rewrite`.
   */
  struct rewrite_stats {
    /*! \brief Total runtime. */
    stopwatch<>::duration time_total{0};

    /*! \brief Expected gain. */
    uint32_t estimated_gain{0};

    /*! \brief Candidates */
    uint32_t candidates{0};

    void report() const { std::cout << fmt::format("[i] total time       = {:>5.2f} secs\n", to_seconds(time_total)); }
  };

  namespace detail {

    template <class Ntk, class Library, uint32_t NumVars, class NodeCostFn>
    class rewrite_impl {
      static constexpr uint32_t num_vars = 4u;
      static constexpr uint32_t max_window_size = 8u;
      using network_cuts_t = dynamic_network_cuts2<Ntk, NumVars, false, cut_enumeration_rewrite_cut>;
      using cut_manager_t = detail2::dynamic_cut_enumeration_impl<Ntk, NumVars, false, cut_enumeration_rewrite_cut>;
      using cut_t = typename network_cuts_t::cut_t;
      using node_data = typename Ntk::storage::element_type::node_type;

    public:
      rewrite_impl(Ntk &ntk, Library &library, rewrite_params const &ps, rewrite_stats &st, NodeCostFn const &cost_fn) : ntk(ntk), library(library), ps(ps), st(st), cost_fn(cost_fn), required(ntk, UINT32_MAX) { register_events(); }

      ~rewrite_impl() {
        if constexpr (has_level_v<Ntk>) {
          ntk.events().release_add_event(add_event);
          ntk.events().release_modified_event(modified_event);
          ntk.events().release_delete_event(delete_event);
        }
      }

      void run() {
        stopwatch t(st.time_total);

        ntk.incr_trav_id();

        if (ps.preserve_depth) {
          compute_required();
        }

        if (ps.use_egg) perform_rewriting_egg();

        st.estimated_gain = _estimated_gain;
        st.candidates = _candidates;
      }

    private:
      void perform_rewriting_egg() {
        const auto size = ntk.size();

        /* initialize the cut manager */
        cut_enumeration_stats cst;
        network_cuts_t cuts(size + (size >> 1));
        cut_manager_t cut_manager(ntk, ps.cut_enumeration_ps, cst, cuts);

        /* initialize cuts for constant nodes and PIs */
        cut_manager.init_cuts();

        std::array<mockturtle::signal<Ntk>, num_vars> best_leaves4;
        std::array<mockturtle::signal<Ntk>, num_vars> leaves4;
        mockturtle::signal<Ntk> best_signal;
        std::vector<node<Ntk>> best_leaves;
        std::vector<uint32_t> leaf_levels;
        std::vector<node<Ntk>> leaves;
        best_leaves.reserve(NumVars);
        leaf_levels.reserve(NumVars);
        leaves.reserve(NumVars);

        ntk.foreach_gate([&](auto const &n, auto i) {
          if (ntk.fanout_size(n) == 0u || ntk.is_dead(n)) return;

          int32_t best_gain = -1;
          uint32_t best_level = UINT32_MAX;
          bool best_phase = false;

          std::string best_expr_aft;
          bool is_on_critical_path = false;

          /* update level for node */
          if constexpr (has_level_v<Ntk>) {
            uint32_t level = 0;
            ntk.foreach_fanin(n, [&](auto const &f) { level = std::max(level, ntk.level(ntk.get_node(f))); });
            ntk.set_level(n, level + 1);
            best_level = level + 1;
            is_on_critical_path = ntk.is_on_critical_path(n);
          }

          cut_manager.clear_cuts(n);
          cut_manager.compute_cuts(n);

          /*
          if (n == 209) {
            bool res = experiments::abc_cec_impl(ntk, "/home/flynn/workplace/MIGBalance/tools/mockturtle/experiments/benchmarks/div.aig");
            printf("%lu\n", n);
          }
          */
          const uint32_t original_level = best_level;
          for (auto &cut : cuts.cuts(ntk.node_to_index(n))) {
            /* skip trivial cut */
            const size_t cur_cut_size = cut->size();
            if ((cur_cut_size == 1 && *cut->begin() == ntk.node_to_index(n))) {
              continue;
            }

            leaves.clear();
            leaf_levels.clear();
            bool flag = 0;
            for (auto const leaf_index : *cut) {
              node<Ntk> cur_leaf = ntk.index_to_node(leaf_index);
              if (ntk.is_dead(cur_leaf)) {
                flag = 1;
                break;
              }
              leaves.push_back(cur_leaf);
              if constexpr (has_level_v<Ntk>) {
                uint32_t cl = ntk.level(cur_leaf);
                if (cl > original_level) {
                  flag = 1;
                  break;
                }
                leaf_levels.push_back(cl);
              }
            }

            // useless cut
            if (flag) continue;

            /* select a cut*/
            if (cur_cut_size > num_vars) {
              if (select_cut_by_egg(n, cut, is_on_critical_path, leaves, leaf_levels, best_expr_aft, best_gain, best_level)) {
                best_leaves = leaves;
              }
            } else {
              cut_view<Ntk> cut_mig{ntk, leaves, ntk.make_signal(n)};
              if (select_cut_by_exact(n, cut, is_on_critical_path, leaves4, cut_mig, best_signal, best_gain, best_level, best_phase)) {
                best_leaves4 = leaves4;
                best_leaves.clear();
              }
            }

            if (cut->size() == 0 || (cut->size() == 1 && *cut->begin() != ntk.node_to_index(n))) break;
          }

          if (best_gain > 0 || (best_gain == 0 && best_level < original_level)) {
            if (!best_leaves.empty()) {
              // build the optimal sub-graph
              best_signal = egg_view<Ntk>::rebuild(ntk, best_expr_aft.data(), best_expr_aft.size(), best_leaves);
              // replace node wth the new structure
              ntk.substitute_node_no_restrash(n, best_signal);
            } else {
              // replace node wth the new structure
              auto &db = library.get_database();
              mockturtle::topo_view topo{db, best_signal};
              best_signal = cleanup_dangling(topo, ntk, best_leaves4.begin(), best_leaves4.end()).front();
              ntk.substitute_node_no_restrash(n, best_signal ^ best_phase);
            }

            clear_cuts_fanout_rec(cuts, cut_manager, ntk.get_node(best_signal));
          }
        });
      }

      bool select_cut_by_egg(mockturtle::node<Ntk> const &n, cut_t const *cut, bool const &is_on_critical_path, std::vector<node<Ntk>> const &leaves, std::vector<uint32_t> const &leaf_levels, std::string &best_expr_aft, int32_t &best_gain = -1,
                             uint32_t &best_level = UINT32_MAX) {

        /* measure the MFFC contained in the cut */
        const uint32_t mffc_size = measure_mffc_deref(n, cut);
        /* restore contained MFFC */
        measure_mffc_ref(n, cut);
        if (mffc_size <= 1) return false;

        // build egg graph
        egg_view<Ntk> eview(ntk, leaves, ntk.make_signal(n));
        // const uint32_t mffc_size2 = eview._mffc_size;
        // assert(mffc_size == mffc_size2 && "mffc_size == mffc_size2");

        // skip bad cut
        if (mffc_size <= 1 || eview._original_size < 2 || eview.has_bug) return false;

        // optimize by egg
        const CCost *dcost = eview.optimize_by_egg_lib(leaf_levels);
        if (!dcost) return false;
        uint32_t aft_dep = dcost->aft_dep;
        bool find_better = false;
        for (const auto &expr : dcost->aft_expr) {
          const std::string aft_expr(expr);

          // skip bad cut
          if (eview._original_expr == aft_expr) continue;

          // rewrite using egg
          const uint32_t size_bef = ntk.size();
          const signal<Ntk> new_f = egg_view<Ntk>::rebuild(ntk, aft_expr.data(), aft_expr.size(), leaves);
          const uint32_t size_aft = ntk.size();
          const int32_t nodes_added = size_aft - size_bef;
          const int32_t gain = mffc_size - nodes_added;
          if constexpr (has_level_v<Ntk>) {
            aft_dep = ntk.level(ntk.get_node(new_f));
          }

          // discard if dag.root and n are the same
          if (n == ntk.get_node(new_f)) {
            assert(nodes_added == 0);
            continue;
          }

          // discard if no gain
          if (gain < 0) {
            // ntk.take_out_node(ntk.get_node(new_f));
            set_news_dead(ntk, size_bef, size_aft);
            continue;
          }

          if ((is_on_critical_path && gain > best_gain && aft_dep <= best_level) || (!is_on_critical_path && gain > best_gain) || (gain == best_gain && aft_dep < best_level)) {
            // if ((gain > best_gain && aft_dep <= best_level) || (gain == best_gain && aft_dep < best_level)) {
            // if ((gain > best_gain) || (gain == best_gain && aft_dep < best_level)) {
            best_gain = gain;
            best_level = aft_dep;
            best_expr_aft = aft_expr;
            find_better = true;
          } else {
            // ntk.take_out_node(ntk.get_node(new_f));
          }
          set_news_dead(ntk, size_bef, size_aft);
          if (!ps.allow_multiple_structures) break;
        }
        return find_better;
      }

      bool select_cut_by_exact(mockturtle::node<Ntk> const &n, cut_t const *cut, bool const &is_on_critical_path, std::array<mockturtle::signal<Ntk>, num_vars> &leaves, mockturtle::cut_view<Ntk> const &cut_mig, mockturtle::signal<Ntk> &best_signal, int32_t &best_gain = -1,
                               uint32_t &best_level = UINT32_MAX, bool &best_phase = false) {
        auto &db = library.get_database();

        std::array<uint8_t, num_vars> permutation;

        /* Boolean matching */
        auto tt = mockturtle::simulate<kitty::static_truth_table<num_vars>>(cut_mig)[0];
        auto config = kitty::exact_npn_canonization(tt);
        auto tt_npn = std::get<0>(config);
        auto neg = std::get<1>(config);
        auto perm = std::get<2>(config);

        auto const structures = library.get_supergates(tt_npn);

        if (structures == nullptr) {
          return false;
        }

        uint32_t negation = 0;
        for (auto j = 0u; j < num_vars; ++j) {
          permutation[perm[j]] = j;
          negation |= ((neg >> perm[j]) & 1) << j;
        }

        /* save output negation to apply */
        bool phase = (neg >> num_vars == 1) ? true : false;

        {
          auto j = 0u;
          for (auto const leaf : *cut) {
            leaves[permutation[j++]] = ntk.make_signal(ntk.index_to_node(leaf));
          }

          while (j < num_vars)
            leaves[permutation[j++]] = ntk.get_constant(false);
        }

        for (auto j = 0u; j < num_vars; ++j) {
          if ((negation >> j) & 1) {
            leaves[j] = !leaves[j];
          }
        }

        bool find_better = false;
        {
          /* measure the MFFC contained in the cut */
          int32_t mffc_size = measure_mffc_deref(n, cut);

          for (auto const &dag : *structures) {
            auto [nodes_added, level] = evaluate_entry(n, db.get_node(dag.root), leaves);
            int32_t gain = mffc_size - nodes_added;

            /* discard if dag.root and n are the same */
            if (ntk.node_to_index(n) == db.value(db.get_node(dag.root)) >> 1) continue;

            /* discard if no gain */
            if (gain < 0) continue;

            /* discard if level increases */
            if constexpr (mockturtle::has_level_v<Ntk>) {
              if (ps.preserve_depth && level > required[n]) continue;
            }

            if ((is_on_critical_path && gain > best_gain && level <= best_level) || (!is_on_critical_path && gain > best_gain) || (gain == best_gain && level < best_level)) {
              // if ((gain > best_gain) || (gain == best_gain && level < best_level)) {
              ++_candidates;
              best_gain = gain;
              best_signal = dag.root;
              best_phase = phase;
              best_level = level;
              find_better = true;
            }

            if (!ps.allow_multiple_structures) break;
          }

          /* restore contained MFFC */
          measure_mffc_ref(n, cut);
        }
        return find_better;
      }

      uint32_t measure_mffc_ref(mockturtle::node<Ntk> const &n, cut_t const *cut) {
        /* reference cut leaves */
        for (auto leaf : *cut) {
          ntk.incr_fanout_size(ntk.index_to_node(leaf));
        }

        uint32_t mffc_size = recursive_ref(n);

        /* dereference leaves */
        for (auto leaf : *cut) {
          ntk.decr_fanout_size(ntk.index_to_node(leaf));
        }

        return mffc_size;
      }

      uint32_t measure_mffc_deref(mockturtle::node<Ntk> const &n, cut_t const *cut) {
        /* reference cut leaves */
        for (auto leaf : *cut) {
          ntk.incr_fanout_size(ntk.index_to_node(leaf));
        }

        uint32_t mffc_size = recursive_deref(n);

        /* dereference leaves */
        for (auto leaf : *cut) {
          ntk.decr_fanout_size(ntk.index_to_node(leaf));
        }

        return mffc_size;
      }

      uint32_t recursive_deref(mockturtle::node<Ntk> const &n) {
        /* terminate? */
        if (ntk.is_constant(n) || ntk.is_pi(n)) return 0;

        /* recursively collect nodes */
        uint32_t value{cost_fn(ntk, n)};
        ntk.foreach_fanin(n, [&](auto const &s) {
          if (ntk.decr_fanout_size(ntk.get_node(s)) == 0) {
            value += recursive_deref(ntk.get_node(s));
          }
        });
        return value;
      }

      uint32_t recursive_ref(mockturtle::node<Ntk> const &n) {
        /* terminate? */
        if (ntk.is_constant(n) || ntk.is_pi(n)) return 0;

        /* recursively collect nodes */
        uint32_t value{cost_fn(ntk, n)};
        ntk.foreach_fanin(n, [&](auto const &s) {
          if (ntk.incr_fanout_size(ntk.get_node(s)) == 0) {
            value += recursive_ref(ntk.get_node(s));
          }
        });
        return value;
      }

      void set_news_dead(Ntk &ntk, uint32_t size_bef, uint32_t size_aft) {
        for (uint32_t i = size_bef; i < size_aft; ++i) {
          set_dead(ntk, ntk.index_to_node(i));
        }
        ntk._storage->nodes.resize(size_bef);
      }

      void set_dead(Ntk &ntk, node<Ntk> const &n) {
        /* we cannot delete CIs, constants, or already dead nodes */
        if (n == 0 || ntk.is_ci(n) || ntk.is_dead(n)) return;

        auto &nobj = ntk._storage->nodes[n];
        nobj.data[0].h1 = UINT32_C(0x80000000); /* fanout size 0, but dead */
        ntk._storage->hash.erase(nobj);

        for (auto const &fn : ntk._events->on_delete) {
          (*fn)(n);
        }

        for (auto i = 0u; i < Ntk::max_fanin_size; ++i) {
          if (ntk.fanout_size(nobj.children[i].index) == 0) {
            continue;
          }
          ntk.decr_fanout_size(nobj.children[i].index);
        }
      }

      inline std::pair<int32_t, uint32_t> evaluate_entry(mockturtle::node<Ntk> const &current_root, mockturtle::node<Ntk> const &n, std::array<mockturtle::signal<Ntk>, num_vars> const &leaves) {
        auto &db = library.get_database();
        db.incr_trav_id();

        return evaluate_entry_rec(current_root, n, leaves);
      }

      std::pair<int32_t, uint32_t> evaluate_entry_rec(mockturtle::node<Ntk> const &current_root, mockturtle::node<Ntk> const &n, std::array<mockturtle::signal<Ntk>, num_vars> const &leaves) {
        auto &db = library.get_database();
        if (db.is_pi(n) || db.is_constant(n)) return {0, 0};
        if (db.visited(n) == db.trav_id()) return {0, 0};

        db.set_visited(n, db.trav_id());

        int32_t area = 0;
        uint32_t level = 0;
        bool hashed = true;

        std::array<mockturtle::signal<Ntk>, Ntk::max_fanin_size> node_data;
        db.foreach_fanin(n, [&](auto const &f, auto i) {
          mockturtle::node<Ntk> g = db.get_node(f);
          if (db.is_constant(g)) {
            node_data[i] = f; /* ntk.get_costant( db.is_complemented( f ) ) */
            return;
          }
          if (db.is_pi(g)) {
            node_data[i] = leaves[db.node_to_index(g) - 1] ^ db.is_complemented(f);
            if constexpr (mockturtle::has_level_v<Ntk>) {
              level = std::max(level, ntk.level(ntk.get_node(leaves[db.node_to_index(g) - 1])));
            }
            return;
          }

          auto [area_rec, level_rec] = evaluate_entry_rec(current_root, g, leaves);
          area += area_rec;
          level = std::max(level, level_rec);

          /* check value */
          if (db.value(g) < UINT32_MAX) {
            mockturtle::signal<Ntk> s;
            s.data = static_cast<uint64_t>(db.value(g));
            node_data[i] = s ^ db.is_complemented(f);
          } else {
            hashed = false;
          }
        });

        if (hashed) {
          /* try hash */
          /* AIG, XAG, MIG, and XMG are supported now */
          std::optional<mockturtle::signal<Ntk>> val;
          do {
            /* XAG */
            if constexpr (mockturtle::has_has_and_v<Ntk> && mockturtle::has_has_xor_v<Ntk>) {
              if (db.is_and(n))
                val = ntk.has_and(node_data[0], node_data[1]);
              else
                val = ntk.has_xor(node_data[0], node_data[1]);
              break;
            }

            /* AIG */
            if constexpr (mockturtle::has_has_and_v<Ntk>) {
              val = ntk.has_and(node_data[0], node_data[1]);
              break;
            }

            /* XMG */
            if constexpr (mockturtle::has_has_maj_v<Ntk> && mockturtle::has_has_xor3_v<Ntk>) {
              if (db.is_maj(n))
                val = ntk.has_maj(node_data[0], node_data[1], node_data[2]);
              else
                val = ntk.has_xor3(node_data[0], node_data[1], node_data[2]);
              break;
            }

            /* MAJ */
            if constexpr (mockturtle::has_has_maj_v<Ntk>) {
              val = ntk.has_maj(node_data[0], node_data[1], node_data[2]);
              break;
            }
            std::cerr << "[e] Only AIGs, XAGs, MAJs, and XMGs are currently supported \n";
          } while (false);

          if (val.has_value()) {
            /* bad condition (current root is contained in the DAG): return a very high cost */
            if (db.get_node(*val) == current_root) return {UINT32_MAX / 2, level + 1};

            /* annotate hashing info */
            db.set_value(n, val->data);
            return {area + (ntk.fanout_size(ntk.get_node(*val)) > 0 ? 0 : cost_fn(ntk, n)), level + 1};
          }
        }

        db.set_value(n, UINT32_MAX);
        return {area + cost_fn(ntk, n), level + 1};
      }

      void compute_required() {
        if constexpr (has_level_v<Ntk>) {
          ntk.foreach_po([&](auto const &f) { required[f] = ntk.depth(); });

          for (uint32_t index = ntk.size() - 1; index > ntk.num_pis(); index--) {
            node<Ntk> n = ntk.index_to_node(index);
            uint32_t req = required[n];

            ntk.foreach_fanin(n, [&](auto const &f) { required[f] = std::min(required[f], req - 1); });
          }
        }
      }

      void clear_cuts_fanout_rec(network_cuts_t &cuts, cut_manager_t &cut_manager, node<Ntk> const &n) {
        ntk.foreach_fanout(n, [&](auto const &g) {
          auto const index = ntk.node_to_index(g);
          if (cuts.cuts(index).size() > 0) {
            cut_manager.clear_cuts(g);
            if (!ntk.is_dead(g)) // stop at a dead node
              clear_cuts_fanout_rec(cuts, cut_manager, g);
          }
        });
      }

    private:
      void register_events() {
        if constexpr (has_level_v<Ntk>) {
          auto const update_level_of_new_node = [&](const auto &n) {
            ntk.resize_levels();
            update_node_level(n);
          };

          auto const update_level_of_existing_node = [&](node<Ntk> const &n, const auto &old_children) {
            (void)old_children;
            ntk.resize_levels();
            update_node_level(n);
          };

          auto const update_level_of_deleted_node = [&](node<Ntk> const &n) { ntk.set_level(n, -1); };

          add_event = ntk.events().register_add_event(update_level_of_new_node);
          modified_event = ntk.events().register_modified_event(update_level_of_existing_node);
          delete_event = ntk.events().register_delete_event(update_level_of_deleted_node);
        }
      }

      /* maybe it should be moved to depth_view */
      void update_node_level(node<Ntk> const &n, bool top_most = true) {
        if constexpr (has_level_v<Ntk>) {
          uint32_t curr_level = ntk.level(n);

          uint32_t max_level = 0;
          ntk.foreach_fanin(n, [&](const auto &f) {
            auto const p = ntk.get_node(f);
            auto const fanin_level = ntk.level(p);
            if (fanin_level > max_level) {
              max_level = fanin_level;
            }
          });
          ++max_level;

          if (curr_level != max_level) {
            ntk.set_level(n, max_level);

            /* update only one more level */
            if (top_most) {
              ntk.foreach_fanout(n, [&](const auto &p) { update_node_level(p, false); });
            }
          }
        }
      }

    private:
      Ntk &ntk;
      Library &library;
      rewrite_params const &ps;
      rewrite_stats &st;
      NodeCostFn cost_fn;

      node_map<uint32_t, Ntk> required;

      uint32_t _candidates{0};
      uint32_t _estimated_gain{0};

      /* events */
      std::shared_ptr<typename network_events<Ntk>::add_event_type> add_event;
      std::shared_ptr<typename network_events<Ntk>::modified_event_type> modified_event;
      std::shared_ptr<typename network_events<Ntk>::delete_event_type> delete_event;
    };

  } /* namespace detail */

  /*! \brief Boolean rewrite.
   *
   * This algorithm rewrites enumerated cuts using new network structures from a database.
   * The algorithm performs changes in-place and keeps the substituted structures dangling
   * in the network.
   *
   * **Required network functions:**
   * - `get_node`
   * - `size`
   * - `make_signal`
   * - `foreach_gate`
   * - `substitute_node`
   * - `clear_visited`
   * - `clear_values`
   * - `fanout_size`
   * - `set_value`
   * - `foreach_node`
   *
   * \param ntk Input network (will be changed in-place)
   * \param library Exact library containing pre-computed structures
   * \param ps Rewrite params
   * \param pst Rewrite statistics
   * \param cost_fn Node cost function (a functor with signature `uint32_t(Ntk const&, node<Ntk> const&)`)
   */
  template <class Ntk, class Library, uint32_t NumVars = 4u, class NodeCostFn = unit_cost<Ntk>>
  void rewrite(Ntk &ntk, Library &library, rewrite_params const &ps = {}, rewrite_stats *pst = nullptr, NodeCostFn const &cost_fn = {}) {
    static_assert(is_network_type_v<Ntk>, "Ntk is not a network type");
    static_assert(has_get_node_v<Ntk>, "Ntk does not implement the get_node method");
    static_assert(has_size_v<Ntk>, "Ntk does not implement the size method");
    static_assert(has_make_signal_v<Ntk>, "Ntk does not implement the make_signal method");
    static_assert(has_foreach_gate_v<Ntk>, "Ntk does not implement the foreach_gate method");
    static_assert(has_substitute_node_v<Ntk>, "Ntk does not implement the substitute_node method");
    static_assert(has_clear_visited_v<Ntk>, "Ntk does not implement the clear_visited method");
    static_assert(has_clear_values_v<Ntk>, "Ntk does not implement the clear_values method");
    static_assert(has_fanout_size_v<Ntk>, "Ntk does not implement the fanout_size method");
    static_assert(has_set_value_v<Ntk>, "Ntk does not implement the set_value method");
    static_assert(has_foreach_node_v<Ntk>, "Ntk does not implement the foreach_node method");

    rewrite_stats st;

    if (ps.preserve_depth || ps.use_dont_cares || ps.use_egg) {
      using depth_view_t = depth_view<Ntk, NodeCostFn>;
      depth_view_t depth_ntk{ntk};
      using fanout_view_t = fanout_view<depth_view_t>;
      fanout_view_t fanout_view{depth_ntk};

      detail::rewrite_impl<fanout_view_t, Library, NumVars, NodeCostFn> p(fanout_view, library, ps, st, cost_fn);
      p.run();
    } else {
      using fanout_view_t = fanout_view<Ntk>;
      fanout_view_t fanout_view{ntk};

      detail::rewrite_impl<fanout_view_t, Library, NumVars, NodeCostFn> p(fanout_view, library, ps, st, cost_fn);
      p.run();
    }

    if (ps.verbose) {
      st.report();
    }

    if (pst) {
      *pst = st;
    }

    ntk = cleanup_dangling(ntk);
  }

} /* namespace mockturtle */