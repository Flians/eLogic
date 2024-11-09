#pragma once

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_enumeration.hpp>
#include <mockturtle/algorithms/cut_enumeration/rewrite_cut.hpp>
#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/algorithms/simulation.hpp>
#include <mockturtle/traits.hpp>
#include <mockturtle/utils/cost_functions.hpp>
#include <mockturtle/utils/node_map.hpp>
#include <mockturtle/utils/stopwatch.hpp>
#include <mockturtle/views/color_view.hpp>
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

    void report() const {
      std::cout << fmt::format("[i] total time       = {:>5.2f} secs\n", to_seconds(time_total));
    }
  };

  namespace detail {

    template <class Ntk, uint32_t NumVars, class NodeCostFn>
    class rewrite_impl {
      using network_cuts_t = dynamic_network_cuts<Ntk, NumVars, true, cut_enumeration_rewrite_cut>;
      using cut_manager_t = detail::dynamic_cut_enumeration_impl<Ntk, NumVars, true, cut_enumeration_rewrite_cut>;
      using cut_t = typename network_cuts_t::cut_t;
      using node_data = typename Ntk::storage::element_type::node_type;

    public:
      rewrite_impl(Ntk &ntk, rewrite_params const &ps, rewrite_stats &st, NodeCostFn const &cost_fn)
          : ntk(ntk), ps(ps), st(st), cost_fn(cost_fn), required(ntk, UINT32_MAX) {
        register_events();
      }

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

        if (ps.use_egg)
          perform_rewriting_egg();

        st.estimated_gain = _estimated_gain;
        st.candidates = _candidates;
      }

    private:
      void perform_rewriting_egg() {
        /* initialize the cut manager */
        cut_enumeration_stats cst;
        network_cuts_t cuts(ntk.size() + (ntk.size() >> 1));
        cut_manager_t cut_manager(ntk, ps.cut_enumeration_ps, cst, cuts);

        /* initialize cuts for constant nodes and PIs */
        cut_manager.init_cuts();

        std::vector<uint32_t> leaf_levels;
        std::vector<node<Ntk>> leaves;
        leaf_levels.reserve(NumVars);
        leaves.reserve(NumVars);
        signal<Ntk> best_signal;

        const uint32_t min_depth_gap = 3;
        const uint32_t original_size = ntk.size();
        ntk.foreach_gate([&](auto const &n, auto i) {
          if (ntk.fanout_size(n) == 0u || ntk.is_dead(n))
            return;

          int32_t best_gain = -1;
          uint32_t best_dep = UINT32_MAX;

          /* update level for node */
          if constexpr (has_level_v<Ntk>) {
            uint32_t level = 0;
            ntk.foreach_fanin(n, [&](auto const &f) {
              level = std::max(level, ntk.level(ntk.get_node(f)));
            });
            ntk.set_level(n, level + 1);
            best_dep = level + 1;
          }

          cut_manager.clear_cuts(n);
          cut_manager.compute_cuts(n);

          /*
          if (n == 209) {
            bool res = experiments::abc_cec_impl(ntk, "/home/flynn/workplace/MIGBalance/tools/mockturtle/experiments/benchmarks/div.aig");
            printf("%lu\n", n);
          }
          */
          const uint32_t original_level = best_dep;
          for (auto &cut : cuts.cuts(ntk.node_to_index(n))) {
            /* skip trivial cut */
            const size_t cur_cut_size = cut->size();
            if ((cur_cut_size == 1 && *cut->begin() == ntk.node_to_index(n))) {
              continue;
            }

            leaves.clear();
            leaf_levels.clear();
            bool flag = 0;
            uint32_t min_fin_level = UINT32_MAX;
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
                min_fin_level = std::min(min_fin_level, cl);
              }
            }

            // useless cut
            if (flag)
              continue;

            /* measure the MFFC contained in the cut */
            const int32_t mffc_size = measure_mffc_deref(n, cut);
            // restore contained MFFC
            measure_mffc_ref(n, cut);

            if (mffc_size > 1) {
              // build egg graph
              egg_view<Ntk> eview(ntk, leaves, ntk.make_signal(n));

              // if (eview._mffc_size != mffc_size) printf("%lu\n", n);

              // skip bad cut
              if (eview._original_size < 2 || eview.has_bug)
                continue;

              // optimize by egg
              const auto dcost = eview.optimize_by_egg(leaf_levels);
              if (!dcost) {
                // free_ccost(dcost);
                continue;
              }
              const uint32_t aft_dep = dcost->aft_dep;
              std::string aft_expr;
              char *cur_chr = dcost->aft_expr;
              for (std::size_t i = 0; i < dcost->aft_expr_len; ++i) {
                aft_expr += *cur_chr++;
              }
              // free_ccost(dcost);

              // skip bad cut
              if (eview._original_expr == aft_expr || best_dep < aft_dep) {
                continue;
              }

              // rewrite using egg
              const uint32_t size_bef = ntk.size();
              const signal<Ntk> new_f = egg_view<Ntk>::rebuild(ntk, aft_expr.data(), aft_expr.size(), leaves);
              const int32_t nodes_added = ntk.size() - size_bef;
              const int32_t gain = mffc_size - nodes_added;

              // discard if dag.root and n are the same
              if (n == ntk.get_node(new_f)) {
                assert(nodes_added == 0);
                continue;
              }

              // discard if no gain
              if (gain < 0) {
                ntk.take_out_node(ntk.get_node(new_f));
                continue;
              }

              if (gain > best_gain || (gain == best_gain && aft_dep < best_dep)) {
                if (best_gain != -1)
                  ntk.take_out_node(ntk.get_node(best_signal));
                best_gain = gain;
                best_signal = new_f;
                best_dep = aft_dep;
              } else {
                ntk.take_out_node(ntk.get_node(new_f));
              }
            }

            if (cut->size() == 0 || (cut->size() == 1 && *cut->begin() != ntk.node_to_index(n)))
              break;
          }

          if (best_gain > 0 || (best_gain == 0 && best_dep < original_level)) {

            // replace node wth the new structure
            ntk.substitute_node_no_restrash(n, best_signal);

            clear_cuts_fanout_rec(cuts, cut_manager, ntk.get_node(best_signal));
          } else if (best_gain != -1) {
            ntk.take_out_node(ntk.get_node(best_signal));
          }
        });
      }

      int32_t measure_mffc_ref(node<Ntk> const &n, cut_t const *cut) {
        /* reference cut leaves */
        for (auto leaf : *cut) {
          ntk.incr_fanout_size(ntk.index_to_node(leaf));
        }

        int32_t mffc_size = static_cast<int32_t>(recursive_ref(n));

        /* dereference leaves */
        for (auto leaf : *cut) {
          ntk.decr_fanout_size(ntk.index_to_node(leaf));
        }

        return mffc_size;
      }

      int32_t measure_mffc_deref(node<Ntk> const &n, cut_t const *cut) {
        /* reference cut leaves */
        for (auto leaf : *cut) {
          ntk.incr_fanout_size(ntk.index_to_node(leaf));
        }

        int32_t mffc_size = static_cast<int32_t>(recursive_deref(n));

        /* dereference leaves */
        for (auto leaf : *cut) {
          ntk.decr_fanout_size(ntk.index_to_node(leaf));
        }

        return mffc_size;
      }

      uint32_t recursive_deref(node<Ntk> const &n) {
        /* terminate? */
        if (ntk.is_constant(n) || ntk.is_pi(n))
          return 0;

        /* recursively collect nodes */
        uint32_t value{cost_fn(ntk, n)};
        ntk.foreach_fanin(n, [&](auto const &s) {
          if (ntk.decr_fanout_size(ntk.get_node(s)) == 0) {
            value += recursive_deref(ntk.get_node(s));
          }
        });
        return value;
      }

      uint32_t recursive_ref(node<Ntk> const &n) {
        /* terminate? */
        if (ntk.is_constant(n) || ntk.is_pi(n))
          return 0;

        /* recursively collect nodes */
        uint32_t value{cost_fn(ntk, n)};
        ntk.foreach_fanin(n, [&](auto const &s) {
          if (ntk.incr_fanout_size(ntk.get_node(s)) == 0) {
            value += recursive_ref(ntk.get_node(s));
          }
        });
        return value;
      }

      void compute_required() {
        if constexpr (has_level_v<Ntk>) {
          ntk.foreach_po([&](auto const &f) {
            required[f] = ntk.depth();
          });

          for (uint32_t index = ntk.size() - 1; index > ntk.num_pis(); index--) {
            node<Ntk> n = ntk.index_to_node(index);
            uint32_t req = required[n];

            ntk.foreach_fanin(n, [&](auto const &f) {
              required[f] = std::min(required[f], req - 1);
            });
          }
        }
      }

      void propagate_required_rec(uint32_t root, node<Ntk> const &n, uint32_t size, uint32_t req) {
        if (ntk.is_constant(n) || ntk.is_pi(n))
          return;

        /* recursively update required time */
        ntk.foreach_fanin(n, [&](auto const &f) {
          auto const g = ntk.get_node(f);

          /* recur if it is still a node to explore and to update */
          if (ntk.node_to_index(g) > root && (ntk.node_to_index(g) >= size || required[g] > req))
            propagate_required_rec(root, g, size, req - 1);

          /* update the required time */
          if (ntk.node_to_index(g) < size)
            required[g] = std::min(required[g], req - 1);
        });
      }

      void clear_cuts_fanout_rec(network_cuts_t &cuts, cut_manager_t &cut_manager, node<Ntk> const &n) {
        ntk.foreach_fanout(n, [&](auto const &g) {
          auto const index = ntk.node_to_index(g);
          if (cuts.cuts(index).size() > 0) {
            cut_manager.clear_cuts(g);
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

          auto const update_level_of_deleted_node = [&](node<Ntk> const &n) {
            ntk.set_level(n, -1);
          };

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
              ntk.foreach_fanout(n, [&](const auto &p) {
                update_node_level(p, false);
              });
            }
          }
        }
      }

    private:
      Ntk &ntk;
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
  template <class Ntk, uint32_t NumVars = 4u, class NodeCostFn = unit_cost<Ntk>>
  void rewrite(Ntk &ntk, rewrite_params const &ps = {}, rewrite_stats *pst = nullptr, NodeCostFn const &cost_fn = {}) {
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

      detail::rewrite_impl<fanout_view_t, NumVars, NodeCostFn> p(fanout_view, ps, st, cost_fn);
      p.run();
    } else {
      using fanout_view_t = fanout_view<Ntk>;
      fanout_view_t fanout_view{ntk};

      detail::rewrite_impl<fanout_view_t, NumVars, NodeCostFn> p(fanout_view, ps, st, cost_fn);
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