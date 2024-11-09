/*!
  \file egg_view.hpp
  \brief Implements an isolated view for egg on a single cut in a network

  \author Rongliang Fu
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <parallel_hashmap/phmap.h>

#include <mockturtle/networks/detail/foreach.hpp>
#include <mockturtle/traits.hpp>
#include <mockturtle/views/immutable_view.hpp>

#include "mig_egg/src/lib.rs.h"
#include "rust/cxx.h"

namespace mockturtle {

  class StrCostTable {
  private:
    std::unordered_map<std::string, std::unique_ptr<CCost, decltype(&free_ccost)>> table;
    std::unordered_set<std::string> bad_exprs;

    bool is_bad(const std::string &key) const {
      return bad_exprs.find(key) != bad_exprs.end();
    }

  public:
    const CCost *insert(const std::string &key, const std::vector<uint32_t> &leaf_levels = {}) {
      if (is_bad(key))
        return nullptr;

      uint32_t min_level = 0, max_level = 0;
      if (!leaf_levels.empty()) {
        min_level = *std::min_element(leaf_levels.begin(), leaf_levels.end());
        max_level = *std::max_element(leaf_levels.begin(), leaf_levels.end());
      }
      std::string key_deps = key;
      if (max_level > min_level) {
        for (auto ll : leaf_levels) {
          key_deps += "_" + std::to_string(ll - min_level);
        }
      }

      size_t tn = table.size();
      if (tn > 100000 && tn % 100000 == 0)
        std::cout << ">>> " << tn << std::endl;

      auto [it, inserted] = table.try_emplace(key_deps, nullptr, free_ccost);

      if (inserted) {
        it->second = std::unique_ptr<CCost, decltype(&free_ccost)>(simplify_mig(key, leaf_levels.data(), leaf_levels.size()), free_ccost);
      }

      return it->second.get();
    }

    void set_bad(const std::string &key) {
      if (bad_exprs.insert(key).second)
        table.erase(key);
    }

    void reset(bool all = false) {
      bad_exprs.clear();
      if (all)
        table.clear();
    }

    const CCost *find(const std::string &key) const {
      auto it = table.find(key);
      return it != table.end() ? it->second.get() : nullptr;
    }
  };

  static StrCostTable exp_map;

  /*! \brief Implements an isolated view on a single cut in a network.
   *
   * This view can create a network from a single cut in a largest network.  This
   * cut has a single output `root` and set of `leaves`.  The view reimplements
   * the methods `size`, `num_pis`, `num_pos`, `foreach_pi`, `foreach_po`,
   * `foreach_node`, `foreach_gate`, `is_pi`, `node_to_index`, and
   * `index_to_node`.
   *
   * This view assumes that all nodes' visited flags are set 0 before creating
   * the view.  The view guarantees that all the nodes in the view will have a 0
   * visited flag after the construction.
   *
   * **Required network functions:**
   * - `set_visited`
   * - `visited`
   * - `get_node`
   * - `get_constant`
   * - `is_constant`
   * - `make_signal`
   */
  template <typename Ntk>
  class egg_view : public immutable_view<Ntk> {
  public:
    using storage = typename Ntk::storage;
    using node = typename Ntk::node;
    using signal = typename Ntk::signal;
    static constexpr bool is_topologically_sorted = true;

  public:
    explicit egg_view(Ntk const &ntk, std::vector<node> const &leaves, signal const &root)
        : immutable_view<Ntk>(ntk), _root(root) {
      construct(leaves);
    }

    template <typename _Ntk = Ntk, typename = std::enable_if_t<!std::is_same_v<typename _Ntk::signal, typename _Ntk::node>>>
    explicit egg_view(Ntk const &ntk, std::vector<signal> const &leaves, signal const &root)
        : immutable_view<Ntk>(ntk), _root(root) {
      construct(leaves);
    }

  private:
    template <typename LeaveType>
    void construct(std::vector<LeaveType> const &leaves) {
      static_assert(is_network_type_v<Ntk>, "Ntk is not a network type");
      static_assert(has_set_visited_v<Ntk>, "Ntk does not implement the set_visited method");
      static_assert(has_visited_v<Ntk>, "Ntk does not implement the visited method");
      static_assert(has_get_node_v<Ntk>, "Ntk does not implement the get_node method");
      static_assert(has_get_constant_v<Ntk>, "Ntk does not implement the get_constant method");
      static_assert(has_is_constant_v<Ntk>, "Ntk does not implement the is_constant method");
      static_assert(has_make_signal_v<Ntk>, "Ntk does not implement the make_signal method");
      static_assert(has_incr_trav_id_v<Ntk>, "Ntk does not implement the incr_trav_id method");
      static_assert(has_trav_id_v<Ntk>, "Ntk does not implement the trav_id method");
      static_assert(std::is_same_v<LeaveType, node> || std::is_same_v<LeaveType, signal>, "leaves must be vector of either node or signal");

      this->incr_trav_id();

      /* constants */
      add_constants();

      /* primary inputs */
      for (auto const &leaf : leaves) {
        if constexpr (std::is_same_v<LeaveType, node>) {
          add_leaf(leaf);
        } else {
          add_leaf(this->get_node(leaf));
        }
      }

      _original_depth = 0;
      _original_size = 0;
      _original_expr.clear();
      if (this->is_complemented(_root)) {
        _original_expr += "(~ ";
      }
      _original_depth = traverse(this->get_node(_root));
      _original_expr += get_egg_expr(this->get_node(_root));
      if (this->is_complemented(_root)) {
        _original_expr += ")";
      }
      uint32_t cur_size = _nodes.size();
      _original_size = cur_size - _num_leaves - _num_constants;
      _mffc_size = 1;
      cur_size -= 1; // except the root
      for (uint32_t ni = _num_constants + _num_leaves; ni < cur_size; ++ni) {
        if (this->fanout_size(_nodes[ni]) <= _out_degs[ni]) {
          ++_mffc_size;
        }
      }
    }

  public:
    inline auto num_pis() const { return _num_leaves; }
    inline auto num_pos() const { return 1; }
    inline auto num_gates() const { return _nodes.size() - _num_leaves - _num_constants; }

    inline auto node_to_index(const node &n) const { return _node_to_index.at(n); }
    inline auto index_to_node(uint32_t index) const { return _nodes[index]; }

    template <typename Fn>
    void foreach_po(Fn &&fn) const {
      std::vector<signal> roots{{_root}};
      detail::foreach_element(roots.begin(), roots.end(), fn);
    }

    inline bool is_pi(node const &pi) const {
      const auto beg = _nodes.begin() + _num_constants;
      return std::find(beg, beg + _num_leaves, pi) != beg + _num_leaves;
    }

    template <typename Fn>
    void foreach_pi(Fn &&fn) const {
      detail::foreach_element(_nodes.begin() + _num_constants, _nodes.begin() + _num_constants + _num_leaves, fn);
    }

    const std::string &get_egg_expr(node const &leaf) const {
      uint32_t index = node_to_index(leaf);
      return _prefix_exprs[index];
    }

    const CCost *optimize_by_egg_lib(const std::vector<uint32_t> &leaf_levels) const {
      return exp_map.insert(_original_expr, leaf_levels);
    }

    CCost *optimize_by_egg(const std::vector<uint32_t> &leaf_levels) const {
      return simplify_mig(_original_expr, leaf_levels.data(), leaf_levels.size());
    }

    void feedback(bool is_bad) const {
      if (is_bad)
        exp_map.set_bad(_original_expr);
    }

    static signal rebuild(Ntk &ntk, const char *aft_expr, const std::uint32_t aft_expr_len, const std::vector<node> &leaves) {
      const signal &const_false = ntk.get_constant(false);

      const signal gap(-1);
      std::stack<signal> signal_stack;
      for (auto start = aft_expr, end = aft_expr + aft_expr_len; start < end; ++start) {
        // skip space
        while (start < end && std::isspace(*start)) {
          ++start;
        }

        if (*start == '(') {
          signal_stack.push(gap);
        } else if (*start == ')') {
          // 1. collect children
          std::vector<signal> children;
          signal cid;
          while (!signal_stack.empty()) {
            cid = signal_stack.top();
            signal_stack.pop();
            if (cid == gap)
              break;
            children.push_back(cid);
          }
          // 2. build node
          std::size_t num_ins = children.size();
          signal new_node;
          if (num_ins == 1) { // NOT
            new_node = ntk.create_not(children[0]);
          } else if (num_ins == 2) { // AND
            new_node = ntk.create_and(children[0], children[1]);
          } else if (num_ins == 3) { // MAJ
            new_node = ntk.create_maj(children[0], children[1], children[2]);
          } else {
            assert(false);
          }
          signal_stack.push(new_node);
        } else {
          if (*start == '~' || *start == 'M' || *start == '&') {
            // pass
          } else {
            if (*start == '0') {
              signal_stack.push(const_false);
            } else if (*start == '1') {
              signal_stack.push(ntk.create_not(const_false));
            } else {
              signal_stack.push(ntk.make_signal(leaves[*start - 'a']));
            }
          }
        }
      }
      signal new_root = signal_stack.top();
      return new_root;
    }

  private:
    inline void add_constants() {
      add_node(this->get_node(this->get_constant(false)), "0");
      this->set_visited(this->get_node(this->get_constant(false)), this->trav_id());
      if (this->get_node(this->get_constant(true)) != this->get_node(this->get_constant(false))) {
        add_node(this->get_node(this->get_constant(true)), "1");
        this->set_visited(this->get_node(this->get_constant(true)), this->trav_id());
        ++_num_constants;
      }
    }

    inline void add_leaf(node const &leaf) {
      if (this->visited(leaf) == this->trav_id())
        return;

      add_node(leaf, std::string(1, 'a' + _num_leaves));
      this->set_visited(leaf, this->trav_id());
      ++_num_leaves;
    }

    inline void add_node(node const &n, const std::string &expr = "") {
      _node_to_index[n] = static_cast<uint32_t>(_nodes.size());
      _nodes.push_back(n);
      _prefix_exprs.push_back(expr);
      _out_degs.push_back(0);
    }

    size_t traverse(node const &n) {
      if (this->visited(n) == this->trav_id()) {
        ++_out_degs[node_to_index(n)];
        return 0;
      }

      // record current node's expression
      std::string cur_node_expr;

      /* AIG */
      if constexpr (has_has_and_v<Ntk>) {
        cur_node_expr += "(&";
      }

      /* MAJ */
      if constexpr (has_has_maj_v<Ntk>) {
        cur_node_expr += "(M";
      }

      size_t depth = 0;
      this->foreach_fanin(n, [&](const auto &f) {
        cur_node_expr += ' ';
        if (this->is_complemented(f)) {
          cur_node_expr += "(~ ";
        }
        size_t td = traverse(this->get_node(f));
        cur_node_expr += get_egg_expr(this->get_node(f));
        if (this->is_complemented(f)) {
          cur_node_expr += ")";
        }
        depth = std::max(depth, td);
      });

      cur_node_expr += ")";

      if (cur_node_expr.size() <= 3) {
        printf("egg_view.traverse meet dead nodes!\n");
        has_bug = 1;
      }

      add_node(n, cur_node_expr);
      ++_out_degs[node_to_index(n)];
      this->set_visited(n, this->trav_id());
      return depth + 1;
    }

  public:
    unsigned _num_constants{1};
    unsigned _num_leaves{0};
    std::vector<node> _nodes;
    phmap::flat_hash_map<node, uint32_t> _node_to_index;
    std::vector<std::string> _prefix_exprs;
    std::vector<uint32_t> _out_degs;
    unsigned _original_depth{0};
    unsigned _original_size{0};
    std::string _original_expr;
    uint32_t _mffc_size;
    signal _root;

    bool has_bug{0};
  };

  template <class T>
  egg_view(T const &, std::vector<node<T>> const &, signal<T> const &) -> egg_view<T>;

  template <class T, typename = std::enable_if_t<!std::is_same_v<typename T::signal, typename T::node>>>
  egg_view(T const &, std::vector<signal<T>> const &, signal<T> const &) -> egg_view<T>;

} /* namespace mockturtle */
