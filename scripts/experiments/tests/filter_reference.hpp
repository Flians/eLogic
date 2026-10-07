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

// Frozen quadratic reference, copied from mockturtle before the bucketing optimization.
// Upstream MIT license: tools/mockturtle/LICENSE.
struct Reference { static constexpr unsigned NInputs=9; struct { bool load_minimum_size_only=false; } _ps;
  bool compare_sizes( composed_gate<NInputs> const& s1, composed_gate<NInputs> const& s2 )
  {
    if ( s1.area < s2.area )
      return true;
    else if ( s1.area > s2.area )
      return false;

    /* compute average pin delay */
    float s1_delay = 0, s2_delay = 0;
    assert( s1.num_vars == s2.num_vars );
    for ( uint32_t i = 0; i < s1.num_vars; ++i )
    {
      s1_delay += s1.tdelay[i];
      s2_delay += s2.tdelay[i];
    }

    if ( s1_delay < s2_delay )
      return true;
    else if ( s1_delay > s2_delay )
      return false;
    else if ( s1.root->name < s2.root->name )
      return true;

    return false;
  }
  void filter_gates( std::deque<composed_gate<NInputs>> const& supergates, std::vector<bool>& skip_gates )
  {
    assert( supergates.size() >= skip_gates.size() );
    for ( uint32_t i = 0; i < skip_gates.size() - 1; ++i )
    {
      if ( supergates[i].root == nullptr )
        continue;

      if ( skip_gates[i] )
        continue;

      auto const& tti = supergates[i].function;
      for ( uint32_t j = i + 1; j < skip_gates.size(); ++j )
      {
        auto const& ttj = supergates[j].function;

        /* get the same functionality */
        if ( skip_gates[j] || tti != ttj )
          continue;

        if ( _ps.load_minimum_size_only )
        {
          if ( compare_sizes( supergates[i], supergates[j] ) )
          {
            skip_gates[j] = true;
            continue;
          }
          else
          {
            skip_gates[i] = true;
            break;
          }
        }

        /* is i smaller than j */
        bool smaller = supergates[i].area <= supergates[j].area;

        /* is i faster for every pin */
        bool faster = true;
        for ( uint32_t k = 0; k < tti.num_vars(); ++k )
        {
          if ( supergates[i].tdelay[k] > supergates[j].tdelay[k] )
            faster = false;
        }

        if ( smaller && faster )
        {
          skip_gates[j] = true;
          continue;
        }

        /* is j faster for every pin */
        smaller = supergates[i].area >= supergates[j].area;
        faster = true;
        for ( uint32_t k = 0; k < tti.num_vars(); ++k )
        {
          if ( supergates[j].tdelay[k] > supergates[i].tdelay[k] )
            faster = false;
        }

        if ( smaller && faster )
        {
          skip_gates[i] = true;
          break;
        }
      }
    }
  }
};
