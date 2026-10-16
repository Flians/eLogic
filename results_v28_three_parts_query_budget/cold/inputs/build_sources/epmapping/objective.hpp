#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

namespace epmapping
{
inline std::optional<uint8_t> parse_objective( std::string_view value )
{
  if ( value == "0" || value == "delay_area" || value == "(delay,area)" )
    return 0u;
  if ( value == "1" || value == "area_delay" || value == "(area,delay)" ||
       value == "area_delay_bound" )
    return 1u;
  return std::nullopt;
}

inline char const* objective_name( uint8_t objective )
{
  return objective == 0u ? "delay_area" : "area_delay";
}

// Match the mapper's existing comparison resolution: area 1e-4, delay 1e-3.
inline std::pair<int64_t, int64_t> objective_key( uint8_t objective, double area, double delay )
{
  auto const a = static_cast<int64_t>( std::llround( area * 1e4 ) );
  auto const d = static_cast<int64_t>( std::llround( delay * 1e3 ) );
  return objective == 0u ? std::make_pair( d, a ) : std::make_pair( a, d );
}
} // namespace epmapping
