#ifndef SNOWFLAKE_ACTIVATION_HPP
#define SNOWFLAKE_ACTIVATION_HPP

#include "../util/table.hpp"

namespace snowflake::layer {
  template< typename _T >
  inline auto fast_sigmoid(_T value) noexcept -> _T {
    static auto table = util::Table<
      _T, decltype([](auto x) noexcept {
        return 1 / (1 + std::exp(-x)); }), 100000,
      SNOWFLAKE_MAKE_TEMPLATE_VALUE((_T)-6),
      SNOWFLAKE_MAKE_TEMPLATE_VALUE((_T) 6) > {};

    return table(value);
  }

  template< typename _T >
  inline auto fast_tanh(_T value) noexcept -> _T {
    static auto table = util::Table<
      _T, decltype([](auto x) noexcept {
        return std::tanh(x); }), 1000000,
      SNOWFLAKE_MAKE_TEMPLATE_VALUE((_T)-6),
      SNOWFLAKE_MAKE_TEMPLATE_VALUE((_T) 6) > {};

    return table(value);
  }
}

#endif // SNOWFLAKE_ACTIVATION_HPP
