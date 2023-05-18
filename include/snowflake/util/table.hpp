#ifndef SNOWFLAKE_TABLE_HPP
#define SNOWFLAKE_TABLE_HPP

#include <vector>
#include <cmath>
#include <algorithm>

// Enable this macro to print the debug information for fast_xxx.
// #define SNOWFLAKE_TABLE_DEBUG

// As clang++ doesn't support float template yet, we have to use this workaround.
#define SNOWFLAKE_MAKE_TEMPLATE_VALUE(x_) decltype([]() noexcept { return x_; })
#define SNOWFLAKE_GET_TEMPLATE_VALUE(name_) (name_()())

#ifdef SNOWFLAKE_TABLE_DEBUG
#  include <iostream>
#  include <limits>
#endif // SNOWFLAKE_TABLE_DEBUG

namespace snowflake::util {
    template<
    typename _T            ,
    typename _F            ,
    int32_t  _STEP         ,
    typename _BEGIN        ,
    typename _END          ,
    bool     _CLAMP  = true,
    bool     _ENABLE = true >
  struct Table {
  private:
    static constexpr auto _BEGIN_VALUE = (_T)SNOWFLAKE_GET_TEMPLATE_VALUE(_BEGIN);
    static constexpr auto _END_VALUE   = (_T)SNOWFLAKE_GET_TEMPLATE_VALUE(_END  );

    static_assert(_STEP        > 0         , "_STEP must be positive."       );
    static_assert(_BEGIN_VALUE < _END_VALUE, "_BEGIN must be less than _END.");

    static constexpr auto _MOVE     = (_END_VALUE - _BEGIN_VALUE) / (_T)_STEP;
    static constexpr auto _MOVE_INV = 1 / _MOVE;

    static inline auto _TABLE = []() noexcept {
      auto ret = std::vector< _T >(_STEP);

      for (auto i = (int32_t)0; i < _STEP; ++i)
        ret[ i ] = _F()(_BEGIN_VALUE + i * _MOVE);

      return ret;
    } ();

#ifdef SNOWFLAKE_TABLE_DEBUG
    _T min_       = std::numeric_limits< _T >::max();
    _T max_       = std::numeric_limits< _T >::min();
    _T max_error_ = 0                               ;
#endif // SNOWFLAKE_TABLE_DEBUG

    static constexpr auto index_(_T value) noexcept -> int32_t {
      auto const index = (int32_t)std::lround((value - _BEGIN_VALUE) * _MOVE_INV);

      if constexpr (_CLAMP)
        return std::clamp(index, 0, _STEP - 1);
      else
        return index;
    }

  public:
#ifdef SNOWFLAKE_TABLE_DEBUG
      constexpr ~Table() noexcept {
        if (min_ == std::numeric_limits< _T >::max())
          return;

        std::cout << "value min: " << min_       << '\n';
        std::cout << "value max: " << max_       << '\n';
        std::cout << "error max: " << max_error_ << '\n';
      }
#endif // SNOWFLAKE_TABLE_DEBUG

    constexpr auto operator()(_T value)
#ifndef SNOWFLAKE_TABLE_DEBUG
      const
#endif // SNOWFLAKE_TABLE_DEBUG
    noexcept -> _T {
#ifdef SNOWFLAKE_TABLE_DEBUG
      if (value < min_)
        min_ = value;

      if (value > max_)
        max_ = value;
#endif // SNOWFLAKE_TABLE_DEBUG

      if constexpr (_ENABLE) {
        auto const ret = _TABLE[ index_(value) ];

#ifdef SNOWFLAKE_TABLE_DEBUG
        if (auto const error = std::abs(_F()(value) - ret); error > max_error_)
          max_error_ = error;
#endif // SNOWFLAKE_TABLE_DEBUG

        return ret;
      }
      else {
        return _F()(value);
      }
    }
  };
}

#endif // SNOWFLAKE_TABLE_HPP
