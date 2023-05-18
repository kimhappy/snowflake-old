#ifndef SNOWFLAKE_LOAD_HPP
#define SNOWFLAKE_LOAD_HPP

#include <cstdint>

namespace snowflake::util {
  template<
    typename _T,
    typename _A >
  constexpr auto data_load(
    _T const* ptr ,
    _A      & dest) noexcept -> _T const* {
    auto const len      = (int32_t)(sizeof dest / sizeof(_T));
    auto       dest_ptr = (_T*)dest;

    for (auto i = (int32_t)0; i < len; ++i)
      *dest_ptr++ = *ptr++;

    return ptr;
  }

  template< typename _T >
  constexpr auto data_load(
    _T const* ptr ,
    _T      * dest,
    int32_t   len) noexcept -> _T const* {
    auto dest_ptr = (_T*)dest;

    for (auto i = (int32_t)0; i < len; ++i)
      *dest_ptr++ = *ptr++;

    return ptr;
  }
}

#endif // SNOWFLAKE_LOAD_HPP
