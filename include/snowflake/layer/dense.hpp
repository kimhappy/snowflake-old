#ifndef SNOWFLAKE_DENSE_HPP
#define SNOWFLAKE_DENSE_HPP

#include "../util/load.hpp"

namespace snowflake::layer {
  template<
    typename _T  ,
    int32_t  _IN ,
    int32_t  _OUT,
    int32_t  _ALIGN = 32 >
  class Dense {
  private:
    _T bw_ alignas(_ALIGN)[ _OUT ][ 1 + _IN ];

  public:
    _T outs alignas(_ALIGN)[ _OUT ];

    constexpr Dense() noexcept:
      outs { 0 } {}

    constexpr auto load(_T const* ptr) noexcept -> _T const* {
      for (auto i = (int32_t)0; i < _OUT; ++i)
        ptr = util::data_load(ptr, bw_[ i ], 1);

      for (auto i = (int32_t)0; i < _OUT; ++i)
        ptr = util::data_load(ptr, &bw_[ i ][ 1 ], _IN);

      return ptr;
    }

    constexpr auto forward(_T const (&ins)[ _IN ]) noexcept -> void {
      for (auto i = (int32_t)0; i < _OUT; ++i) {
        auto acc = bw_[ i ][ 0 ];

        for (auto j = (int32_t)0; j < _IN; ++j)
          acc += ins[ j ] * bw_[ i ][ 1 + j ];

        outs[ i ] = acc;
      }
    }
  };
}

#endif // SNOWFLAKE_DENSE_HPP
