#ifndef SNOWFLAKE_GRU_HPP
#define SNOWFLAKE_GRU_HPP

#include "activation.hpp"
#include "../util/load.hpp"

namespace snowflake::layer {
  template<
    typename _T  , // float
    int32_t  _IN , //  2
    int32_t  _OUT, // 40
    int32_t  _ALIGN = 32 >
  class GRU {
  private:
    _T br_  alignas(_ALIGN)[ _OUT ]; // Pre-calculated bias_i + bias_h
    _T bz_  alignas(_ALIGN)[ _OUT ]; // Pre-calculated bias_i + bias_h
    _T bin_ alignas(_ALIGN)[ _OUT ];
    _T bhn_ alignas(_ALIGN)[ _OUT ];
    _T wir_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T wiz_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T win_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T whr_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T whz_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T whn_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T r_   alignas(_ALIGN)[ _OUT ];
    _T z_   alignas(_ALIGN)[ _OUT ];
    _T n_   alignas(_ALIGN)[ _OUT ];

    template<
      int32_t  _N,
      int32_t  _M >
    constexpr auto _muladd(
      _T const (&weight)[ _N ][ _M ],
      _T const (&bias  )[ _N ]      ,
      _T const (&x     )      [ _M ],
      _T       (&dest  )[ _N ]) noexcept -> void {
      for (auto i = (int32_t)0; i < _N; ++i) {
        auto gate = bias[ i ];

        for (auto j = (int32_t)0; j < _M; ++j)
          gate += x[ j ] * weight[ i ][ j ];

        dest[ i ] = gate;
      }
    }

  public:
    _T outs alignas(_ALIGN)[ _OUT ];

    constexpr GRU() noexcept:
      outs { 0 } {}

    // Param order: bias_i_rzn bias_h_rzn weight_i_rzn weight_h_rzn
    constexpr auto load(_T const* ptr) noexcept -> _T const* {
      ptr = util::data_load(ptr, br_ );
      ptr = util::data_load(ptr, bz_ );
      ptr = util::data_load(ptr, bin_);
      ptr = util::data_load(ptr, bhn_);

      for (auto i = (int32_t)0; i < _OUT; ++i)
        br_[ i ] += bhn_[ i ];

      ptr = util::data_load(ptr, bhn_);

      for (auto i = (int32_t)0; i < _OUT; ++i)
        bz_[ i ] += bhn_[ i ];

      ptr = util::data_load(ptr, bhn_);
      ptr = util::data_load(ptr, wir_);
      ptr = util::data_load(ptr, wiz_);
      ptr = util::data_load(ptr, win_);
      ptr = util::data_load(ptr, whr_);
      ptr = util::data_load(ptr, whz_);
      ptr = util::data_load(ptr, whn_);

      return ptr;
    }

    constexpr auto forward(_T const (&ins)[ _IN ]) noexcept -> void {
      _muladd(wir_, br_, ins , r_);
      _muladd(whr_, r_ , outs, r_);
      _muladd(wiz_, bz_, ins , z_);
      _muladd(whz_, z_ , outs, z_);

      for (auto i = (int32_t)0; i < _OUT; ++i) {
        auto gate = bhn_[ i ];

        for (auto j = (int32_t)0; j < _OUT; ++j)
          gate += outs[ j ] * whn_[ i ][ j ];

        r_[ i ] = fast_sigmoid(r_[ i ]) * gate;
      }

      _muladd(win_, bin_, ins, n_);

      for (auto i = (int32_t)0; i < _OUT; ++i) {
        auto const z = fast_sigmoid(z_[ i ]);
        auto const n = fast_tanh   (n_[ i ] + r_[ i ]);
        outs[ i ] = (1 - z) * n + z * outs[ i ];
      }
    }
  };
}

#endif // SNOWFLAKE_GRU_HPP
