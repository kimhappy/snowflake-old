#ifndef SNOWFLAKE_LSTM_HPP
#define SNOWFLAKE_LSTM_HPP

#include "activation.hpp"
#include "../util/load.hpp"

namespace snowflake::layer {
  template<
    typename _T  ,
    int32_t  _IN ,
    int32_t  _OUT,
    int32_t  _ALIGN = 32 >
  class LSTM {
  private:
    _T bi_  alignas(_ALIGN)[ _OUT ]; // Pre-calculated bias_i + bias_h
    _T bf_  alignas(_ALIGN)[ _OUT ]; // Pre-calculated bias_i + bias_h
    _T bg_  alignas(_ALIGN)[ _OUT ]; // Pre-calculated bias_i + bias_h
    _T bo_  alignas(_ALIGN)[ _OUT ]; // Pre-calculated bias_i + bias_h
    _T wii_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T wif_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T wig_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T wio_ alignas(_ALIGN)[ _OUT ][ _IN  ];
    _T whi_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T whf_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T whg_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T who_ alignas(_ALIGN)[ _OUT ][ _OUT ];
    _T o_   alignas(_ALIGN)[ _OUT ];
    _T c_   alignas(_ALIGN)[ _OUT ];

  public:
    _T outs alignas(_ALIGN)[ _OUT ];

    constexpr LSTM() noexcept:
      c_    { 0 },
      outs  { 0 } {}

    // Param order: bias_i_ifgo bias_h_ifgo weight_i_ifgo weight_h_ifgo
    constexpr auto load(_T const* ptr) noexcept -> _T const* {
      ptr = util::data_load(ptr, bi_ );
      ptr = util::data_load(ptr, bf_ );
      ptr = util::data_load(ptr, bg_ );
      ptr = util::data_load(ptr, bo_ );
      ptr = util::data_load(ptr, o_  );

      for (auto i = (int32_t)0; i < _OUT; ++i)
        bi_[ i ] += o_[ i ];

      ptr = util::data_load(ptr, o_  );

      for (auto i = (int32_t)0; i < _OUT; ++i)
        bf_[ i ] += o_[ i ];

      ptr = util::data_load(ptr, o_  );

      for (auto i = (int32_t)0; i < _OUT; ++i)
        bg_[ i ] += o_[ i ];

      ptr = util::data_load(ptr, o_  );

      for (auto i = (int32_t)0; i < _OUT; ++i)
        bo_[ i ] += o_[ i ];

      ptr = util::data_load(ptr, wii_);
      ptr = util::data_load(ptr, wif_);
      ptr = util::data_load(ptr, wig_);
      ptr = util::data_load(ptr, wio_);
      ptr = util::data_load(ptr, whi_);
      ptr = util::data_load(ptr, whf_);
      ptr = util::data_load(ptr, whg_);
      ptr = util::data_load(ptr, who_);

      return ptr;
    }

    constexpr auto forward(_T const (&ins)[ _IN ]) noexcept -> void {
      for (auto i = (int32_t)0; i < _OUT; ++i) {
        auto acc_i = bi_[ i ];
        auto acc_f = bf_[ i ];
        auto acc_g = bg_[ i ];
        auto acc_o = bo_[ i ];

        for (auto j = (int32_t)0; j < _IN; ++j) {
          acc_i += ins[ j ] * wii_[ i ][ j ];
          acc_f += ins[ j ] * wif_[ i ][ j ];
          acc_g += ins[ j ] * wig_[ i ][ j ];
          acc_o += ins[ j ] * wio_[ i ][ j ];
        }

        for (auto j = (int32_t)0; j < _OUT; ++j) {
          acc_i += outs[ j ] * whi_[ i ][ j ];
          acc_f += outs[ j ] * whf_[ i ][ j ];
          acc_g += outs[ j ] * whg_[ i ][ j ];
          acc_o += outs[ j ] * who_[ i ][ j ];
        }

        o_[ i ] = fast_sigmoid(acc_o);
        c_[ i ] = fast_sigmoid(acc_i) * fast_tanh(acc_g) + fast_sigmoid(acc_f) * c_[ i ];
      }

      for (auto i = (int32_t)0; i < _OUT; ++i)
        outs[ i ] = o_[ i ] * fast_tanh(c_[ i ]);
    }
  };
}

#endif // SNOWFLAKE_LSTM_HPP
