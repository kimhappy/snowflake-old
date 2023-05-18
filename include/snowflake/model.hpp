#ifndef SNOWFLAKE_MODEL_HPP
#define SNOWFLAKE_MODEL_HPP

#include "layer/dense.hpp"
#include "layer/lstm.hpp"
#include "layer/gru.hpp"

namespace snowflake {
  template<
    typename _T   ,
    template<
      typename,
      int32_t ,
      int32_t ,
      int32_t >
    typename _RNN ,
    int32_t  _SIZE,
    int32_t  _COND,
    int32_t  _ALIGN = 32 >
  class Model {
  private:
    _T input_ alignas(_ALIGN)[ _COND + 1 ];

    _RNN        < _T, _COND + 1, _SIZE, _ALIGN > rnn_  ;
    layer::Dense< _T, _SIZE    , 1    , _ALIGN > dense_;

    template<
      std::size_t... _Is,
      typename   ... _Ps >
    constexpr auto set_params_(
      std::index_sequence< _Is... >,
      _Ps... ps) noexcept -> void {
      ((input_[ _Is + 1 ] = ps), ...);
    }

  public:
    constexpr Model(_T const* ptr) noexcept {
      ptr = rnn_  .load(ptr);
      ptr = dense_.load(ptr);
    }

    // TODO: fade i/o for parameters
    template< typename... _Ps >
    constexpr auto process(
      float const* const __restrict    input  ,
      float      * const __restrict    output ,
      int32_t      const               samples,
      _Ps                          ... ps) {
      set_params_(std::make_index_sequence< _COND >(), ps...);

      for (auto i = (int32_t)0; i < samples; ++i) {
        input_[ 0 ] = input[ i ];
        rnn_  .forward(input_   );
        dense_.forward(rnn_.outs);
        output[ i ] = dense_.outs[ 0 ] + input[ i ];
      }
    }
  };
}

#endif // SNOWFLAKE_MODEL_HPP
