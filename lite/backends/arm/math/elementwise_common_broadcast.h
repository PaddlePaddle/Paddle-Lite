// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Check "lite/kernels/host/elementwise_op_func.h" to get cpu version
 * This file contains same functions that optimized for arm
 *
 * @warning All function in this file will only focus on single thread
 * performance,
 * and so, will never create any thread. It is caller's duty to create threads
 * and mange these threads.
 */
#pragma once

namespace paddle {
namespace lite {
namespace arm {
namespace math {

/**
 *
 * @tparam Config Config is a struct that stores parameters to run this
 * function.
 *
 * e.g. A F32AddReluConfig will looks like
 *
 * ```cpp
 * struct F32AddReluConfig{
 *   using T = float;
 *   using NeonT = float32x4_t;
 *   constexpr static auto neon_dup = vdupq_n_f32;
 *   constexpr static auto neon_ld = vld1q_f32;
 *   constexpr static auto neon_st = vst1q_f32;
 *   constexpr static float (*naive_active)(float) = naive_relu<float>;
 *   constexpr static float32x4_t (*neon_active)(const float32x4_t&) =
 *   neon_relu_float;
 *   constexpr static auto naive_op = naive_add<float>;
 *   constexpr static auto neon_op = vaddq_f32;
 * };
 * ```
 */
template <class Config, bool IS_X_SINGLE, bool IS_Y_SINGLE>
inline void do_neon_elementwise(const typename Config::T* dinx,
                                const typename Config::T* diny,
                                typename Config::T* dout,
                                int num) {
  static_assert((IS_X_SINGLE && IS_Y_SINGLE) != true,
                "X and Y could not be both single");

  using T = typename Config::T;
  using NeonT = typename Config::NeonT;
  constexpr auto neon_dup = Config::neon_dup;
  constexpr auto neon_ld = Config::neon_ld;
  constexpr auto neon_st = Config::neon_st;
  constexpr auto neon_op = Config::neon_op;
  constexpr auto neon_active = Config::neon_active;
  constexpr auto naive_op = Config::naive_op;
  constexpr auto naive_active = Config::naive_active;
  constexpr bool has_active = static_cast<bool>(Config::naive_active);
  constexpr bool neon_active_defined = static_cast<bool>(Config::neon_active);

  constexpr int k_neont_element_num = sizeof(NeonT) / sizeof(T);
  constexpr int k_neon_t_num_per_loop = Config::cnt_num;
  constexpr int k_batch_element_num =
      k_neon_t_num_per_loop * k_neont_element_num;
  int cnt = num / k_batch_element_num;
  int remain = num % k_batch_element_num;

  auto dinx_ptr = dinx;
  auto diny_ptr = diny;
  auto dout_ptr = dout;

  NeonT rbx;
  if (IS_X_SINGLE) {
    rbx = neon_dup(*dinx);
  }
  NeonT rby;
  if (IS_Y_SINGLE) {
    rby = neon_dup(*diny);
  }

  for (int i = 0; i < cnt; ++i) {
    NeonT dinx0;
    NeonT dinx1;
    NeonT dinx2;
    NeonT dinx3;
    NeonT diny0;
    NeonT diny1;
    NeonT diny2;
    NeonT diny3;
    if (!IS_X_SINGLE) {
      dinx0 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
      dinx1 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
      dinx2 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
      dinx3 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
    }

    if (!IS_Y_SINGLE) {
      diny0 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
      diny1 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
      diny2 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
      diny3 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
    }
    if (IS_X_SINGLE && !IS_Y_SINGLE) {
      dinx0 = neon_op(rbx, diny0);
      dinx1 = neon_op(rbx, diny1);
      dinx2 = neon_op(rbx, diny2);
      dinx3 = neon_op(rbx, diny3);
    }
    if (!IS_X_SINGLE && IS_Y_SINGLE) {
      dinx0 = neon_op(dinx0, rby);
      dinx1 = neon_op(dinx1, rby);
      dinx2 = neon_op(dinx2, rby);
      dinx3 = neon_op(dinx3, rby);
    }
    if (!IS_X_SINGLE && !IS_Y_SINGLE) {
      dinx0 = neon_op(dinx0, diny0);
      dinx1 = neon_op(dinx1, diny1);
      dinx2 = neon_op(dinx2, diny2);
      dinx3 = neon_op(dinx3, diny3);
    }

    if (has_active && neon_active_defined) {
      dinx0 = neon_active(dinx0);
      dinx1 = neon_active(dinx1);
      dinx2 = neon_active(dinx2);
      dinx3 = neon_active(dinx3);
    } else if (has_active) {
      for (int k = 0; k < k_neont_element_num; ++k) {
        dinx0[k] = naive_active(dinx0[k]);
        dinx1[k] = naive_active(dinx1[k]);
        dinx2[k] = naive_active(dinx2[k]);
        dinx3[k] = naive_active(dinx3[k]);
      }
    }

    neon_st(dout_ptr, dinx0);
    dout_ptr += k_neont_element_num;
    neon_st(dout_ptr, dinx1);
    dout_ptr += k_neont_element_num;
    neon_st(dout_ptr, dinx2);
    dout_ptr += k_neont_element_num;
    neon_st(dout_ptr, dinx3);
    dout_ptr += k_neont_element_num;
  }

  if (remain >= k_batch_element_num / 2) {
    NeonT dinx0;
    NeonT dinx1;
    NeonT diny0;
    NeonT diny1;
    if (!IS_X_SINGLE) {
      dinx0 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
      dinx1 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
    }

    if (!IS_Y_SINGLE) {
      diny0 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
      diny1 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
    }
    if (IS_X_SINGLE && !IS_Y_SINGLE) {
      dinx0 = neon_op(rbx, diny0);
      dinx1 = neon_op(rbx, diny1);
    }
    if (!IS_X_SINGLE && IS_Y_SINGLE) {
      dinx0 = neon_op(dinx0, rby);
      dinx1 = neon_op(dinx1, rby);
    }
    if (!IS_X_SINGLE && !IS_Y_SINGLE) {
      dinx0 = neon_op(dinx0, diny0);
      dinx1 = neon_op(dinx1, diny1);
    }

    if (has_active && neon_active_defined) {
      dinx0 = neon_active(dinx0);
      dinx1 = neon_active(dinx1);
    } else if (has_active) {
      for (int k = 0; k < k_neont_element_num; ++k) {
        dinx0[k] = naive_active(dinx0[k]);
        dinx1[k] = naive_active(dinx1[k]);
      }
    }

    neon_st(dout_ptr, dinx0);
    dout_ptr += k_neont_element_num;
    neon_st(dout_ptr, dinx1);
    dout_ptr += k_neont_element_num;

    remain -= k_batch_element_num / 2;
  }
  if (remain >= k_batch_element_num / 4) {
    NeonT dinx0;
    NeonT diny0;
    if (!IS_X_SINGLE) {
      dinx0 = neon_ld(dinx_ptr);
      dinx_ptr += k_neont_element_num;
    }
    if (!IS_Y_SINGLE) {
      diny0 = neon_ld(diny_ptr);
      diny_ptr += k_neont_element_num;
    }
    if (IS_X_SINGLE && !IS_Y_SINGLE) {
      dinx0 = neon_op(rbx, diny0);
    }
    if (!IS_X_SINGLE && IS_Y_SINGLE) {
      dinx0 = neon_op(dinx0, rby);
    }
    if (!IS_X_SINGLE && !IS_Y_SINGLE) {
      dinx0 = neon_op(dinx0, diny0);
    }

    if (has_active && neon_active_defined) {
      dinx0 = neon_active(dinx0);
    } else if (has_active) {
      for (int k = 0; k < k_neont_element_num; ++k) {
        dinx0[k] = naive_active(dinx0[k]);
      }
    }
    neon_st(dout_ptr, dinx0);
    dout_ptr += k_neont_element_num;
    remain -= k_batch_element_num / 4;
  }
  if (remain > 0) {
    T tmp = 0;
    for (int p = 0; p < remain; p++) {
      tmp = naive_op(*dinx_ptr, *diny_ptr);
      if (has_active) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      if (!IS_X_SINGLE) {
        dinx_ptr++;
      }
      if (!IS_Y_SINGLE) {
        diny_ptr++;
      }
    }
  }
}

template <class Config>
inline void neon_elementwise_one_to_range(const typename Config::T* dinx,
                                          const typename Config::T* diny,
                                          typename Config::T* dout,
                                          int num) {
  do_neon_elementwise<Config, true, false>(dinx, diny, dout, num);
}

template <class Config>
inline void neon_elementwise_range_to_one(const typename Config::T* dinx,
                                          const typename Config::T* diny,
                                          typename Config::T* dout,
                                          int num) {
  do_neon_elementwise<Config, false, true>(dinx, diny, dout, num);
}

template <class Config>
inline void neon_elementwise_range_to_range(const typename Config::T* dinx,
                                            const typename Config::T* diny,
                                            typename Config::T* dout,
                                            int num) {
  do_neon_elementwise<Config, false, false>(dinx, diny, dout, num);
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
