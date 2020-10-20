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
template <class Config>
inline void neon_elementwise_range_to_one(const typename Config::T* dinx,
                                          const typename Config::T* diny,
                                          typename Config::T* dout,
                                          int num) {
  using T = typename Config::T;
  using NeonT = typename Config::NeonT;
  constexpr auto neon_ld = Config::neon_ld;
  constexpr auto neon_st = Config::neon_st;
  constexpr auto neon_op = Config::neon_op;
  constexpr auto neon_active = Config::neon_active;
  constexpr auto neon_dup = Config::neon_dup;
  constexpr auto naive_op = Config::naive_op;
  constexpr auto naive_active = Config::naive_active;
  constexpr bool has_active = static_cast<bool>(Config::naive_active);
  constexpr bool neon_active_defined = static_cast<bool>(Config::neon_active);

  const T diny_data = *diny;

  int cnt = num >> 4;
  int remain = num % 16;
  NeonT rb = neon_dup(diny_data);
  for (int i = 0; i < cnt; ++i) {
    int offset = i << 4;
    auto dinx_ptr = dinx + offset;
    auto dout_ptr = dout + offset;

    NeonT din0 = neon_ld(dinx_ptr);
    NeonT din1 = neon_ld(dinx_ptr + 4);
    NeonT din2 = neon_ld(dinx_ptr + 8);
    NeonT din3 = neon_ld(dinx_ptr + 12);

    din0 = neon_op(din0, rb);
    din1 = neon_op(din1, rb);
    din2 = neon_op(din2, rb);
    din3 = neon_op(din3, rb);

    if (has_active && neon_active_defined) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
      din2 = neon_active(din2);
      din3 = neon_active(din3);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        din0[k] = naive_active(din0[k]);
        din1[k] = naive_active(din1[k]);
        din2[k] = naive_active(din2[k]);
        din3[k] = naive_active(din3[k]);
      }
    }

    neon_st(dout_ptr, din0);
    neon_st(dout_ptr + 4, din1);
    neon_st(dout_ptr + 8, din2);
    neon_st(dout_ptr + 12, din3);
  }
  int offset = cnt << 4;
  auto dinx_ptr = dinx + offset;
  auto dout_ptr = dout + offset;

  if (remain >= 8) {
    NeonT din0 = neon_ld(dinx_ptr);
    NeonT din1 = neon_ld(dinx_ptr + 4);

    din0 = neon_op(din0, rb);
    din1 = neon_op(din1, rb);

    if (has_active && neon_active_defined) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        din0[k] = naive_active(din0[k]);
        din1[k] = naive_active(din1[k]);
      }
    }

    neon_st(dout_ptr, din0);
    neon_st(dout_ptr + 4, din1);

    dinx_ptr += 8;
    dout_ptr += 8;
    remain -= 8;
  }
  if (remain >= 4) {
    NeonT din0 = neon_ld(dinx_ptr);

    din0 = neon_op(din0, rb);

    if (has_active && neon_active_defined) {
      din0 = neon_active(din0);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        din0[k] = naive_active(din0[k]);
      }
    }
    neon_st(dout_ptr, din0);
    dinx_ptr += 4;
    dout_ptr += 4;
    remain -= 4;
  }
  if (remain > 0) {
    T tmp = 0;
    for (int p = 0; p < remain; p++) {
      tmp = naive_op(*dinx_ptr, diny_data);
      if (has_active) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      dinx_ptr++;
    }
  }
}

/**
 * see neon_elementwise_range_to_one to get how to use Config
 */
template <class Config>
inline void neon_elementwise_one_to_range(const typename Config::T* dinx,
                                          const typename Config::T* diny,
                                          typename Config::T* dout,
                                          int num) {
  using T = typename Config::T;
  using NeonT = typename Config::NeonT;
  constexpr auto neon_ld = Config::neon_ld;
  constexpr auto neon_st = Config::neon_st;
  constexpr auto neon_op = Config::neon_op;
  constexpr auto neon_active = Config::neon_active;
  constexpr auto neon_dup = Config::neon_dup;
  constexpr auto naive_op = Config::naive_op;
  constexpr auto naive_active = Config::naive_active;
  constexpr bool has_active = static_cast<bool>(Config::naive_active);
  constexpr bool neon_active_defined = static_cast<bool>(Config::neon_active);

  const T dinx_data = *dinx;

  int cnt = num >> 4;
  int remain = num % 16;
  NeonT rb = neon_dup(dinx_data);
  for (int i = 0; i < cnt; ++i) {
    int offset = i << 4;
    auto diny_ptr = diny + offset;
    auto dout_ptr = dout + offset;

    NeonT din0 = neon_ld(diny_ptr);
    NeonT din1 = neon_ld(diny_ptr + 4);
    NeonT din2 = neon_ld(diny_ptr + 8);
    NeonT din3 = neon_ld(diny_ptr + 12);

    din0 = neon_op(rb, din0);
    din1 = neon_op(rb, din1);
    din2 = neon_op(rb, din2);
    din3 = neon_op(rb, din3);

    if (has_active && neon_active_defined) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
      din2 = neon_active(din2);
      din3 = neon_active(din3);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        din0[k] = naive_active(din0[k]);
        din1[k] = naive_active(din1[k]);
        din2[k] = naive_active(din2[k]);
        din3[k] = naive_active(din3[k]);
      }
    }

    neon_st(dout_ptr, din0);
    neon_st(dout_ptr + 4, din1);
    neon_st(dout_ptr + 8, din2);
    neon_st(dout_ptr + 12, din3);
  }
  int offset = cnt << 4;
  auto diny_ptr = diny + offset;
  auto dout_ptr = dout + offset;
  if (remain >= 8) {
    NeonT din0 = neon_ld(diny_ptr);
    NeonT din1 = neon_ld(diny_ptr + 4);

    din0 = neon_op(rb, din0);
    din1 = neon_op(rb, din1);

    if (has_active && neon_active_defined) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        din0[k] = naive_active(din0[k]);
        din1[k] = naive_active(din1[k]);
      }
    }

    neon_st(dout_ptr, din0);
    neon_st(dout_ptr + 4, din1);

    diny_ptr += 8;
    dout_ptr += 8;
    remain -= 8;
  }
  if (remain >= 4) {
    NeonT din0 = neon_ld(diny_ptr);

    din0 = neon_op(rb, din0);

    if (has_active && neon_active_defined) {
      din0 = neon_active(din0);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        din0[k] = naive_active(din0[k]);
      }
    }
    neon_st(dout_ptr, din0);
    diny_ptr += 4;
    dout_ptr += 4;
    remain -= 4;
  }
  if (remain > 0) {
    T tmp = 0;
    for (int p = 0; p < remain; p++) {
      tmp = naive_op(dinx_data, *diny_ptr);
      if (has_active) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      diny_ptr++;
    }
  }
}

/**
 * see neon_elementwise_range_to_one to get how to use Config
 */
template <class Config>
inline void neon_elementwise_range_to_range(const typename Config::T* dinx,
                                            const typename Config::T* diny,
                                            typename Config::T* dout,
                                            int num) {
  using T = typename Config::T;
  using NeonT = typename Config::NeonT;
  constexpr auto neon_ld = Config::neon_ld;
  constexpr auto neon_st = Config::neon_st;
  constexpr auto neon_op = Config::neon_op;
  constexpr auto neon_active = Config::neon_active;
  constexpr auto naive_op = Config::naive_op;
  constexpr auto naive_active = Config::naive_active;
  constexpr bool has_active = static_cast<bool>(Config::naive_active);
  constexpr bool neon_active_defined = static_cast<bool>(Config::neon_active);

  int cnt = num >> 4;
  int remain = num % 16;

  for (int i = 0; i < cnt; ++i) {
    int offset = i << 4;
    auto dinx_ptr = dinx + offset;
    auto diny_ptr = diny + offset;
    auto dout_ptr = dout + offset;

    NeonT dinx0 = neon_ld(dinx_ptr);
    NeonT dinx1 = neon_ld(dinx_ptr + 4);
    NeonT dinx2 = neon_ld(dinx_ptr + 8);
    NeonT dinx3 = neon_ld(dinx_ptr + 12);

    NeonT diny0 = neon_ld(diny_ptr);
    NeonT diny1 = neon_ld(diny_ptr + 4);
    NeonT diny2 = neon_ld(diny_ptr + 8);
    NeonT diny3 = neon_ld(diny_ptr + 12);

    dinx0 = neon_op(dinx0, diny0);
    dinx1 = neon_op(dinx1, diny1);
    dinx2 = neon_op(dinx2, diny2);
    dinx3 = neon_op(dinx3, diny3);

    if (has_active && neon_active_defined) {
      dinx0 = neon_active(dinx0);
      dinx1 = neon_active(dinx1);
      dinx2 = neon_active(dinx2);
      dinx3 = neon_active(dinx3);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        dinx0[k] = naive_active(dinx0[k]);
        dinx1[k] = naive_active(dinx1[k]);
        dinx2[k] = naive_active(dinx2[k]);
        dinx3[k] = naive_active(dinx3[k]);
      }
    }

    neon_st(dout_ptr, dinx0);
    neon_st(dout_ptr + 4, dinx1);
    neon_st(dout_ptr + 8, dinx2);
    neon_st(dout_ptr + 12, dinx3);
  }

  int offset = cnt << 4;
  auto dinx_ptr = dinx + offset;
  auto diny_ptr = diny + offset;
  auto dout_ptr = dout + offset;
  if (remain >= 8) {
    NeonT dinx0 = neon_ld(dinx_ptr);
    NeonT dinx1 = neon_ld(dinx_ptr + 4);

    NeonT diny0 = neon_ld(diny_ptr);
    NeonT diny1 = neon_ld(diny_ptr + 4);

    dinx0 = neon_op(dinx0, diny0);
    dinx1 = neon_op(dinx1, diny1);

    if (has_active && neon_active_defined) {
      dinx0 = neon_active(dinx0);
      dinx1 = neon_active(dinx1);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        dinx0[k] = naive_active(dinx0[k]);
        dinx1[k] = naive_active(dinx1[k]);
      }
    }

    neon_st(dout_ptr, dinx0);
    neon_st(dout_ptr + 4, dinx1);

    dinx_ptr += 8;
    diny_ptr += 8;
    dout_ptr += 8;
    remain -= 8;
  }
  if (remain >= 4) {
    NeonT dinx0 = neon_ld(dinx_ptr);

    NeonT diny0 = neon_ld(diny_ptr);

    dinx0 = neon_op(dinx0, diny0);

    if (has_active && neon_active_defined) {
      dinx0 = neon_active(dinx0);
    } else if (has_active) {
      for (int k = 0; k < 4; ++k) {
        dinx0[k] = naive_active(dinx0[k]);
      }
    }
    neon_st(dout_ptr, dinx0);
    dinx_ptr += 4;
    diny_ptr += 4;
    dout_ptr += 4;
    remain -= 4;
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
      dinx_ptr++;
      diny_ptr++;
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
