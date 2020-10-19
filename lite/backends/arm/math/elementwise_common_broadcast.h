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

#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

enum class FusedActive { NO, YES };
enum class HasNeonActive { NO, YES };

template <class T,
          class NeonT,
          T naive_op(T, T),
          NeonT neon_dup(T),
          NeonT neon_ld(const T*),
          void neon_st(T*, NeonT),
          NeonT neon_op(NeonT, NeonT),
          NeonT neon_active(const NeonT&) = nullptr,
          T naive_active(T) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_range_to_one(const T* dinx,
                                          const T* diny,
                                          T* dout,
                                          int num) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
      din2 = neon_active(din2);
      din3 = neon_active(din3);
    } else if (fused_active == FusedActive::YES) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
    } else if (fused_active == FusedActive::YES) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      din0 = neon_active(din0);
    } else if (fused_active == FusedActive::YES) {
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
      if (fused_active == FusedActive::YES) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      dinx_ptr++;
    }
  }
}

template <class T,
          class NeonT,
          T naive_op(T, T),
          NeonT neon_dup(T),
          NeonT neon_ld(const T*),
          void neon_st(T*, NeonT),
          NeonT neon_op(NeonT, NeonT),
          NeonT neon_active(const NeonT&) = nullptr,
          T naive_active(T) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_one_to_range(const T* dinx,
                                          const T* diny,
                                          T* dout,
                                          int num) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
      din2 = neon_active(din2);
      din3 = neon_active(din3);
    } else if (fused_active == FusedActive::YES) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      din0 = neon_active(din0);
      din1 = neon_active(din1);
    } else if (fused_active == FusedActive::YES) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      din0 = neon_active(din0);
    } else if (fused_active == FusedActive::YES) {
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
      if (fused_active == FusedActive::YES) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      diny_ptr++;
    }
  }
}

template <class T,
          class NeonT,
          T naive_op(T, T),
          NeonT neon_ld(const T*),
          void neon_st(T*, NeonT),
          NeonT neon_op(NeonT, NeonT),
          NeonT neon_active(const NeonT&) = nullptr,
          T naive_active(T) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_range_to_range(const T* dinx,
                                            const T* diny,
                                            T* dout,
                                            int num) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      dinx0 = neon_active(dinx0);
      dinx1 = neon_active(dinx1);
      dinx2 = neon_active(dinx2);
      dinx3 = neon_active(dinx3);
    } else if (fused_active == FusedActive::YES) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      dinx0 = neon_active(dinx0);
      dinx1 = neon_active(dinx1);
    } else if (fused_active == FusedActive::YES) {
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

    if (fused_active == FusedActive::YES &&
        has_neon_active == HasNeonActive::YES) {
      dinx0 = neon_active(dinx0);
    } else if (fused_active == FusedActive::YES) {
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
      if (fused_active == FusedActive::YES) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

///////////////////// Helper Functions Int32 /////////////////////

template <int32x4_t neon_op(int32x4_t, int32x4_t),
          int32_t naive_op(int32_t, int32_t),
          int32_t naive_active(int32_t) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          int32x4_t neon_active(const int32x4_t&) = nullptr,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_range_to_one_i32(const int32_t* dinx,
                                              const int32_t* diny,
                                              int32_t* dout,
                                              int num) {
  neon_elementwise_range_to_one<int32_t,
                                int32x4_t,
                                naive_op,
                                vdupq_n_s32,
                                vld1q_s32,
                                vst1q_s32,
                                neon_op,
                                neon_active,
                                naive_active,
                                fused_active,
                                has_neon_active>(dinx, diny, dout, num);
}

template <int32x4_t neon_op(int32x4_t, int32x4_t),
          int32_t naive_op(int32_t, int32_t),
          int32_t naive_active(int32_t) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          int32x4_t neon_active(const int32x4_t&) = nullptr,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_one_to_range_i32(const int32_t* dinx,
                                              const int32_t* diny,
                                              int32_t* dout,
                                              int num) {
  neon_elementwise_one_to_range<int32_t,
                                int32x4_t,
                                naive_op,
                                vdupq_n_s32,
                                vld1q_s32,
                                vst1q_s32,
                                neon_op,
                                naive_active,
                                fused_active,
                                has_neon_active>(dinx, diny, dout, num);
}

template <int32x4_t neon_op(int32x4_t, int32x4_t),
          int32_t naive_op(int32_t, int32_t),
          int32_t naive_active(int32_t) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          int32x4_t neon_active(const int32x4_t&) = nullptr,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_range_to_range_i32(const int32_t* dinx,
                                                const int32_t* diny,
                                                int32_t* dout,
                                                int num) {
  neon_elementwise_range_to_range<int32_t,
                                  int32x4_t,
                                  naive_op,
                                  vld1q_s32,
                                  vst1q_s32,
                                  neon_op,
                                  naive_active,
                                  fused_active,
                                  has_neon_active>(dinx, diny, dout, num);
}

///////////////////// Helper Functions Float /////////////////////

template <float32x4_t neon_op(float32x4_t, float32x4_t),
          float naive_op(float, float),
          float naive_active(float) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          float32x4_t neon_active(const float32x4_t&) = nullptr,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_range_to_one_i32(const float* dinx,
                                              const float* diny,
                                              float* dout,
                                              int num) {
  neon_elementwise_range_to_one<float,
                                float32x4_t,
                                naive_op,
                                vdupq_n_f32,
                                vld1q_f32,
                                vst1q_f32,
                                neon_op,
                                neon_active,
                                naive_active,
                                fused_active,
                                has_neon_active>(dinx, diny, dout, num);
}

template <float32x4_t neon_op(float32x4_t, float32x4_t),
          float naive_op(float, float),
          float naive_active(float) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          float32x4_t neon_active(const float32x4_t&) = nullptr,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_one_to_range_i32(const float* dinx,
                                              const float* diny,
                                              float* dout,
                                              int num) {
  neon_elementwise_one_to_range<float,
                                float32x4_t,
                                naive_op,
                                vdupq_n_f32,
                                vld1q_f32,
                                vst1q_f32,
                                neon_op,
                                naive_active,
                                fused_active,
                                has_neon_active>(dinx, diny, dout, num);
}

template <float32x4_t neon_op(float32x4_t, float32x4_t),
          float naive_op(float, float),
          float naive_active(float) = nullptr,
          FusedActive fused_active = FusedActive::NO,
          float32x4_t neon_active(const float32x4_t&) = nullptr,
          HasNeonActive has_neon_active = HasNeonActive::NO>
inline void neon_elementwise_range_to_range_i32(const float* dinx,
                                                const float* diny,
                                                float* dout,
                                                int num) {
  neon_elementwise_range_to_range<float,
                                  float32x4_t,
                                  naive_op,
                                  vld1q_f32,
                                  vst1q_f32,
                                  neon_op,
                                  naive_active,
                                  fused_active,
                                  has_neon_active>(dinx, diny, dout, num);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
