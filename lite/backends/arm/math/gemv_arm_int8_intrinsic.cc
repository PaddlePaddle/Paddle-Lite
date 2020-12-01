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

#include "lite/backends/arm/math/gemv_arm_int8_intrinsic.h"

#include <arm_neon.h>

#include <cmath>

#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <bool has_bias,
          float32x2_t active_fn(float32x2_t, const void*),
          int loop_num>
struct neon_process_output_int8_x2 {
  static inline void run(const int32x2_t* in,
                         const float* __restrict__ scale,
                         const float* __restrict__ bias,
                         const void* __restrict__ active_args,
                         int32x2_t* out_buf) {
    neon_process_output_int8_x2<has_bias, active_fn, loop_num - 1>::run(
        in, scale, bias, active_args, out_buf);
    constexpr bool has_active_fn = static_cast<bool>(active_fn);
    constexpr int i = loop_num - 1;
    float32x2_t vscale0 = vld1_f32(scale + 2 * i);
    float32x2_t vin0 = vcvt_f32_s32(in[i]);
    float32x2_t tmp;
    if (has_bias) {
      float32x2_t vbias0 = vld1_f32(bias + 2 * i);
      if (has_active_fn) {
        tmp = active_fn(vmla_f32(vbias0, vin0, vscale0), active_args);
      } else {
        tmp = vmla_f32(vbias0, vin0, vscale0);
      }
    } else {
      if (has_active_fn) {
        tmp = active_fn(vmul_f32(vin0, vscale0), active_args);
      } else {
        tmp = vmul_f32(vin0, vscale0);
      }
    }

    float32x2_t round_bias = vbsl_f32(
        vcgt_f32(tmp, vmov_n_f32(0)), vmov_n_f32(0.5), vmov_n_f32(-0.5));
    out_buf[i] = vcvt_s32_f32(tmp + round_bias);

    out_buf[i] = vmax_s32(out_buf[i], vmov_n_s32(-127));
    out_buf[i] = vmin_s32(out_buf[i], vmov_n_s32(127));
  }
};

template <bool has_bias, float32x2_t active_fn(float32x2_t, const void*)>
struct neon_process_output_int8_x2<has_bias, active_fn, 0> {
  static inline void run(const int32x2_t* in,
                         const float* __restrict__ scale,
                         const float* __restrict__ bias,
                         const void* __restrict__ active_args,
                         int32x2_t* out_buf) {
    return;
  }
};

static inline float32x2_t neon_relu(float32x2_t in, const void* unused) {
  return vmax_f32(in, vmov_n_f32(0));
}

struct relu6_params_x2 {
  float32x2_t six;
};
static inline float32x2_t neon_relu6(float32x2_t in,
                                     const void* active_params) {
  return vmin_f32(vmax_f32(in, vmov_n_f32(0)),
                  static_cast<const relu6_params_x2*>(active_params)->six);
}

struct leaky_relu_params_x2 {
  float32x2_t slope;
};
static inline float32x2_t neon_leaky_relu(float32x2_t in,
                                          const void* active_params) {
  float32x2_t vacc0123 = vmul_f32(
      in, static_cast<const leaky_relu_params_x2*>(active_params)->slope);
  const uint32x2_t vmask = vclt_s32(vreinterpret_s32_f32(in), vmov_n_s32(0));
  return vbsl_f32(vmask, vacc0123, in);
}

template <int kSize, bool has_bias, bool has_active>
inline void neon_write_gemv_int8_out(const int32x2_t* __restrict__ in,
                                     signed char* __restrict__ out,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ bias,
                                     lite_api::ActivationType act,
                                     float six,
                                     float alpha) {
  int32x2_t out_buf[kSize];

  if (has_active) {
    switch (act) {
      case lite_api::ActivationType::kRelu:
        neon_process_output_int8_x2<has_bias, neon_relu, kSize>::run(
            in, scale, bias, nullptr, out_buf);
        break;
      case lite_api::ActivationType::kRelu6: {
        relu6_params_x2 arg;
        arg.six = vmov_n_f32(six);
        neon_process_output_int8_x2<has_bias, neon_relu6, kSize>::run(
            in, scale, bias, &arg, out_buf);
        break;
      }
      case lite_api::ActivationType::kLeakyRelu: {
        leaky_relu_params_x2 arg;
        arg.slope = vmov_n_f32(alpha);
        neon_process_output_int8_x2<has_bias, neon_leaky_relu, kSize>::run(
            in, scale, bias, &arg, out_buf);
        break;
      }
      default:
        LOG(FATAL) << "act not supported";
    }

  } else {
    neon_process_output_int8_x2<has_bias, nullptr, kSize>::run(
        in, scale, bias, nullptr, out_buf);
  }

  for (int i = 0; i < kSize; ++i) {
    out[0] = out_buf[i][0];
    out[1] = out_buf[i][1];
    out += 2;
  }
}

template <bool has_bias,
          float32x4_t active_fn(float32x4_t, const void*),
          int loop_num>
struct neon_process_output_int8 {
  static inline void run(const int32x4_t* in,
                         const float* __restrict__ scale,
                         const float* __restrict__ bias,
                         const void* __restrict__ active_args,
                         int32x4_t* out_buf) {
    neon_process_output_int8<has_bias, active_fn, loop_num - 1>::run(
        in, scale, bias, active_args, out_buf);
    constexpr bool has_active_fn = static_cast<bool>(active_fn);
    constexpr int i = loop_num - 1;
    float32x4_t vscale0 = vld1q_f32(scale + 4 * i);
    float32x4_t vin0 = vcvtq_f32_s32(in[i]);

    float32x4_t tmp;
    if (has_bias) {
      float32x4_t vbias0 = vld1q_f32(bias + 4 * i);
      if (has_active_fn) {
        tmp = active_fn(vmlaq_f32(vbias0, vin0, vscale0), active_args);
      } else {
        tmp = vmlaq_f32(vbias0, vin0, vscale0);
      }
    } else {
      if (has_active_fn) {
        tmp = active_fn(vmulq_f32(vin0, vscale0), active_args);
      } else {
        tmp = vmulq_f32(vin0, vscale0);
      }
    }
    float32x4_t round_bias = vbslq_f32(
        vcgtq_f32(tmp, vmovq_n_f32(0)), vmovq_n_f32(0.5), vmovq_n_f32(-0.5));
    out_buf[i] = vcvtq_s32_f32(tmp + round_bias);

    out_buf[i] = vmaxq_s32(out_buf[i], vmovq_n_s32(-127));
    out_buf[i] = vminq_s32(out_buf[i], vmovq_n_s32(127));
  }
};

template <bool has_bias, float32x4_t active_fn(float32x4_t, const void*)>
struct neon_process_output_int8<has_bias, active_fn, 0> {
  static inline void run(const int32x4_t* in,
                         const float* __restrict__ scale,
                         const float* __restrict__ bias,
                         const void* __restrict__ active_args,
                         int32x4_t* out_buf) {
    return;
  }
};

static inline float32x4_t neon_relu(float32x4_t in, const void* unused) {
  return vmaxq_f32(in, vmovq_n_f32(0));
}

struct relu6_params {
  float32x4_t six;
};
static inline float32x4_t neon_relu6(float32x4_t in,
                                     const void* active_params) {
  return vminq_f32(vmaxq_f32(in, vmovq_n_f32(0)),
                   static_cast<const relu6_params*>(active_params)->six);
}

struct leaky_relu_params {
  float32x4_t slope;
};
static inline float32x4_t neon_leaky_relu(float32x4_t in,
                                          const void* active_params) {
  float32x4_t vacc0123 = vmulq_f32(
      in, static_cast<const leaky_relu_params*>(active_params)->slope);
  const uint32x4_t vmask = vcltq_s32(vreinterpretq_s32_f32(in), vmovq_n_s32(0));
  return vbslq_f32(vmask, vacc0123, in);
}

template <int kSize, bool has_bias, bool has_active>
inline void neon_write_gemv_int8_out(const int32x4_t* __restrict__ in,
                                     signed char* __restrict__ out,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ bias,
                                     lite_api::ActivationType act,
                                     float six,
                                     float alpha) {
  int32x4_t out_buf[kSize];

  if (has_active) {
    switch (act) {
      case lite_api::ActivationType::kRelu:
        neon_process_output_int8<has_bias, neon_relu, kSize>::run(
            in, scale, bias, nullptr, out_buf);
        break;
      case lite_api::ActivationType::kRelu6: {
        relu6_params arg;
        arg.six = vmovq_n_f32(six);
        neon_process_output_int8<has_bias, neon_relu6, kSize>::run(
            in, scale, bias, &arg, out_buf);
        break;
      }
      case lite_api::ActivationType::kLeakyRelu: {
        leaky_relu_params arg;
        arg.slope = vmovq_n_f32(alpha);
        neon_process_output_int8<has_bias, neon_leaky_relu, kSize>::run(
            in, scale, bias, &arg, out_buf);
        break;
      }
      default:
        LOG(FATAL) << "act not supported";
    }

  } else {
    neon_process_output_int8<has_bias, nullptr, kSize>::run(
        in, scale, bias, nullptr, out_buf);
  }

  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[j] = out_buf[i][j];
    }
    out += 4;
  }
}

template <int kSize, bool has_bias, bool has_active>
inline void write_gemv_int8_out(const int32_t* __restrict__ in,
                                signed char* __restrict__ out,
                                const float* __restrict__ scale,
                                const float* __restrict__ bias,
                                lite_api::ActivationType act,
                                float six,
                                float alpha) {
  int32_t out_buf[kSize];

  if (has_active) {
    switch (act) {
      case lite_api::ActivationType::kRelu:
        for (int i = 0; i < kSize; ++i) {
          float tmp = in[i];
          tmp = tmp * scale[i] + (has_bias ? (bias[i]) : 0);
          out_buf[i] = roundf(tmp > 0 ? tmp : 0);
        }
        break;
      case lite_api::ActivationType::kRelu6: {
        for (int i = 0; i < kSize; ++i) {
          float tmp = in[i];
          tmp = tmp * scale[i] + (has_bias ? (bias[i]) : 0);
          tmp = tmp > 0 ? tmp : 0;
          out_buf[i] = roundf(tmp > six ? six : tmp);
        }
        break;
      }
      case lite_api::ActivationType::kLeakyRelu: {
        for (int i = 0; i < kSize; ++i) {
          float tmp = in[i];
          tmp = tmp * scale[i] + (has_bias ? (bias[i]) : 0);
          out_buf[i] = roundf(tmp > 0 ? tmp : (tmp * alpha));
        }
        break;
      }
      default:
        LOG(FATAL) << "act not supported";
    }

  } else {
    for (int i = 0; i < kSize; ++i) {
      float tmp = in[i] * scale[i] + (has_bias ? (bias[i]) : 0);
      out_buf[i] = roundf(tmp);
    }
  }

  for (int i = 0; i < kSize; ++i) {
    int tmp = out_buf[i];
    tmp = tmp > 127 ? 127 : tmp;
    tmp = tmp < -127 ? -127 : tmp;
    out[i] = tmp;
  }
}

template <int i>
inline void loop_mul_add(const int8x16_t& col0,
                         const int8x16_t* tile_row,
                         int32x4_t* q) {
  loop_mul_add<i - 1>(col0, tile_row, q);
  int16x8_t mul_lo = vmull_s8(vget_low_s8(col0), vget_low_s8(tile_row[i - 1]));
  int16x8_t mul_hi =
      vmull_s8(vget_high_s8(col0), vget_high_s8(tile_row[i - 1]));
  q[i - 1] = vpadalq_s16(q[i - 1], mul_lo + mul_hi);
}

template <>
inline void loop_mul_add<0>(const int8x16_t& col0,
                            const int8x16_t* tile_row,
                            int32x4_t* q) {
  return;
}

template <int TILE_A_WIDTH, int i>
struct loop_mul_add_load {
  static inline void run(const int8_t* __restrict__ tile_A_row_data[],
                         const int8x16_t& col0,
                         int8x16_t* tile_row,
                         int32x4_t* q) {
    loop_mul_add_load<TILE_A_WIDTH, i - 1>::run(
        tile_A_row_data, col0, tile_row, q);
    int16x8_t mul_lo =
        vmull_s8(vget_low_s8(col0), vget_low_s8(tile_row[i - 1]));
    int16x8_t mul_hi =
        vmull_s8(vget_high_s8(col0), vget_high_s8(tile_row[i - 1]));
    tile_row[i - 1] = vld1q_s8(tile_A_row_data[i - 1]);
    tile_A_row_data[i - 1] += TILE_A_WIDTH;
    q[i - 1] = vpadalq_s16(q[i - 1], mul_lo + mul_hi);
  }
};

template <int TILE_A_WIDTH>
struct loop_mul_add_load<TILE_A_WIDTH, 0> {
  static inline void run(const int8_t* __restrict__ tile_A_row_data[],
                         const int8x16_t& col0,
                         int8x16_t* tile_row,
                         int32x4_t* q) {
    return;
  }
};

template <int TILE_A_WIDTH,
          int TILE_A_HEIGHT,
          int const_tile_j_Max,
          bool has_bias,
          bool has_active>
void major_loop(int N,
                const float* __restrict__ scale,
                const float* __restrict__ bias,
                const lite_api::ActivationType& act,
                float six,
                float alpha,
                int8_t* __restrict__ ptr_y,
                const int8_t* __restrict__ ptr_x,
                const int8_t* __restrict__ ptr_A,
                const int dynamic_tile_j_Max,
                const int row_remain,
                int tile_i_Max) {
  static_assert(TILE_A_WIDTH == 16, "MUST be 16");
  static_assert(TILE_A_HEIGHT / 2 * 2 == TILE_A_HEIGHT, "MUST be 2*k");

  if (const_tile_j_Max == -1) {
    assert(dynamic_tile_j_Max > 0);
  }
  using dtype = int8_t;
  for (int tile_i = 0; tile_i < tile_i_Max; ++tile_i) {
    int tile_start_row = tile_i * TILE_A_HEIGHT;

    dtype* __restrict__ out_ptr = ptr_y + tile_start_row;
    const float* __restrict__ scale_ptr = scale + tile_start_row;
    const int8_t* __restrict__ tile_b_data = ptr_x;

    const int8_t* __restrict__ tile_A_data = ptr_A + (N * tile_start_row);
    const int8_t* __restrict__ tile_A_row_data[TILE_A_HEIGHT];
    for (int i = 0; i < TILE_A_HEIGHT; ++i) {
      tile_A_row_data[i] = tile_A_data + i * N;
    }
    const float* __restrict__ bias_ptr =
        has_bias ? bias + tile_start_row : nullptr;

    int8x16_t col0;
    int8x16_t tile_row[TILE_A_HEIGHT];
    int32x4_t q[TILE_A_HEIGHT];
    for (int i = 0; i < TILE_A_HEIGHT; ++i) {
      q[i] = vmovq_n_s32(0);
    }
    if (const_tile_j_Max != 0) {
      col0 = vld1q_s8(tile_b_data);
      tile_b_data += TILE_A_WIDTH;

      for (int i = 0; i < TILE_A_HEIGHT; ++i) {
        tile_row[i] = vld1q_s8(tile_A_row_data[i]);
        tile_A_row_data[i] += TILE_A_WIDTH;
      }
      if (const_tile_j_Max == -1) {
        for (int tile_j = 0; tile_j < (dynamic_tile_j_Max - 1); ++tile_j) {
          loop_mul_add_load<TILE_A_WIDTH, TILE_A_HEIGHT>::run(
              tile_A_row_data, col0, tile_row, q);

          col0 = vld1q_s8(tile_b_data);
          tile_b_data += TILE_A_WIDTH;
        }
      } else {
        for (int tile_j = 0; tile_j < (const_tile_j_Max - 1); ++tile_j) {
          loop_mul_add_load<TILE_A_WIDTH, TILE_A_HEIGHT>::run(
              tile_A_row_data, col0, tile_row, q);

          col0 = vld1q_s8(tile_b_data);
          tile_b_data += TILE_A_WIDTH;
        }
      }
      loop_mul_add<TILE_A_HEIGHT>(col0, tile_row, q);
    }
    int this_run_row_remain = row_remain;
    if (this_run_row_remain >= 8) {
      int8x8_t col0 = vld1_s8(tile_b_data);
      int8x8_t tile_row[TILE_A_HEIGHT];
      tile_b_data += 8;

      for (int i = 0; i < TILE_A_HEIGHT; ++i) {
        tile_row[i] = vld1_s8(tile_A_row_data[i]);
        tile_A_row_data[i] += 8;

        int16x8_t mul = vmull_s8(col0, tile_row[i]);
        q[i] = vpadalq_s16(q[i], mul);
      }
      this_run_row_remain -= 8;
    }

    int32x2_t ans[TILE_A_HEIGHT / 2];
    for (int row_id0 = 0, row_id1 = 1, i = 0; i < TILE_A_HEIGHT / 2; ++i) {
      ans[i] = vpadd_s32(
          vpadd_s32(vget_low_s32(q[row_id0]), vget_high_s32(q[row_id0])),
          vpadd_s32(vget_low_s32(q[row_id1]), vget_high_s32(q[row_id1])));
      row_id0 += 2;
      row_id1 += 2;
    }

    for (int i = 0; i < this_run_row_remain; ++i) {
      int32_t v = tile_b_data[i];

      for (int row_id0 = 0, row_id1 = 1, j = 0; j < TILE_A_HEIGHT / 2; ++j) {
        ans[j][0] += v * tile_A_row_data[row_id0][i];
        row_id0 += 2;
        ans[j][1] += v * tile_A_row_data[row_id1][i];
        row_id1 += 2;
      }
    }
    if (TILE_A_HEIGHT / 4 > 0) {
      int32x4_t out[TILE_A_HEIGHT / 4];
      for (int row_id0 = 0, row_id1 = 1, i = 0; i < TILE_A_HEIGHT / 4; ++i) {
        out[i] = vcombine_s32(ans[row_id0], ans[row_id1]);
        row_id0 += 2;
        row_id1 += 2;
      }
      neon_write_gemv_int8_out<TILE_A_HEIGHT / 4, has_bias, has_active>(
          &out[0], out_ptr, scale_ptr, bias_ptr, act, six, alpha);
    }

    constexpr int HEIGHT_4x = TILE_A_HEIGHT / 4 * 4;
    if (HEIGHT_4x != TILE_A_HEIGHT) {
      neon_write_gemv_int8_out<1, has_bias, has_active>(
          &ans[TILE_A_HEIGHT / 2 - 1],
          out_ptr + HEIGHT_4x,
          scale_ptr + HEIGHT_4x,
          bias_ptr + HEIGHT_4x,
          act,
          six,
          alpha);
    }
  }
}

template <int TILE_A_WIDTH, int TILE_A_HEIGHT, bool HAS_BIAS, bool HAS_ACTIVE>
inline void hard_switch(int N,
                        const float* __restrict__ scale,
                        const float* __restrict__ bias,
                        const lite_api::ActivationType& act,
                        float six,
                        float alpha,
                        signed char* __restrict__ ptr_y,
                        const int8_t* __restrict__ ptr_x,
                        const int8_t* __restrict__ ptr_A,
                        const int tile_j_Max,
                        const int row_remain,
                        int tile_i_Max) {
#define INPUT_ARGS                                                  \
  N, scale, bias, act, six, alpha, ptr_y, ptr_x, ptr_A, tile_j_Max, \
      row_remain, tile_i_Max

  switch (tile_j_Max) {
    case 0:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, 0, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
      break;
    case 1:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, 1, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
      break;
    case 2:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, 2, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
      break;
    case 3:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, 3, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
      break;
    case 4:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, 4, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
      break;
    case 5:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, 5, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
      break;
    default:
      major_loop<TILE_A_WIDTH, TILE_A_HEIGHT, -1, HAS_BIAS, HAS_ACTIVE>(
          INPUT_ARGS);
  }
#undef INPUT_ARGS
}

template <int TILE_A_WIDTH, int TILE_A_HEIGHT>
inline void select_major_loop(int N,
                              const float* __restrict__ scale,
                              bool is_bias,
                              const float* __restrict__ bias,
                              bool is_act,
                              const lite_api::ActivationType& act,
                              float six,
                              float alpha,
                              signed char* __restrict__ ptr_y,
                              const int8_t* __restrict__ ptr_x,
                              const int8_t* __restrict__ ptr_A,
                              const int tile_j_Max,
                              const int row_remain,
                              int tile_i_Max) {
#define INPUT_ARGS                                                  \
  N, scale, bias, act, six, alpha, ptr_y, ptr_x, ptr_A, tile_j_Max, \
      row_remain, tile_i_Max
  if (is_bias) {
    constexpr bool has_bias = true;
    if (is_act) {
      constexpr bool has_active = true;
      hard_switch<TILE_A_WIDTH, TILE_A_HEIGHT, has_bias, has_active>(
          INPUT_ARGS);
    } else {
      constexpr bool has_active = false;
      hard_switch<TILE_A_WIDTH, TILE_A_HEIGHT, has_bias, has_active>(
          INPUT_ARGS);
    }
  } else {
    constexpr bool has_bias = false;
    if (is_act) {
      constexpr bool has_active = true;
      hard_switch<TILE_A_WIDTH, TILE_A_HEIGHT, has_bias, has_active>(
          INPUT_ARGS);
    } else {
      constexpr bool has_active = false;
      hard_switch<TILE_A_WIDTH, TILE_A_HEIGHT, has_bias, has_active>(
          INPUT_ARGS);
    }
  }
#undef INPUT_ARGS
}
bool gemv_int8_oth_intrinsic(const int8_t* __restrict__ A,
                             const int8_t* __restrict__ x,
                             int8_t* __restrict__ y,
                             bool transA,
                             int M,
                             int N,
                             const float* __restrict__ scale,
                             bool is_bias,
                             const float* __restrict__ bias,
                             bool flag_act,
                             lite_api::ActivationType act,
                             float six,
                             float alpha) {
  constexpr int TILE_A_WIDTH = 16;
  constexpr int TILE_A_HEIGHT = 4;

  if (transA) {
    LOG(ERROR) << "ERROR: sgemv, transA is not supported now";
    return false;
  }
  using dtype = int8_t;
  dtype* __restrict__ ptr_y = y;
  const int8_t* __restrict__ ptr_x = x;
  const int8_t* __restrict__ ptr_A = A;

  const int tile_j_Max = N / TILE_A_WIDTH;
  const int row_remain = N % TILE_A_WIDTH;
  int tile_i_Max = M / TILE_A_HEIGHT;

  select_major_loop<TILE_A_WIDTH, TILE_A_HEIGHT>(N,
                                                 scale,
                                                 is_bias,
                                                 bias,
                                                 flag_act,
                                                 act,
                                                 six,
                                                 alpha,
                                                 ptr_y,
                                                 ptr_x,
                                                 ptr_A,
                                                 tile_j_Max,
                                                 row_remain,
                                                 tile_i_Max);

  int row_idx = tile_i_Max * TILE_A_HEIGHT;
  if ((M - row_idx) >= 2) {
    select_major_loop<TILE_A_WIDTH, 2>(N,
                                       scale + row_idx,
                                       is_bias,
                                       bias + row_idx,
                                       flag_act,
                                       act,
                                       six,
                                       alpha,
                                       ptr_y + row_idx,
                                       ptr_x,
                                       ptr_A + row_idx * N,
                                       tile_j_Max,
                                       row_remain,
                                       1);
    row_idx += 2;
  }

  if (row_idx != M) {
    dtype* __restrict__ out_ptr = ptr_y + row_idx;
    const float* __restrict__ scale_ptr = scale + row_idx;
    const int8_t* __restrict__ tile_b_data = ptr_x;

    const int8_t* __restrict__ row_A_data = ptr_A + (N * row_idx);
    const float* __restrict__ bias_ptr = is_bias ? bias + row_idx : nullptr;

    int32x4_t q = vmovq_n_s32(0);

    for (int tile_j = 0; tile_j < tile_j_Max; ++tile_j) {
      int8x16_t col0 = vld1q_s8(tile_b_data);
      tile_b_data += TILE_A_WIDTH;

      int8x16_t one_row;
      one_row = vld1q_s8(row_A_data);
      row_A_data += TILE_A_WIDTH;

      int16x8_t mul_lo = vmull_s8(vget_low_s8(col0), vget_low_s8(one_row));
      int16x8_t mul_hi = vmull_s8(vget_high_s8(col0), vget_high_s8(one_row));

      q = vpadalq_s16(q, mul_lo + mul_hi);
    }
    int32x2_t ans_tmp = vpadd_s32(vget_low_s32(q), vget_high_s32(q));
    int32_t ans = ans_tmp[0] + ans_tmp[1];
    for (int i = 0; i < row_remain; ++i) {
      ans += tile_b_data[i] * row_A_data[i];
    }

    if (is_bias) {
      if (flag_act) {
        write_gemv_int8_out<1, true, true>(
            &ans, out_ptr, scale_ptr, bias_ptr, act, six, alpha);
      } else {
        write_gemv_int8_out<1, true, false>(
            &ans, out_ptr, scale_ptr, bias_ptr, act, six, alpha);
      }
    } else {
      if (flag_act) {
        write_gemv_int8_out<1, false, true>(
            &ans, out_ptr, scale_ptr, bias_ptr, act, six, alpha);
      } else {
        write_gemv_int8_out<1, false, false>(
            &ans, out_ptr, scale_ptr, bias_ptr, act, six, alpha);
      }
    }
  }
  return true;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
