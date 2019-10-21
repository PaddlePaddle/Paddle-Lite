// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "lite/utils/cv/image_transform.h"
#include <arm_neon.h>
#include <limits.h>
#include <math.h>
#include <algorithm>
namespace paddle {
namespace lite {
namespace utils {
namespace cv {

void resize_hwc(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int dstw, int dsth) {
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;

  double scale_x = static_cast<double>(srcw / dstw);
  double scale_y = static_cast<double>(srch / dsth);

  int* buf = new int[dsth * 2];

  int* yofs = buf;
  int16_t* ibeta = reinterpret_cast<int16_t*>(buf + dsth);

  float fy = 0.f;
  int sx = 0;
  int sy = 0;
#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  for (int dy = 0; dy < dsth; dy++) {
    fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    sy = floor(fy);
    fy -= sy;

    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= h_in - 1) {
      sy = h_in - 2;
      fy = 1.f;
    }

    yofs[dy] = sy;

    float b0 = (1.f - fy) * resize_coef_scale;
    float b1 = fy * resize_coef_scale;

    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }
#undef SATURATE_CAST_SHORT
  // template varible
  int size = dsth * (dstw + 1);
  int16_t* rowsbuf0 = new int16_t[size];
  int16_t* rowsbuf1 = new int16_t[size];
  // hwc1
  resize_hwc1(src, yofs, rowsbuf0, rowsbuf1, scale_x, srcw, srch, dstw, dsth);
#pragma omp parallel for
  for (int dy = 0; dy < dsth; dy++) {
    int16_t* rows0p = rowsbuf0 + dy * (dstw + 1);
    int16_t* rows1p = rowsbuf1 + dy * (dstw + 1);
#ifdef __aarch64__
    /*
    for (; cnt > 0; cnt--){
        int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
        int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
        int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
        int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

        int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
        int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
        int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
        int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

        int32x4_t _acc = _v2;
        _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
        _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

        int32x4_t _acc_1 = _v2;
        _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
        _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

        int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
        int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

        uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

        vst1_u8(dp_ptr, _dout);

        dp_ptr += 8;
        rows0p += 8;
        rows1p += 8;
    }
    */
    if (cnt > 0) {
      asm volatile(
          "1: \n"
          "ld1 {v0.8h}, [%[rows0p]], #16 \n"
          "ld1 {v1.8h}, [%[rows1p]], #16 \n"
          "orr v6.16b, %w[_v2].16b, %w[_v2].16b \n"
          "orr v7.16b, %w[_v2].16b, %w[_v2].16b \n"
          "smull v2.4s, v0.4h, %w[_b0].4h \n"
          "smull2 v4.4s, v0.8h, %w[_b0].8h \n"
          "smull v3.4s, v1.4h, %w[_b1].4h \n"
          "smull2 v5.4s, v1.8h, %w[_b1].8h \n"

          "ssra v6.4s, v2.4s, #16 \n"
          "ssra v7.4s, v4.4s, #16 \n"
          "ssra v6.4s, v3.4s, #16 \n"
          "ssra v7.4s, v5.4s, #16 \n"

          "shrn v0.4h, v6.4s, #2 \n"
          "shrn2 v0.8h, v7.4s, #2 \n"
          "subs %w[cnt], %w[cnt], #1 \n"
          "sqxtun v1.8b, v0.8h \n"
          "st1 {v1.8b}, [%[dp]], #8 \n"
          "bne 1b \n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [cnt] "+r"(cnt),
            [dp] "+r"(dp_ptr)
          : [_b0] "w"(_b0), [_b1] "w"(_b1), [_v2] "w"(_v2)
          : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }

#else
    if (cnt > 0) {
      asm volatile(
          "mov        r4, #2          \n"
          "vdup.s32   q12, r4         \n"
          "0:                         \n"
          "pld        [%[rows0p], #128]      \n"
          "pld        [%[rows1p], #128]      \n"
          "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
          "vld1.s16   {d6-d7}, [%[rows1p]]!\n"
          "pld        [%[rows0p], #128]      \n"
          "pld        [%[rows1p], #128]      \n"
          "vmull.s16  q0, d2, %[_b0]     \n"
          "vmull.s16  q1, d3, %[_b0]     \n"
          "vmull.s16  q2, d6, %[_b1]     \n"
          "vmull.s16  q3, d7, %[_b1]     \n"

          "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
          "vld1.s16   {d6-d7}, [%[rows1p]]!\n"

          "vorr.s32   q10, q12, q12   \n"
          "vorr.s32   q11, q12, q12   \n"
          "vsra.s32   q10, q0, #16    \n"
          "vsra.s32   q11, q1, #16    \n"
          "vsra.s32   q10, q2, #16    \n"
          "vsra.s32   q11, q3, #16    \n"

          "vshrn.s32  d20, q10, #2    \n"
          "vshrn.s32  d21, q11, #2    \n"
          "subs       %[cnt], #1          \n"
          "vqmovun.s16 d20, q10        \n"
          "vst1.8     {d20}, [%[dp]]!    \n"
          "bne        0b              \n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [cnt] "+r"(cnt),
            [dp] "+r"(dp_ptr)
          : [_b0] "w"(_b0), [_b1] "w"(_b1)
          : "cc",
            "memory",
            "r4",
            "q0",
            "q1",
            "q2",
            "q3",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12");
    }
#endif  // __aarch64__
    for (; remain; --remain) {
      //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >>
      //             INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
    ibeta += 2;
  }
}
// gray
void resize_hwc1(const uint8_t* src,
                 const int* yofs,
                 int16_t* rowsbuf0,
                 int16_t* rowsbuf1,
                 double scale_x,
                 int srcw,
                 int srch,
                 int dstw,
                 int dsth) {
  int* buf = new int[dstw * 2];
  int* xofs = buf;
  int16_t* ialpha = reinterpret_cast<int16_t*>(buf + dstw);
  float fx = 0.f;
  int sx = 0;
#define SATURATE_CAST_SHORT(X)                                         \
  (int16_t)::std::min(                                                 \
      ::std::max(reinterpret_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), \
                 SHRT_MIN),                                            \
      SHRT_MAX);
  for (int dx = 0; dx < dstw; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= w_in - 1) {
      sx = w_in - 2;
      fx = 1.f;
    }

    xofs[dx] = sx;

    float a0 = (1.f - fx) * resize_coef_scale;
    float a1 = fx * resize_coef_scale;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
#undef SATURATE_CAST_SHORT
  int prev_sy1 = -1;
  int wout = dstw + 1;
  for (int dy = 0; dy < dsth; dy++) {
    int16_t* rows0 = rowsbuf0 + static_cast<int16_t>(dy * wout);
    int16_t* rows1 = rowsbuf1 + static_cast<int16_t>(dy * wout);

    int sy = yofs[dy];
    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + srcw * (sy);
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    }
    prev_sy1 = sy + 1;
  }
}

// gray
void resize_hwc2(const uint8_t* src,
                 const int* xofs,
                 const int* yofs,
                 int16_t* rowsbuf0,
                 int16_t* rowsbuf1,
                 int srcw,
                 int srch,
                 int dstw,
                 int dsth) {
  int prev_sy1 = -1;
  int wout = dstw + 1;
  for (int dy = 0; dy < dsth; dy++) {
    int16_t* rows0 = rowsbuf0 + static_cast<int16_t>(dy * wout);
    int16_t* rows1 = rowsbuf1 + dy * wout;

    int sy = yofs[dy];
    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw / 2; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + srcw * (sy);
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    }
    prev_sy1 = sy + 1;
  }
}

void ImageTransform::resize(const uint8_t* src,
                            uint8_t* dst,
                            ImageFormat srcFormat,
                            int srcw,
                            int srch,
                            int dstw,
                            int dsth) {}
void ImageTransform::rotate(const uint8_t* src,
                            uint8_t* dst,
                            ImageFormat srcFormat,
                            int srcw,
                            int srch,
                            float degree) {}
void ImageTransform::flip(const uint8_t* src,
                          uint8_t* dst,
                          ImageFormat srcFormat,
                          int srcw,
                          int srch,
                          FlipParm flip_param) {}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
