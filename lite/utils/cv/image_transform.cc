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

#include "lite/utils/cv/image_transform.h"
#include <arm_neon.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include "lite/utils/cv/image_flip.h"
#include "lite/utils/cv/image_rotate.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {

// compute xofs, yofs, alpha, beta
void compute_xy(int wout,
                int srch,
                double scale_x,
                double scale_y,
                int dstw,
                int dsth,
                int* xofs,
                int* yofs,
                int16_t* ialpha,
                int16_t* ibeta) {
  float fy = 0.f;
  float fx = 0.f;
  int sy = 0;
  int sx = 0;
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;
#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  for (int dx = 0; dx < wout; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= wout - 1) {
      sx = wout - 2;
      fx = 1.f;
    }

    xofs[dx] = sx;

    float a0 = (1.f - fx) * resize_coef_scale;
    float a1 = fx * resize_coef_scale;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
  for (int dy = 0; dy < dsth; dy++) {
    fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    sy = floor(fy);
    fy -= sy;

    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= srch - 1) {
      sy = srch - 2;
      fy = 1.f;
    }

    yofs[dy] = sy;

    float b0 = (1.f - fy) * resize_coef_scale;
    float b1 = fy * resize_coef_scale;

    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }
#undef SATURATE_CAST_SHORT
}

// gray
void resize_hwc1(const uint8_t* src,
                 const int16_t* ialpha,
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

// uv
void resize_hwc2(const uint8_t* src,
                 const int16_t* ialpha,
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
      for (int dx = 0; dx < dstw / 2; dx++) {
        int sx = xofs[dx] * 2;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 2;
        rows1p[tmp] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + srcw * (sy);
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw / 2; dx++) {
        int sx = xofs[dx] * 2;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 2;
        rows0p[tmp] = (S0p[0] * a0 + S0p[2] * a1) >> 4;
        rows1p[tmp] = (S1p[0] * a0 + S1p[2] * a1) >> 4;

        rows0p[tmp + 1] = (S0p[1] * a0 + S0p[3] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;

        ialphap += 2;
      }
    }
    prev_sy1 = sy + 1;
  }
}

// bgr rgb
void resize_hwc3(const uint8_t* src,
                 const int16_t* ialpha,
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
      for (int dx = 0; dx < dstw / 3; dx++) {
        int sx = xofs[dx] * 3;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 3;
        rows1p[tmp] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
        rows1p[tmp + 2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + srcw * (sy);
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw / 3; dx++) {
        int sx = xofs[dx] * 3;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 3;
        rows0p[tmp] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
        rows1p[tmp] = (S1p[0] * a0 + S1p[3] * a1) >> 4;

        rows0p[tmp + 1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;

        rows0p[tmp + 2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
        rows1p[tmp + 2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;

        ialphap += 2;
      }
    }
    prev_sy1 = sy + 1;
  }
}
// bgra rgba
void resize_hwc4(const uint8_t* src,
                 const int16_t* ialpha,
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
      for (int dx = 0; dx < dstw / 4; dx++) {
        int sx = xofs[dx] * 4;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 4;
        rows1p[tmp] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
        rows1p[tmp + 2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
        rows1p[tmp + 3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + srcw * (sy);
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < dstw / 4; dx++) {
        int sx = xofs[dx] * 4;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 4;
        rows0p[tmp] = (S0p[0] * a0 + S0p[4] * a1) >> 4;
        rows1p[tmp] = (S1p[0] * a0 + S1p[4] * a1) >> 4;

        rows0p[tmp + 1] = (S0p[1] * a0 + S0p[5] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;

        rows0p[tmp + 2] = (S0p[2] * a0 + S0p[6] * a1) >> 4;
        rows1p[tmp + 2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;

        rows0p[tmp + 3] = (S0p[3] * a0 + S0p[7] * a1) >> 4;
        rows1p[tmp + 3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;

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
                            int dsth) {
  if (srcw == dstw && srch == dsth) {
    int size = srcw * srch;
    if (srcFormat == NV12 || srcFormat == NV21) {
      size = srcw * (static_cast<double>(1.5 * srch));
    } else if (srcFormat == BGR || srcFormat == RGB) {
      size = 3 * size;
    } else if (srcFormat == BGRA || srcFormat == RGBA) {
      size = 4 * size;
    }
    memcpy(dst, src, sizeof(uint8_t) * size);
    return;
  }
  double scale_x = static_cast<double>(srcw / dstw);
  double scale_y = static_cast<double>(srch / dsth);

  int* buf = new int[dstw * 2 + dsth * 2];

  int* xofs = buf;
  int* yofs = buf + dstw;
  int16_t* ialpha = reinterpret_cast<int16_t*>(buf + dstw + dsth);
  int16_t* ibeta = reinterpret_cast<int16_t*>(buf + 2 * dstw + dsth);

  // template varible
  int size = dsth * (dstw + 1);
  int16_t* rowsbuf0 = new int16_t[size];
  int16_t* rowsbuf1 = new int16_t[size];

  if (srcFormat == GRAY) {
    compute_xy(
        srcw, srch, scale_x, scale_y, dstw, dsth, xofs, yofs, ialpha, ibeta);
    // hwc1
    resize_hwc1(
        src, ialpha, xofs, yofs, rowsbuf0, rowsbuf1, srcw, srch, dstw, dsth);
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    int hout = static_cast<int>(0.5 * dsth);
    int size2 = size + hout * (dstw + 1);
    int16_t* new_rowsbuf0 = new int16_t[size2];
    int16_t* new_rowsbuf1 = new int16_t[size2];
    rowsbuf0 = new_rowsbuf0;
    rowsbuf1 = new_rowsbuf1;
    int wout = srcw;  // y
    compute_xy(
        wout, srch, scale_x, scale_y, dstw, dsth, xofs, yofs, ialpha, ibeta);
    // hwc1
    resize_hwc1(
        src, ialpha, xofs, yofs, rowsbuf0, rowsbuf1, srcw, srch, dstw, dsth);
    // uv todo
    wout = wout / 2;
    compute_xy(
        wout, srch, scale_x, scale_y, dstw, dsth, xofs, yofs, ialpha, ibeta);
    // hwc2
    resize_hwc2(src,
                ialpha,
                xofs,
                yofs,
                rowsbuf0 + size,
                rowsbuf1 + size,
                srcw,
                srch,
                dstw,
                dsth);
    dsth += hout;
  } else if (srcFormat == BGR || srcFormat == RGB) {
    int wout = srcw / 3;
    compute_xy(
        wout, srch, scale_x, scale_y, dstw, dsth, xofs, yofs, ialpha, ibeta);
    // hwc1
    resize_hwc3(
        src, ialpha, xofs, yofs, rowsbuf0, rowsbuf1, srcw, srch, dstw, dsth);

  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    int wout = srcw / 4;
    compute_xy(
        wout, srch, scale_x, scale_y, dstw, dsth, xofs, yofs, ialpha, ibeta);
    // hwc1
    resize_hwc4(
        src, ialpha, xofs, yofs, rowsbuf0, rowsbuf1, srcw, srch, dstw, dsth);
  }

  int cnt = dstw >> 3;
  int remain = dstw % 8;
  int32x4_t _v2 = vdupq_n_s32(2);
#pragma omp parallel for
  for (int dy = 0; dy < dsth; dy++) {
    int16_t b0 = ibeta[0];
    int16_t b1 = ibeta[0];
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    uint8_t* dp_ptr = dst + dy * dstw;
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
    for (int i = 0; i < remain; i++) {
      //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >>
      //             INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
    ibeta += 2;
  }
  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}
void ImageTransform::rotate(const uint8_t* src,
                            uint8_t* dst,
                            ImageFormat srcFormat,
                            int srcw,
                            int srch,
                            float degree) {
  if (srcFormat == GRAY) {
    rotate_hwc1(src, dst, srcw, srch, degree);
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    rotate_hwc2(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    rotate_hwc3(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    rotate_hwc4(src, dst, srcw, srch, degree);
  }
}
void ImageTransform::flip(const uint8_t* src,
                          uint8_t* dst,
                          ImageFormat srcFormat,
                          int srcw,
                          int srch,
                          FlipParm flip_param) {
  if (srcFormat == GRAY) {
    flip_hwc1(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    flip_hwc2(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    flip_hwc3(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    flip_hwc4(src, dst, srcw, srch, flip_param);
  }
}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
