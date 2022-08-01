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

#include <limits.h>
#include <math.h>
#include "lite/tests/cv/anakin/cv_utils.h"

void resize_one_channel(
    const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out);
void resize_one_channel_uv(
    const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out);
void nv21_resize(const uint8_t* src,
                 uint8_t* dst,
                 int w_in,
                 int h_in,
                 int w_out,
                 int h_out) {
  if (w_out == w_in && h_out == h_in) {
    printf("nv21_resize equal \n");
    memcpy(dst, src, sizeof(uint8_t) * w_in * static_cast<int>(1.5 * h_in));
    return;
  }
  int y_h = h_in;
  int uv_h = h_in / 2;
  const uint8_t* y_ptr = src;
  const uint8_t* uv_ptr = src + y_h * w_in;
  // out
  int dst_y_h = h_out;
  int dst_uv_h = h_out / 2;
  uint8_t* dst_ptr = dst + dst_y_h * w_out;

  resize_one_channel(y_ptr, w_in, y_h, dst, w_out, dst_y_h);
  resize_one_channel_uv(uv_ptr, w_in, uv_h, dst_ptr, w_out, dst_uv_h);
}

void resize_one_channel(const uint8_t* src,
                        int w_in,
                        int h_in,
                        uint8_t* dst,
                        int w_out,
                        int h_out) {
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;

  double scale_x = static_cast<double>(w_in) / w_out;
  double scale_y = static_cast<double>(h_in) / h_out;

  int* buf = new int[w_out * 2 + h_out * 2];

  int* xofs = buf;          // new int[w];
  int* yofs = buf + w_out;  // new int[h];

  int16_t* ialpha =
      reinterpret_cast<int16_t*>(buf + w_out + h_out);  // new short[w * 2];
  int16_t* ibeta =
      reinterpret_cast<int16_t*>(buf + w_out * 2 + h_out);  // new short[h * 2];

  float fx = 0.f;
  float fy = 0.f;
  int sx = 0;
  int sy = 0;

#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  for (int dx = 0; dx < w_out; dx++) {
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
  for (int dy = 0; dy < h_out; dy++) {
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
  // loop body
  int16_t* rowsbuf0 = new int16_t[w_out + 1];
  int16_t* rowsbuf1 = new int16_t[w_out + 1];
  int16_t* rows0 = rowsbuf0;
  int16_t* rows1 = rowsbuf1;

  int prev_sy1 = -1;
  for (int dy = 0; dy < h_out; dy++) {
    int sy = yofs[dy];

    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + w_in * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + w_in * (sy);
      const uint8_t* S1 = src + w_in * (sy + 1);

      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out; dx++) {
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

    // vresize
    int16_t b0 = ibeta[0];
    int16_t b1 = ibeta[1];

    int16_t* rows0p = rows0;
    int16_t* rows1p = rows1;
    uint8_t* dp_ptr = dst + w_out * (dy);

    int cnt = w_out >> 3;
    int remain = w_out - (cnt << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);

// #pragma omp parallel for

#if 1  // __aarch64__
    for (cnt = w_out >> 3; cnt > 0; cnt--) {
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
#else
#pragma omp parallel for  // TODO(chenjiaoAngel): asm is not right , 1 only use
    // rows0p; 2 can
    // not parallel because address(rows0p) depend
    if (cnt > 0) {
      asm volatile(
          "mov        r4, #2          \n"
          "vdup.s32   q12, r4         \n"
          "0:                         \n"
          "pld        [%[rows0p], #128]      \n"
          "pld        [%[rows1p], #128]      \n"
          "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
          "vld1.s16   {d6-d7}, [%[rows0p]]!\n"
          "pld        [%[rows0p], #128]      \n"
          "pld        [%[rows1p], #128]      \n"
          "vmull.s16  q0, d2, %[_b0]     \n"
          "vmull.s16  q1, d3, %[_b0]     \n"
          "vmull.s16  q2, d6, %[_b1]     \n"
          "vmull.s16  q3, d7, %[_b1]     \n"

          "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
          "vld1.s16   {d6-d7}, [%[rows0p]]!\n"

          "vorr.s32   q10, q12, q12   \n"
          "vorr.s32   q11, q12, q12   \n"
          "vsra.s32   q10, q0, #16    \n"
          "vsra.s32   q11, q1, #16    \n"
          "vsra.s32   q10, q2, #16    \n"
          "vsra.s32   q11, q3, #16    \n"

          "vmull.s16  q0, d2, %[_b0]     \n"
          "vmull.s16  q1, d3, %[_b0]     \n"
          "vmull.s16  q2, d6, %[_b1]     \n"
          "vmull.s16  q3, d7, %[_b1]     \n"

          "vsra.s32   q10, q0, #16    \n"
          "vsra.s32   q11, q1, #16    \n"
          "vsra.s32   q10, q2, #16    \n"
          "vsra.s32   q11, q3, #16    \n"

          "vshrn.s32  d20, q10, #2    \n"
          "vshrn.s32  d21, q11, #2    \n"
          "vqmovun.s16 d20, q10        \n"
          "vst1.8     {d20}, [%[dp]]!    \n"
          "subs       %[cnt], #1          \n"
          "bne        0b              \n"
          "sub        %[rows0p], #16         \n"
          "sub        %[rows1p], #16         \n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [_b0] "+w"(_b0),
            [_b1] "+w"(_b1),
            [cnt] "+r"(cnt),
            [dp] "+r"(dp_ptr)
          :
          : "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12");
    }
#endif                    // __aarch64__
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

  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}

void resize_one_channel_uv(const uint8_t* src,
                           int w_in,
                           int h_in,
                           uint8_t* dst,
                           int w_out,
                           int h_out) {
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;

  double scale_x = static_cast<double>(w_in) / w_out;
  double scale_y = static_cast<double>(h_in) / h_out;

  int* buf = new int[w_out * 2 + h_out * 2];

  int* xofs = buf;          // new int[w];
  int* yofs = buf + w_out;  // new int[h];

  int16_t* ialpha =
      reinterpret_cast<int16_t*>(buf + w_out + h_out);  // new int16_t[w * 2];
  int16_t* ibeta = reinterpret_cast<int16_t*>(buf + w_out * 2 +
                                              h_out);  // new int16_t[h * 2];

  float fx = 0.f;
  float fy = 0.f;
  int sx = 0.f;
  int sy = 0.f;

#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  for (int dx = 0; dx < w_out / 2; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= w_in / 2 - 1) {
      sx = w_in / 2 - 2;
      fx = 1.f;
    }

    xofs[dx] = sx;

    float a0 = (1.f - fx) * resize_coef_scale;
    float a1 = fx * resize_coef_scale;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
  for (int dy = 0; dy < h_out; dy++) {
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
  // loop body
  int16_t* rowsbuf0 = new int16_t[w_out + 1];
  int16_t* rowsbuf1 = new int16_t[w_out + 1];
  int16_t* rows0 = rowsbuf0;
  int16_t* rows1 = rowsbuf1;

  int prev_sy1 = -1;
  for (int dy = 0; dy < h_out; dy++) {
    int sy = yofs[dy];
    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + w_in * (sy + 1);

      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out / 2; dx++) {
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
      const uint8_t* S0 = src + w_in * (sy);
      const uint8_t* S1 = src + w_in * (sy + 1);

      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out / 2; dx++) {
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

    // vresize
    int16_t b0 = ibeta[0];
    int16_t b1 = ibeta[1];

    int16_t* rows0p = rows0;
    int16_t* rows1p = rows1;
    uint8_t* dp_ptr = dst + w_out * (dy);

    int cnt = w_out >> 3;
    int remain = w_out - (cnt << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);

    for (cnt = w_out >> 3; cnt > 0; cnt--) {
      int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
      int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
      int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
      int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

      int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
      int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
      int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
      int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

      int32x4_t _acc = _v2;
      _acc = vsraq_n_s32(
          _acc, _rows0p_sr4_mb0, 16);  // _acc >> 16 + _rows0p_sr4_mb0 >> 16
      _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

      int32x4_t _acc_1 = _v2;
      _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
      _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

      int16x4_t _acc16 = vshrn_n_s32(_acc, 2);  // _acc >> 2
      int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

      uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

      vst1_u8(dp_ptr, _dout);

      dp_ptr += 8;
      rows0p += 8;
      rows1p += 8;
    }
    for (; remain; --remain) {
      // D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
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
