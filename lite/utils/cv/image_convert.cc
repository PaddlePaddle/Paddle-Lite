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

#include "lite/utils/cv/image_convert.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include "lite/core/parallel_defines.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
void nv_to_bgr(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int x_num, int y_num);

void nv_to_bgra(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, int x_num, int y_num);

void nv21_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch);
void nv21_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch);
void nv12_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch);
void nv12_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgra rgba to gray
void hwc4_to_hwc1(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgr rgb to gray
void hwc3_to_hwc1(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// gray to bgr rgb
void hwc1_to_hwc3(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// gray to bgra rgba
void hwc1_to_hwc4(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgr to bgra or rgb to rgba
void hwc3_to_hwc4(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgra to bgr or rgba to rgb
void hwc4_to_hwc3(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgr to rgb or rgb to bgr
void hwc3_trans(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgra to rgba or rgba to bgra
void hwc4_trans(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgra to rgb or rgba to bgr
void hwc4_trans_hwc3(const uint8_t* src, uint8_t* dst, int srcw, int srch);
// bgr to rgba or rgb to bgra
void hwc3_trans_hwc4(const uint8_t* src, uint8_t* dst, int srcw, int srch);

/*
  * image color convert
  * support NV12/NV21_to_BGR(RGB), NV12/NV21_to_BGRA(RGBA),
  * BGR(RGB)and BGRA(RGBA) transform,
  * BGR(RGB)and RGB(BGR) transform,
  * BGR(RGB)and RGBA(BGRA) transform,
  * BGR(RGB)and GRAY transform,
  * param src: input image data
  * param dst: output image data
  * param srcFormat: input image image format support: GRAY, NV12(NV21),
 * BGR(RGB) and BGRA(RGBA)
  * param dstFormat: output image image format, support GRAY, BGR(RGB) and
 * BGRA(RGBA)
*/
void ImageConvert::choose(const uint8_t* src,
                          uint8_t* dst,
                          ImageFormat srcFormat,
                          ImageFormat dstFormat,
                          int srcw,
                          int srch) {
  if (srcFormat == dstFormat) {
    // copy
    int size = srcw * srch;
    if (srcFormat == NV12 || srcFormat == NV21) {
      size = srcw * (ceil(1.5 * srch));
    } else if (srcFormat == BGR || srcFormat == RGB) {
      size = 3 * srcw * srch;
    } else if (srcFormat == BGRA || srcFormat == RGBA) {
      size = 4 * srcw * srch;
    }
    memcpy(dst, src, sizeof(uint8_t) * size);
    return;
  } else {
    if (srcFormat == NV12 && (dstFormat == BGR || dstFormat == RGB)) {
      impl_ = nv12_to_bgr;
    } else if (srcFormat == NV21 && (dstFormat == BGR || dstFormat == RGB)) {
      impl_ = nv21_to_bgr;
    } else if (srcFormat == NV12 && (dstFormat == BGRA || dstFormat == RGBA)) {
      impl_ = nv12_to_bgra;
    } else if (srcFormat == NV21 && (dstFormat == BGRA || dstFormat == RGBA)) {
      impl_ = nv21_to_bgra;
    } else if ((srcFormat == RGBA && dstFormat == RGB) ||
               (srcFormat == BGRA && dstFormat == BGR)) {
      impl_ = hwc4_to_hwc3;
    } else if ((srcFormat == RGB && dstFormat == RGBA) ||
               (srcFormat == BGR && dstFormat == BGRA)) {
      impl_ = hwc3_to_hwc4;
    } else if ((srcFormat == RGB && dstFormat == BGR) ||
               (srcFormat == BGR && dstFormat == RGB)) {
      impl_ = hwc3_trans;
    } else if ((srcFormat == RGBA && dstFormat == BGRA) ||
               (srcFormat == BGRA && dstFormat == RGBA)) {
      impl_ = hwc4_trans;
    } else if ((srcFormat == RGB && dstFormat == GRAY) ||
               (srcFormat == BGR && dstFormat == GRAY)) {
      impl_ = hwc3_to_hwc1;
    } else if ((srcFormat == GRAY && dstFormat == RGB) ||
               (srcFormat == GRAY && dstFormat == BGR)) {
      impl_ = hwc1_to_hwc3;
    } else if ((srcFormat == RGBA && dstFormat == BGR) ||
               (srcFormat == BGRA && dstFormat == RGB)) {
      impl_ = hwc4_trans_hwc3;
    } else if ((srcFormat == RGB && dstFormat == BGRA) ||
               (srcFormat == BGR && dstFormat == RGBA)) {
      impl_ = hwc3_trans_hwc4;
    } else if ((srcFormat == GRAY && dstFormat == RGBA) ||
               (srcFormat == GRAY && dstFormat == BGRA)) {
      impl_ = hwc1_to_hwc4;
    } else if ((srcFormat == RGBA && dstFormat == GRAY) ||
               (srcFormat == BGRA && dstFormat == GRAY)) {
      impl_ = hwc4_to_hwc1;
    } else {
      printf("srcFormat: %d, dstFormat: %d does not support! \n",
             srcFormat,
             dstFormat);
    }
  }
  impl_(src, dst, srcw, srch);
}
/*
nv12(yuv) to BGR: stroe hwc dsth * dstw = srch * (srcw)
y_w = srcw, y_h = srch uv_w = srcw uv_h = 1/2 * srch
R = Y + 1.402*(V-128);
G = Y - 0.34414*(U-128) - 0.71414*(V-128);
B = Y + 1.772*(U-128);
浮点乘法用 7位精度处理（即a*b = ((a << 7)*b )>>7）
ra = 1.402 *128 = 179.456 = 179
ga = 0.34414 * 64 = 44.3721 = 44
gb = 0.71414 * 64 = 91.40992 = 91
ba = 1.772 * 62 = 226.816 = 227
*/
inline void nv12_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  int y_h = srch;
  int wout = srcw * 3;
  const uint8_t* y = src;
  const uint8_t* vu = src + y_h * srcw;

  int16x8_t bias = vdupq_n_s16(128);
  int16x8_t ga = vdupq_n_s16(44);
  int16x8_t ra = vdupq_n_s16(179);
  int16x8_t ba = vdupq_n_s16(227);
  int16x8_t gb = vdupq_n_s16(91);
  int16x8_t zero = vdupq_n_s16(0);
  int16x8_t max = vdupq_n_s16(255);

  uint8_t* zerobuf = new uint8_t[srcw];
  uint8_t* writebuf = new uint8_t[wout];
  memset(zerobuf, 0, sizeof(uint8_t) * srcw);

  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, y_h, 0, 2) {
    const uint8_t* ptr_y1 = y + i * srcw;
    const uint8_t* ptr_y2 = ptr_y1 + srcw;
    const uint8_t* ptr_vu = vu + (i / 2) * srcw;
    uint8_t* ptr_bgr1 = dst + i * wout;
    uint8_t* ptr_bgr2 = ptr_bgr1 + wout;
    if (i + 2 > y_h) {
      ptr_y2 = zerobuf;
      ptr_bgr2 = writebuf;
    }
    int j = 0;
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr_y1]]                \n"
        "prfm   pldl1keep, [%[ptr_y1], #64]   \n"
        "prfm   pldl1keep, [%[ptr_y2]]        \n"
        "prfm   pldl1keep, [%[ptr_y2], #64]   \n"
        "prfm   pldl1keep, [%[ptr_vu]]        \n"
        "prfm   pldl1keep, [%[ptr_vu], #64]   \n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr_y1]]                         @ preload a, 64byte\n"
        "pld [%[ptr_y1], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_y2]]            @ preload a, 64byte\n"
        "pld [%[ptr_y2], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_vu]]            @ preload a, 64byte\n"
        "pld [%[ptr_vu], #128]                         @ preload a, 64byte\n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#endif
    for (; j < srcw - 15; j += 16) {
      uint8x8x2_t y1 = vld2_u8(ptr_y1);  // d8 = y0y2y4y6...y14 d9 =
                                         // y1y3y5...y15
      uint8x8x2_t vu =
          vld2_u8(ptr_vu);  // d0 = v0v1v2v3v4v5...v7 d1 = u0u1u2...u7

      uint8x8x2_t y2 = vld2_u8(ptr_y2);

      uint16x8_t v = vmovl_u8(vu.val[1]);
      uint16x8_t u = vmovl_u8(vu.val[0]);
      int16x8_t v_s = vreinterpretq_s16_u16(v);
      int16x8_t u_s = vreinterpretq_s16_u16(u);
      int16x8_t v_bias = vsubq_s16(v_s, bias);
      int16x8_t u_bias = vsubq_s16(u_s, bias);

      // G = Y - 0.34414*(U-128) - 0.71414*(V-128);
      int16x8_t g0 = vmulq_s16(ga, u_bias);
      // R = Y + 1.402*(V-128);
      int16x8_t r0 = vmulq_s16(ra, v_bias);
      // B = Y + 1.772*(U-128);
      int16x8_t b0 = vmulq_s16(ba, u_bias);

      g0 = vmlaq_s16(g0, gb, v_bias);

      int16x8_t y1_0_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[0]));
      int16x8_t y1_1_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[1]));

      int16x8_t y2_0_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[0]));
      int16x8_t y2_1_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[1]));

      int16x8_t r0_bias = vshrq_n_s16(r0, 7);  // r0 / 128
      int16x8_t b0_bias = vshrq_n_s16(b0, 7);
      int16x8_t g0_bias = vshrq_n_s16(g0, 7);

      int16x8_t r0_1 = vaddq_s16(y1_0_8, r0_bias);
      int16x8_t b0_1 = vaddq_s16(y1_0_8, b0_bias);
      int16x8_t g0_1 = vsubq_s16(y1_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r0_2 = vaddq_s16(y1_1_8, r0_bias);
      int16x8_t b0_2 = vaddq_s16(y1_1_8, b0_bias);
      int16x8_t g0_2 = vsubq_s16(y1_1_8, g0_bias);

      r0_1 = vmaxq_s16(r0_1, zero);
      b0_1 = vmaxq_s16(b0_1, zero);
      g0_1 = vmaxq_s16(g0_1, zero);

      r0_2 = vmaxq_s16(r0_2, zero);
      b0_2 = vmaxq_s16(b0_2, zero);
      g0_2 = vmaxq_s16(g0_2, zero);

      r0_1 = vminq_s16(r0_1, max);
      b0_1 = vminq_s16(b0_1, max);
      g0_1 = vminq_s16(g0_1, max);

      r0_2 = vminq_s16(r0_2, max);
      b0_2 = vminq_s16(b0_2, max);
      g0_2 = vminq_s16(g0_2, max);

      uint8x8_t r00 = vreinterpret_u8_s8(vmovn_s16(r0_1));
      uint8x8_t b00 = vreinterpret_u8_s8(vmovn_s16(b0_1));
      uint8x8_t g00 = vreinterpret_u8_s8(vmovn_s16(g0_1));

      uint8x8_t r01 = vreinterpret_u8_s8(vmovn_s16(r0_2));
      uint8x8_t b01 = vreinterpret_u8_s8(vmovn_s16(b0_2));
      uint8x8_t g01 = vreinterpret_u8_s8(vmovn_s16(g0_2));

      int16x8_t r1_1 = vaddq_s16(y2_0_8, r0_bias);
      int16x8_t b1_1 = vaddq_s16(y2_0_8, b0_bias);
      int16x8_t g1_1 = vsubq_s16(y2_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r1_2 = vaddq_s16(y2_1_8, r0_bias);
      int16x8_t b1_2 = vaddq_s16(y2_1_8, b0_bias);
      int16x8_t g1_2 = vsubq_s16(y2_1_8, g0_bias);

      uint8x8x2_t r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      uint8x8x2_t b00_0 = vtrn_u8(b00, b01);
      uint8x8x2_t g00_0 = vtrn_u8(g00, g01);

      r1_1 = vmaxq_s16(r1_1, zero);
      b1_1 = vmaxq_s16(b1_1, zero);
      g1_1 = vmaxq_s16(g1_1, zero);

      r1_2 = vmaxq_s16(r1_2, zero);
      b1_2 = vmaxq_s16(b1_2, zero);
      g1_2 = vmaxq_s16(g1_2, zero);

      uint16x4_t r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      uint16x4_t r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      uint16x4_t b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      uint16x4_t b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      uint16x4_t g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      uint16x4_t g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      uint16x4x2_t r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      uint16x4x2_t b00_1 = vtrn_u16(b0_16, b1_16);
      uint16x4x2_t g00_1 = vtrn_u16(g0_16, g1_16);

      r1_1 = vminq_s16(r1_1, max);
      b1_1 = vminq_s16(b1_1, max);
      g1_1 = vminq_s16(g1_1, max);

      r1_2 = vminq_s16(r1_2, max);
      b1_2 = vminq_s16(b1_2, max);
      g1_2 = vminq_s16(g1_2, max);

      uint32x2_t r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      uint32x2_t r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      uint32x2_t b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      uint32x2_t b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      uint32x2_t g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      uint32x2_t g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      uint32x2x2_t r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      uint32x2x2_t b00_2 = vtrn_u32(b0_32, b1_32);
      uint32x2x2_t g00_2 = vtrn_u32(g0_32, g1_32);

      r00 = vreinterpret_u8_s8(vmovn_s16(r1_1));
      b00 = vreinterpret_u8_s8(vmovn_s16(b1_1));
      g00 = vreinterpret_u8_s8(vmovn_s16(g1_1));

      r01 = vreinterpret_u8_s8(vmovn_s16(r1_2));
      b01 = vreinterpret_u8_s8(vmovn_s16(b1_2));
      g01 = vreinterpret_u8_s8(vmovn_s16(g1_2));

      uint8x8_t r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      uint8x8_t b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      uint8x8_t g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      uint8x8_t r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      uint8x8_t b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      uint8x8_t g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      uint8x8x3_t v_bgr;
      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;

      r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      b00_0 = vtrn_u8(b00, b01);
      g00_0 = vtrn_u8(g00, g01);
      vst3_u8(ptr_bgr1, v_bgr);

      r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      ptr_bgr1 += 24;
      uint8x8x3_t v_bgr1;
      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;

      r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      b00_1 = vtrn_u16(b0_16, b1_16);
      g00_1 = vtrn_u16(g0_16, g1_16);

      vst3_u8(ptr_bgr1, v_bgr1);

      r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      ptr_bgr1 += 24;

      r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      b00_2 = vtrn_u32(b0_32, b1_32);
      g00_2 = vtrn_u32(g0_32, g1_32);

      ptr_vu += 16;
      ptr_y1 += 16;
      ptr_y2 += 16;

      r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;

      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;

      vst3_u8(ptr_bgr2, v_bgr);
      vst3_u8(ptr_bgr2 + 24, v_bgr1);

      ptr_bgr2 += 48;
    }
    // two data
    for (; j < srcw; j += 2) {
      uint8_t _y0 = ptr_y1[0];
      uint8_t _y1 = ptr_y1[1];
      uint8_t _v = ptr_vu[1];
      uint8_t _u = ptr_vu[0];
      uint8_t _y0_1 = ptr_y2[0];
      uint8_t _y1_1 = ptr_y2[1];

      int ra = floor((179 * (_v - 128)) >> 7);
      int ga = floor((44 * (_u - 128) + 91 * (_v - 128)) >> 7);
      int ba = floor((227 * (_u - 128)) >> 7);

      int r = _y0 + ra;
      int g = _y0 - ga;
      int b = _y0 + ba;

      int r1 = _y1 + ra;
      int g1 = _y1 - ga;
      int b1 = _y1 + ba;

      r = r < 0 ? 0 : (r > 255) ? 255 : r;
      g = g < 0 ? 0 : (g > 255) ? 255 : g;
      b = b < 0 ? 0 : (b > 255) ? 255 : b;

      r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
      g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
      b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

      *ptr_bgr1++ = b;
      *ptr_bgr1++ = g;
      *ptr_bgr1++ = r;

      int r2 = _y0_1 + ra;
      int g2 = _y0_1 - ga;
      int b2 = _y0_1 + ba;

      int r3 = _y1_1 + ra;
      int g3 = _y1_1 - ga;
      int b3 = _y1_1 + ba;

      r2 = r2 < 0 ? 0 : (r2 > 255) ? 255 : r2;
      g2 = g2 < 0 ? 0 : (g2 > 255) ? 255 : g2;
      b2 = b2 < 0 ? 0 : (b2 > 255) ? 255 : b2;

      r3 = r3 < 0 ? 0 : (r3 > 255) ? 255 : r3;
      g3 = g3 < 0 ? 0 : (g3 > 255) ? 255 : g3;
      b3 = b3 < 0 ? 0 : (b3 > 255) ? 255 : b3;

      if (j + 1 < srcw) {
        *ptr_bgr1++ = b1;
        *ptr_bgr1++ = g1;
        *ptr_bgr1++ = r1;
      }

      *ptr_bgr2++ = b2;
      *ptr_bgr2++ = g2;
      *ptr_bgr2++ = r2;

      ptr_y1 += 2;
      ptr_y2 += 2;
      ptr_vu += 2;

      if (j + 1 < srcw) {
        *ptr_bgr2++ = b3;
        *ptr_bgr2++ = g3;
        *ptr_bgr2++ = r3;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuf;
  delete[] writebuf;
}

/*
nv21(yvu) to BGR: stroe hwc dsth * dstw = srch * (srcw)
*/
inline void nv21_to_bgr(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  int y_h = srch;
  int wout = srcw * 3;
  const uint8_t* y = src;
  const uint8_t* vu = src + y_h * srcw;

  int16x8_t bias = vdupq_n_s16(128);
  int16x8_t ga = vdupq_n_s16(44);
  int16x8_t ra = vdupq_n_s16(179);
  int16x8_t ba = vdupq_n_s16(227);
  int16x8_t gb = vdupq_n_s16(91);
  int16x8_t zero = vdupq_n_s16(0);
  int16x8_t max = vdupq_n_s16(255);

  uint8_t* zerobuf = new uint8_t[srcw];
  uint8_t* writebuf = new uint8_t[wout];
  memset(zerobuf, 0, sizeof(uint8_t) * srcw);

  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, y_h, 0, 2) {
    const uint8_t* ptr_y1 = y + i * srcw;
    const uint8_t* ptr_y2 = ptr_y1 + srcw;
    const uint8_t* ptr_vu = vu + (i / 2) * srcw;
    uint8_t* ptr_bgr1 = dst + i * wout;
    uint8_t* ptr_bgr2 = ptr_bgr1 + wout;
    if (i + 2 > y_h) {
      ptr_y2 = zerobuf;
      ptr_bgr2 = writebuf;
    }
    int j = 0;
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr_y1]]                \n"
        "prfm   pldl1keep, [%[ptr_y1], #64]   \n"
        "prfm   pldl1keep, [%[ptr_y2]]        \n"
        "prfm   pldl1keep, [%[ptr_y2], #64]   \n"
        "prfm   pldl1keep, [%[ptr_vu]]        \n"
        "prfm   pldl1keep, [%[ptr_vu], #64]   \n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr_y1]]                         @ preload a, 64byte\n"
        "pld [%[ptr_y1], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_y2]]            @ preload a, 64byte\n"
        "pld [%[ptr_y2], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_vu]]            @ preload a, 64byte\n"
        "pld [%[ptr_vu], #128]                         @ preload a, 64byte\n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#endif
    for (; j < srcw - 15; j += 16) {
      uint8x8x2_t y1 = vld2_u8(ptr_y1);  // d8 = y0y2y4y6...y14 d9 =
                                         // y1y3y5...y15
      uint8x8x2_t vu =
          vld2_u8(ptr_vu);  // d0 = v0v1v2v3v4v5...v7 d1 = u0u1u2...u7

      uint8x8x2_t y2 = vld2_u8(ptr_y2);

      uint16x8_t v = vmovl_u8(vu.val[0]);
      uint16x8_t u = vmovl_u8(vu.val[1]);
      int16x8_t v_s = vreinterpretq_s16_u16(v);
      int16x8_t u_s = vreinterpretq_s16_u16(u);
      int16x8_t v_bias = vsubq_s16(v_s, bias);
      int16x8_t u_bias = vsubq_s16(u_s, bias);

      // G = Y - 0.34414*(U-128) - 0.71414*(V-128);
      int16x8_t g0 = vmulq_s16(ga, u_bias);
      // R = Y + 1.402*(V-128);
      int16x8_t r0 = vmulq_s16(ra, v_bias);
      // B = Y + 1.772*(U-128);
      int16x8_t b0 = vmulq_s16(ba, u_bias);

      g0 = vmlaq_s16(g0, gb, v_bias);

      int16x8_t y1_0_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[0]));
      int16x8_t y1_1_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[1]));

      int16x8_t y2_0_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[0]));
      int16x8_t y2_1_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[1]));

      int16x8_t r0_bias = vshrq_n_s16(r0, 7);  // r0 / 128
      int16x8_t b0_bias = vshrq_n_s16(b0, 7);
      int16x8_t g0_bias = vshrq_n_s16(g0, 7);

      int16x8_t r0_1 = vaddq_s16(y1_0_8, r0_bias);
      int16x8_t b0_1 = vaddq_s16(y1_0_8, b0_bias);
      int16x8_t g0_1 = vsubq_s16(y1_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r0_2 = vaddq_s16(y1_1_8, r0_bias);
      int16x8_t b0_2 = vaddq_s16(y1_1_8, b0_bias);
      int16x8_t g0_2 = vsubq_s16(y1_1_8, g0_bias);

      r0_1 = vmaxq_s16(r0_1, zero);
      b0_1 = vmaxq_s16(b0_1, zero);
      g0_1 = vmaxq_s16(g0_1, zero);

      r0_2 = vmaxq_s16(r0_2, zero);
      b0_2 = vmaxq_s16(b0_2, zero);
      g0_2 = vmaxq_s16(g0_2, zero);

      r0_1 = vminq_s16(r0_1, max);
      b0_1 = vminq_s16(b0_1, max);
      g0_1 = vminq_s16(g0_1, max);

      r0_2 = vminq_s16(r0_2, max);
      b0_2 = vminq_s16(b0_2, max);
      g0_2 = vminq_s16(g0_2, max);

      uint8x8_t r00 = vreinterpret_u8_s8(vmovn_s16(r0_1));
      uint8x8_t b00 = vreinterpret_u8_s8(vmovn_s16(b0_1));
      uint8x8_t g00 = vreinterpret_u8_s8(vmovn_s16(g0_1));

      uint8x8_t r01 = vreinterpret_u8_s8(vmovn_s16(r0_2));
      uint8x8_t b01 = vreinterpret_u8_s8(vmovn_s16(b0_2));
      uint8x8_t g01 = vreinterpret_u8_s8(vmovn_s16(g0_2));

      int16x8_t r1_1 = vaddq_s16(y2_0_8, r0_bias);
      int16x8_t b1_1 = vaddq_s16(y2_0_8, b0_bias);
      int16x8_t g1_1 = vsubq_s16(y2_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r1_2 = vaddq_s16(y2_1_8, r0_bias);
      int16x8_t b1_2 = vaddq_s16(y2_1_8, b0_bias);
      int16x8_t g1_2 = vsubq_s16(y2_1_8, g0_bias);

      uint8x8x2_t r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      uint8x8x2_t b00_0 = vtrn_u8(b00, b01);
      uint8x8x2_t g00_0 = vtrn_u8(g00, g01);

      r1_1 = vmaxq_s16(r1_1, zero);
      b1_1 = vmaxq_s16(b1_1, zero);
      g1_1 = vmaxq_s16(g1_1, zero);

      r1_2 = vmaxq_s16(r1_2, zero);
      b1_2 = vmaxq_s16(b1_2, zero);
      g1_2 = vmaxq_s16(g1_2, zero);

      uint16x4_t r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      uint16x4_t r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      uint16x4_t b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      uint16x4_t b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      uint16x4_t g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      uint16x4_t g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      uint16x4x2_t r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      uint16x4x2_t b00_1 = vtrn_u16(b0_16, b1_16);
      uint16x4x2_t g00_1 = vtrn_u16(g0_16, g1_16);

      r1_1 = vminq_s16(r1_1, max);
      b1_1 = vminq_s16(b1_1, max);
      g1_1 = vminq_s16(g1_1, max);

      r1_2 = vminq_s16(r1_2, max);
      b1_2 = vminq_s16(b1_2, max);
      g1_2 = vminq_s16(g1_2, max);

      uint32x2_t r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      uint32x2_t r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      uint32x2_t b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      uint32x2_t b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      uint32x2_t g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      uint32x2_t g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      uint32x2x2_t r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      uint32x2x2_t b00_2 = vtrn_u32(b0_32, b1_32);
      uint32x2x2_t g00_2 = vtrn_u32(g0_32, g1_32);

      r00 = vreinterpret_u8_s8(vmovn_s16(r1_1));
      b00 = vreinterpret_u8_s8(vmovn_s16(b1_1));
      g00 = vreinterpret_u8_s8(vmovn_s16(g1_1));

      r01 = vreinterpret_u8_s8(vmovn_s16(r1_2));
      b01 = vreinterpret_u8_s8(vmovn_s16(b1_2));
      g01 = vreinterpret_u8_s8(vmovn_s16(g1_2));

      uint8x8_t r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      uint8x8_t b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      uint8x8_t g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      uint8x8_t r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      uint8x8_t b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      uint8x8_t g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      uint8x8x3_t v_bgr;
      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;

      r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      b00_0 = vtrn_u8(b00, b01);
      g00_0 = vtrn_u8(g00, g01);

      vst3_u8(ptr_bgr1, v_bgr);

      r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      ptr_bgr1 += 24;
      uint8x8x3_t v_bgr1;
      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;

      r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      b00_1 = vtrn_u16(b0_16, b1_16);
      g00_1 = vtrn_u16(g0_16, g1_16);

      vst3_u8(ptr_bgr1, v_bgr1);

      r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      ptr_bgr1 += 24;

      r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      b00_2 = vtrn_u32(b0_32, b1_32);
      g00_2 = vtrn_u32(g0_32, g1_32);

      ptr_vu += 16;
      ptr_y1 += 16;
      ptr_y2 += 16;

      r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;

      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;

      vst3_u8(ptr_bgr2, v_bgr);
      vst3_u8(ptr_bgr2 + 24, v_bgr1);

      ptr_bgr2 += 48;
    }
    // two data
    for (; j < srcw; j += 2) {
      uint8_t _y0 = ptr_y1[0];
      uint8_t _y1 = ptr_y1[1];
      uint8_t _v = ptr_vu[0];
      uint8_t _u = ptr_vu[1];
      uint8_t _y0_1 = ptr_y2[0];
      uint8_t _y1_1 = ptr_y2[1];

      int ra = floor((179 * (_v - 128)) >> 7);
      int ga = floor((44 * (_u - 128) + 91 * (_v - 128)) >> 7);
      int ba = floor((227 * (_u - 128)) >> 7);

      int r = _y0 + ra;
      int g = _y0 - ga;
      int b = _y0 + ba;

      int r1 = _y1 + ra;
      int g1 = _y1 - ga;
      int b1 = _y1 + ba;

      r = r < 0 ? 0 : (r > 255) ? 255 : r;
      g = g < 0 ? 0 : (g > 255) ? 255 : g;
      b = b < 0 ? 0 : (b > 255) ? 255 : b;

      r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
      g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
      b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

      *ptr_bgr1++ = b;
      *ptr_bgr1++ = g;
      *ptr_bgr1++ = r;

      int r2 = _y0_1 + ra;
      int g2 = _y0_1 - ga;
      int b2 = _y0_1 + ba;

      int r3 = _y1_1 + ra;
      int g3 = _y1_1 - ga;
      int b3 = _y1_1 + ba;

      r2 = r2 < 0 ? 0 : (r2 > 255) ? 255 : r2;
      g2 = g2 < 0 ? 0 : (g2 > 255) ? 255 : g2;
      b2 = b2 < 0 ? 0 : (b2 > 255) ? 255 : b2;

      r3 = r3 < 0 ? 0 : (r3 > 255) ? 255 : r3;
      g3 = g3 < 0 ? 0 : (g3 > 255) ? 255 : g3;
      b3 = b3 < 0 ? 0 : (b3 > 255) ? 255 : b3;

      if (j + 1 < srcw) {
        *ptr_bgr1++ = b1;
        *ptr_bgr1++ = g1;
        *ptr_bgr1++ = r1;
      }

      *ptr_bgr2++ = b2;
      *ptr_bgr2++ = g2;
      *ptr_bgr2++ = r2;

      ptr_y1 += 2;
      ptr_y2 += 2;
      ptr_vu += 2;

      if (j + 1 < srcw) {
        *ptr_bgr2++ = b3;
        *ptr_bgr2++ = g3;
        *ptr_bgr2++ = r3;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuf;
  delete[] writebuf;
}

// nv12(yuv) to BGRA: stroe hwc dsth * dstw = srch * (srcw) y_w = srcw, y_h =
// srch uv_w = srcw uv_h = 1/2 * srch
inline void nv12_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  int y_h = srch;
  int vu_h = 1 / 2 * srch;
  const uint8_t* y = src;
  const uint8_t* vu = src + y_h * srcw;
  int wout = srcw * 4;

  uint8_t* zerobuf = new uint8_t[srcw];
  uint8_t* writebuf = new uint8_t[wout];
  memset(zerobuf, 0, sizeof(uint8_t) * srcw);

  int16x8_t bias = vdupq_n_s16(128);
  int16x8_t ga = vdupq_n_s16(44);
  int16x8_t ra = vdupq_n_s16(179);
  int16x8_t ba = vdupq_n_s16(227);
  int16x8_t gb = vdupq_n_s16(91);
  int16x8_t zero = vdupq_n_s16(0);
  int16x8_t max = vdupq_n_s16(255);
  uint8x8_t a_8 = vdup_n_u8(255);
  LITE_PARALLEL_COMMON_BEGIN(i, tid, y_h, 0, 2) {
    const uint8_t* ptr_y1 = y + i * srcw;
    const uint8_t* ptr_y2 = ptr_y1 + srcw;
    const uint8_t* ptr_vu = vu + (i / 2) * srcw;
    uint8_t* ptr_bgr1 = dst + i * wout;
    uint8_t* ptr_bgr2 = ptr_bgr1 + wout;
    if (i + 2 > y_h) {
      ptr_y2 = zerobuf;
      ptr_bgr2 = writebuf;
    }
    int j = 0;
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr_y1]]                \n"
        "prfm   pldl1keep, [%[ptr_y1], #64]   \n"
        "prfm   pldl1keep, [%[ptr_y2]]        \n"
        "prfm   pldl1keep, [%[ptr_y2], #64]   \n"
        "prfm   pldl1keep, [%[ptr_vu]]        \n"
        "prfm   pldl1keep, [%[ptr_vu], #64]   \n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr_y1]]                         @ preload a, 64byte\n"
        "pld [%[ptr_y1], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_y2]]            @ preload a, 64byte\n"
        "pld [%[ptr_y2], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_vu]]            @ preload a, 64byte\n"
        "pld [%[ptr_vu], #128]                         @ preload a, 64byte\n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#endif
    for (; j < srcw - 15; j += 16) {
      uint8x8x2_t y1 = vld2_u8(ptr_y1);  // d8 = y0y2y4y6...y14 d9 =
                                         // y1y3y5...y15
      uint8x8x2_t vu =
          vld2_u8(ptr_vu);  // d0 = v0v1v2v3v4v5...v7 d1 = u0u1u2...u7

      uint8x8x2_t y2 = vld2_u8(ptr_y2);

      uint16x8_t v = vmovl_u8(vu.val[1]);
      uint16x8_t u = vmovl_u8(vu.val[0]);
      int16x8_t v_s = vreinterpretq_s16_u16(v);
      int16x8_t u_s = vreinterpretq_s16_u16(u);
      int16x8_t v_bias = vsubq_s16(v_s, bias);
      int16x8_t u_bias = vsubq_s16(u_s, bias);

      // G = Y - 0.34414*(U-128) - 0.71414*(V-128);
      int16x8_t g0 = vmulq_s16(ga, u_bias);
      // R = Y + 1.402*(V-128);
      int16x8_t r0 = vmulq_s16(ra, v_bias);
      // B = Y + 1.772*(U-128);
      int16x8_t b0 = vmulq_s16(ba, u_bias);

      g0 = vmlaq_s16(g0, gb, v_bias);

      int16x8_t y1_0_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[0]));
      int16x8_t y1_1_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[1]));

      int16x8_t y2_0_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[0]));
      int16x8_t y2_1_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[1]));

      int16x8_t r0_bias = vshrq_n_s16(r0, 7);  // r0 / 128
      int16x8_t b0_bias = vshrq_n_s16(b0, 7);
      int16x8_t g0_bias = vshrq_n_s16(g0, 7);

      int16x8_t r0_1 = vaddq_s16(y1_0_8, r0_bias);
      int16x8_t b0_1 = vaddq_s16(y1_0_8, b0_bias);
      int16x8_t g0_1 = vsubq_s16(y1_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r0_2 = vaddq_s16(y1_1_8, r0_bias);
      int16x8_t b0_2 = vaddq_s16(y1_1_8, b0_bias);
      int16x8_t g0_2 = vsubq_s16(y1_1_8, g0_bias);

      r0_1 = vmaxq_s16(r0_1, zero);
      b0_1 = vmaxq_s16(b0_1, zero);
      g0_1 = vmaxq_s16(g0_1, zero);

      r0_2 = vmaxq_s16(r0_2, zero);
      b0_2 = vmaxq_s16(b0_2, zero);
      g0_2 = vmaxq_s16(g0_2, zero);

      r0_1 = vminq_s16(r0_1, max);
      b0_1 = vminq_s16(b0_1, max);
      g0_1 = vminq_s16(g0_1, max);

      r0_2 = vminq_s16(r0_2, max);
      b0_2 = vminq_s16(b0_2, max);
      g0_2 = vminq_s16(g0_2, max);

      uint8x8_t r00 = vreinterpret_u8_s8(vmovn_s16(r0_1));
      uint8x8_t b00 = vreinterpret_u8_s8(vmovn_s16(b0_1));
      uint8x8_t g00 = vreinterpret_u8_s8(vmovn_s16(g0_1));

      uint8x8_t r01 = vreinterpret_u8_s8(vmovn_s16(r0_2));
      uint8x8_t b01 = vreinterpret_u8_s8(vmovn_s16(b0_2));
      uint8x8_t g01 = vreinterpret_u8_s8(vmovn_s16(g0_2));

      int16x8_t r1_1 = vaddq_s16(y2_0_8, r0_bias);
      int16x8_t b1_1 = vaddq_s16(y2_0_8, b0_bias);
      int16x8_t g1_1 = vsubq_s16(y2_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r1_2 = vaddq_s16(y2_1_8, r0_bias);
      int16x8_t b1_2 = vaddq_s16(y2_1_8, b0_bias);
      int16x8_t g1_2 = vsubq_s16(y2_1_8, g0_bias);

      uint8x8x2_t r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      uint8x8x2_t b00_0 = vtrn_u8(b00, b01);
      uint8x8x2_t g00_0 = vtrn_u8(g00, g01);

      r1_1 = vmaxq_s16(r1_1, zero);
      b1_1 = vmaxq_s16(b1_1, zero);
      g1_1 = vmaxq_s16(g1_1, zero);

      r1_2 = vmaxq_s16(r1_2, zero);
      b1_2 = vmaxq_s16(b1_2, zero);
      g1_2 = vmaxq_s16(g1_2, zero);

      uint16x4_t r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      uint16x4_t r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      uint16x4_t b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      uint16x4_t b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      uint16x4_t g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      uint16x4_t g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      uint16x4x2_t r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      uint16x4x2_t b00_1 = vtrn_u16(b0_16, b1_16);
      uint16x4x2_t g00_1 = vtrn_u16(g0_16, g1_16);

      r1_1 = vminq_s16(r1_1, max);
      b1_1 = vminq_s16(b1_1, max);
      g1_1 = vminq_s16(g1_1, max);

      r1_2 = vminq_s16(r1_2, max);
      b1_2 = vminq_s16(b1_2, max);
      g1_2 = vminq_s16(g1_2, max);

      uint32x2_t r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      uint32x2_t r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      uint32x2_t b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      uint32x2_t b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      uint32x2_t g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      uint32x2_t g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      uint32x2x2_t r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      uint32x2x2_t b00_2 = vtrn_u32(b0_32, b1_32);
      uint32x2x2_t g00_2 = vtrn_u32(g0_32, g1_32);

      r00 = vreinterpret_u8_s8(vmovn_s16(r1_1));
      b00 = vreinterpret_u8_s8(vmovn_s16(b1_1));
      g00 = vreinterpret_u8_s8(vmovn_s16(g1_1));

      r01 = vreinterpret_u8_s8(vmovn_s16(r1_2));
      b01 = vreinterpret_u8_s8(vmovn_s16(b1_2));
      g01 = vreinterpret_u8_s8(vmovn_s16(g1_2));

      uint8x8_t r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      uint8x8_t b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      uint8x8_t g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      uint8x8_t r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      uint8x8_t b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      uint8x8_t g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      uint8x8x4_t v_bgr;
      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;
      v_bgr.val[3] = a_8;

      r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      b00_0 = vtrn_u8(b00, b01);
      g00_0 = vtrn_u8(g00, g01);

      vst4_u8(ptr_bgr1, v_bgr);

      r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      ptr_bgr1 += 32;
      // uint8x8x3_t v_bgr1;
      uint8x8x4_t v_bgr1;
      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;
      v_bgr1.val[3] = a_8;

      r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      b00_1 = vtrn_u16(b0_16, b1_16);
      g00_1 = vtrn_u16(g0_16, g1_16);

      // vst3_u8(ptr_bgr1, v_bgr1);
      vst4_u8(ptr_bgr1, v_bgr1);

      r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      // ptr_bgr1 += 24;
      ptr_bgr1 += 32;

      r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      b00_2 = vtrn_u32(b0_32, b1_32);
      g00_2 = vtrn_u32(g0_32, g1_32);

      ptr_vu += 16;
      ptr_y1 += 16;
      ptr_y2 += 16;

      r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;

      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;

      vst4_u8(ptr_bgr2, v_bgr);
      vst4_u8(ptr_bgr2 + 32, v_bgr1);

      ptr_bgr2 += 64;
    }
    // two data
    for (; j < srcw; j += 2) {
      uint8_t _y0 = ptr_y1[0];
      uint8_t _y1 = ptr_y1[1];
      uint8_t _v = ptr_vu[1];
      uint8_t _u = ptr_vu[0];
      uint8_t _y0_1 = ptr_y2[0];
      uint8_t _y1_1 = ptr_y2[1];

      int ra = floor((179 * (_v - 128)) >> 7);
      int ga = floor((44 * (_u - 128) + 91 * (_v - 128)) >> 7);
      int ba = floor((227 * (_u - 128)) >> 7);

      int r = _y0 + ra;
      int g = _y0 - ga;
      int b = _y0 + ba;

      int r1 = _y1 + ra;
      int g1 = _y1 - ga;
      int b1 = _y1 + ba;

      r = r < 0 ? 0 : (r > 255) ? 255 : r;
      g = g < 0 ? 0 : (g > 255) ? 255 : g;
      b = b < 0 ? 0 : (b > 255) ? 255 : b;

      r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
      g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
      b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

      *ptr_bgr1++ = b;
      *ptr_bgr1++ = g;
      *ptr_bgr1++ = r;
      *ptr_bgr1++ = 255;

      int r2 = _y0_1 + ra;
      int g2 = _y0_1 - ga;
      int b2 = _y0_1 + ba;

      int r3 = _y1_1 + ra;
      int g3 = _y1_1 - ga;
      int b3 = _y1_1 + ba;

      r2 = r2 < 0 ? 0 : (r2 > 255) ? 255 : r2;
      g2 = g2 < 0 ? 0 : (g2 > 255) ? 255 : g2;
      b2 = b2 < 0 ? 0 : (b2 > 255) ? 255 : b2;

      r3 = r3 < 0 ? 0 : (r3 > 255) ? 255 : r3;
      g3 = g3 < 0 ? 0 : (g3 > 255) ? 255 : g3;
      b3 = b3 < 0 ? 0 : (b3 > 255) ? 255 : b3;

      if (j + 1 < srcw) {
        *ptr_bgr1++ = b1;
        *ptr_bgr1++ = g1;
        *ptr_bgr1++ = r1;
        *ptr_bgr1++ = 255;
      }

      *ptr_bgr2++ = b2;
      *ptr_bgr2++ = g2;
      *ptr_bgr2++ = r2;
      *ptr_bgr2++ = 255;

      ptr_y1 += 2;
      ptr_y2 += 2;
      ptr_vu += 2;

      if (j + 1 < srcw) {
        *ptr_bgr2++ = b3;
        *ptr_bgr2++ = g3;
        *ptr_bgr2++ = r3;
        *ptr_bgr2++ = 255;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuf;
  delete[] writebuf;
}

// nv21(yvu) to BGRA:store hwc dsth * dstw = srch * srcw y_w = srcw, y_h = srch
// uv_w = srcw uv_h = 1/2 * srch
inline void nv21_to_bgra(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  int y_h = srch;
  int vu_h = 1 / 2 * srch;
  const uint8_t* y = src;
  const uint8_t* vu = src + y_h * srcw;
  int wout = srcw * 4;

  uint8_t* zerobuf = new uint8_t[srcw];
  uint8_t* writebuf = new uint8_t[wout];
  memset(zerobuf, 0, sizeof(uint8_t) * srcw);
  uint8x8_t a_8 = vdup_n_u8(255);
  int16x8_t bias = vdupq_n_s16(128);
  int16x8_t ga = vdupq_n_s16(44);
  int16x8_t ra = vdupq_n_s16(179);
  int16x8_t ba = vdupq_n_s16(227);
  int16x8_t gb = vdupq_n_s16(91);
  int16x8_t zero = vdupq_n_s16(0);
  int16x8_t max = vdupq_n_s16(255);

  LITE_PARALLEL_COMMON_BEGIN(i, tid, y_h, 0, 2) {
    const uint8_t* ptr_y1 = y + i * srcw;
    const uint8_t* ptr_y2 = ptr_y1 + srcw;
    const uint8_t* ptr_vu = vu + (i / 2) * srcw;
    uint8_t* ptr_bgr1 = dst + i * wout;
    uint8_t* ptr_bgr2 = ptr_bgr1 + wout;
    if (i + 2 > y_h) {
      ptr_y2 = zerobuf;
      ptr_bgr2 = writebuf;
    }
    int j = 0;
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr_y1]]                \n"
        "prfm   pldl1keep, [%[ptr_y1], #64]   \n"
        "prfm   pldl1keep, [%[ptr_y2]]        \n"
        "prfm   pldl1keep, [%[ptr_y2], #64]   \n"
        "prfm   pldl1keep, [%[ptr_vu]]        \n"
        "prfm   pldl1keep, [%[ptr_vu], #64]   \n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr_y1]]                         @ preload a, 64byte\n"
        "pld [%[ptr_y1], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_y2]]            @ preload a, 64byte\n"
        "pld [%[ptr_y2], #128]                         @ preload a, 64byte\n"
        "pld [%[ptr_vu]]            @ preload a, 64byte\n"
        "pld [%[ptr_vu], #128]                         @ preload a, 64byte\n"
        :
        : [ptr_y1] "r"(ptr_y1), [ptr_y2] "r"(ptr_y2), [ptr_vu] "r"(ptr_vu)
        : "memory");
#endif
    for (; j < srcw - 15; j += 16) {
      uint8x8x2_t y1 = vld2_u8(ptr_y1);  // d8 = y0y2y4y6...y14 d9 =
                                         // y1y3y5...y15
      uint8x8x2_t vu =
          vld2_u8(ptr_vu);  // d0 = v0v1v2v3v4v5...v7 d1 = u0u1u2...u7

      uint8x8x2_t y2 = vld2_u8(ptr_y2);

      uint16x8_t v = vmovl_u8(vu.val[0]);
      uint16x8_t u = vmovl_u8(vu.val[1]);
      int16x8_t v_s = vreinterpretq_s16_u16(v);
      int16x8_t u_s = vreinterpretq_s16_u16(u);
      int16x8_t v_bias = vsubq_s16(v_s, bias);
      int16x8_t u_bias = vsubq_s16(u_s, bias);

      // G = Y - 0.34414*(U-128) - 0.71414*(V-128);
      int16x8_t g0 = vmulq_s16(ga, u_bias);
      // R = Y + 1.402*(V-128);
      int16x8_t r0 = vmulq_s16(ra, v_bias);
      // B = Y + 1.772*(U-128);
      int16x8_t b0 = vmulq_s16(ba, u_bias);

      g0 = vmlaq_s16(g0, gb, v_bias);

      int16x8_t y1_0_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[0]));
      int16x8_t y1_1_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[1]));

      int16x8_t y2_0_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[0]));
      int16x8_t y2_1_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[1]));

      int16x8_t r0_bias = vshrq_n_s16(r0, 7);  // r0 / 128
      int16x8_t b0_bias = vshrq_n_s16(b0, 7);
      int16x8_t g0_bias = vshrq_n_s16(g0, 7);

      int16x8_t r0_1 = vaddq_s16(y1_0_8, r0_bias);
      int16x8_t b0_1 = vaddq_s16(y1_0_8, b0_bias);
      int16x8_t g0_1 = vsubq_s16(y1_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r0_2 = vaddq_s16(y1_1_8, r0_bias);
      int16x8_t b0_2 = vaddq_s16(y1_1_8, b0_bias);
      int16x8_t g0_2 = vsubq_s16(y1_1_8, g0_bias);

      r0_1 = vmaxq_s16(r0_1, zero);
      b0_1 = vmaxq_s16(b0_1, zero);
      g0_1 = vmaxq_s16(g0_1, zero);

      r0_2 = vmaxq_s16(r0_2, zero);
      b0_2 = vmaxq_s16(b0_2, zero);
      g0_2 = vmaxq_s16(g0_2, zero);

      r0_1 = vminq_s16(r0_1, max);
      b0_1 = vminq_s16(b0_1, max);
      g0_1 = vminq_s16(g0_1, max);

      r0_2 = vminq_s16(r0_2, max);
      b0_2 = vminq_s16(b0_2, max);
      g0_2 = vminq_s16(g0_2, max);

      uint8x8_t r00 = vreinterpret_u8_s8(vmovn_s16(r0_1));
      uint8x8_t b00 = vreinterpret_u8_s8(vmovn_s16(b0_1));
      uint8x8_t g00 = vreinterpret_u8_s8(vmovn_s16(g0_1));

      uint8x8_t r01 = vreinterpret_u8_s8(vmovn_s16(r0_2));
      uint8x8_t b01 = vreinterpret_u8_s8(vmovn_s16(b0_2));
      uint8x8_t g01 = vreinterpret_u8_s8(vmovn_s16(g0_2));

      int16x8_t r1_1 = vaddq_s16(y2_0_8, r0_bias);
      int16x8_t b1_1 = vaddq_s16(y2_0_8, b0_bias);
      int16x8_t g1_1 = vsubq_s16(y2_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

      int16x8_t r1_2 = vaddq_s16(y2_1_8, r0_bias);
      int16x8_t b1_2 = vaddq_s16(y2_1_8, b0_bias);
      int16x8_t g1_2 = vsubq_s16(y2_1_8, g0_bias);

      uint8x8x2_t r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      uint8x8x2_t b00_0 = vtrn_u8(b00, b01);
      uint8x8x2_t g00_0 = vtrn_u8(g00, g01);

      r1_1 = vmaxq_s16(r1_1, zero);
      b1_1 = vmaxq_s16(b1_1, zero);
      g1_1 = vmaxq_s16(g1_1, zero);

      r1_2 = vmaxq_s16(r1_2, zero);
      b1_2 = vmaxq_s16(b1_2, zero);
      g1_2 = vmaxq_s16(g1_2, zero);

      uint16x4_t r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      uint16x4_t r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      uint16x4_t b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      uint16x4_t b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      uint16x4_t g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      uint16x4_t g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      uint16x4x2_t r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      uint16x4x2_t b00_1 = vtrn_u16(b0_16, b1_16);
      uint16x4x2_t g00_1 = vtrn_u16(g0_16, g1_16);

      r1_1 = vminq_s16(r1_1, max);
      b1_1 = vminq_s16(b1_1, max);
      g1_1 = vminq_s16(g1_1, max);

      r1_2 = vminq_s16(r1_2, max);
      b1_2 = vminq_s16(b1_2, max);
      g1_2 = vminq_s16(g1_2, max);

      uint32x2_t r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      uint32x2_t r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      uint32x2_t b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      uint32x2_t b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      uint32x2_t g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      uint32x2_t g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      uint32x2x2_t r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      uint32x2x2_t b00_2 = vtrn_u32(b0_32, b1_32);
      uint32x2x2_t g00_2 = vtrn_u32(g0_32, g1_32);

      r00 = vreinterpret_u8_s8(vmovn_s16(r1_1));
      b00 = vreinterpret_u8_s8(vmovn_s16(b1_1));
      g00 = vreinterpret_u8_s8(vmovn_s16(g1_1));

      r01 = vreinterpret_u8_s8(vmovn_s16(r1_2));
      b01 = vreinterpret_u8_s8(vmovn_s16(b1_2));
      g01 = vreinterpret_u8_s8(vmovn_s16(g1_2));

      uint8x8_t r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      uint8x8_t b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      uint8x8_t g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      uint8x8_t r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      uint8x8_t b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      uint8x8_t g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      uint8x8x4_t v_bgr;
      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;
      v_bgr.val[3] = a_8;

      r00_0 = vtrn_u8(r00, r01);  // 014589  236710
      b00_0 = vtrn_u8(b00, b01);
      g00_0 = vtrn_u8(g00, g01);

      vst4_u8(ptr_bgr1, v_bgr);

      r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
      r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

      b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
      b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

      g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
      g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

      ptr_bgr1 += 32;
      // uint8x8x3_t v_bgr1;
      uint8x8x4_t v_bgr1;
      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;
      v_bgr1.val[3] = a_8;

      r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
      b00_1 = vtrn_u16(b0_16, b1_16);
      g00_1 = vtrn_u16(g0_16, g1_16);

      // vst3_u8(ptr_bgr1, v_bgr1);
      vst4_u8(ptr_bgr1, v_bgr1);

      r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
      r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

      b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
      b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

      g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
      g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

      // ptr_bgr1 += 24;
      ptr_bgr1 += 32;

      r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
      b00_2 = vtrn_u32(b0_32, b1_32);
      g00_2 = vtrn_u32(g0_32, g1_32);

      ptr_vu += 16;
      ptr_y1 += 16;
      ptr_y2 += 16;

      r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
      b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
      g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

      r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
      b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
      g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

      v_bgr.val[0] = b0_8;
      v_bgr.val[1] = g0_8;
      v_bgr.val[2] = r0_8;

      v_bgr1.val[0] = b1_8;
      v_bgr1.val[1] = g1_8;
      v_bgr1.val[2] = r1_8;

      vst4_u8(ptr_bgr2, v_bgr);
      vst4_u8(ptr_bgr2 + 32, v_bgr1);

      ptr_bgr2 += 64;
    }
    // two data
    for (; j < srcw; j += 2) {
      uint8_t _y0 = ptr_y1[0];
      uint8_t _y1 = ptr_y1[1];
      uint8_t _v = ptr_vu[0];
      uint8_t _u = ptr_vu[1];
      uint8_t _y0_1 = ptr_y2[0];
      uint8_t _y1_1 = ptr_y2[1];

      int ra = floor((179 * (_v - 128)) >> 7);
      int ga = floor((44 * (_u - 128) + 91 * (_v - 128)) >> 7);
      int ba = floor((227 * (_u - 128)) >> 7);

      int r = _y0 + ra;
      int g = _y0 - ga;
      int b = _y0 + ba;

      int r1 = _y1 + ra;
      int g1 = _y1 - ga;
      int b1 = _y1 + ba;

      r = r < 0 ? 0 : (r > 255) ? 255 : r;
      g = g < 0 ? 0 : (g > 255) ? 255 : g;
      b = b < 0 ? 0 : (b > 255) ? 255 : b;

      r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
      g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
      b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

      *ptr_bgr1++ = b;
      *ptr_bgr1++ = g;
      *ptr_bgr1++ = r;
      *ptr_bgr1++ = 255;

      int r2 = _y0_1 + ra;
      int g2 = _y0_1 - ga;
      int b2 = _y0_1 + ba;

      int r3 = _y1_1 + ra;
      int g3 = _y1_1 - ga;
      int b3 = _y1_1 + ba;

      r2 = r2 < 0 ? 0 : (r2 > 255) ? 255 : r2;
      g2 = g2 < 0 ? 0 : (g2 > 255) ? 255 : g2;
      b2 = b2 < 0 ? 0 : (b2 > 255) ? 255 : b2;

      r3 = r3 < 0 ? 0 : (r3 > 255) ? 255 : r3;
      g3 = g3 < 0 ? 0 : (g3 > 255) ? 255 : g3;
      b3 = b3 < 0 ? 0 : (b3 > 255) ? 255 : b3;

      if (j + 1 < srcw) {
        *ptr_bgr1++ = b1;
        *ptr_bgr1++ = g1;
        *ptr_bgr1++ = r1;
        *ptr_bgr1++ = 255;
      }

      *ptr_bgr2++ = b2;
      *ptr_bgr2++ = g2;
      *ptr_bgr2++ = r2;
      *ptr_bgr2++ = 255;

      ptr_y1 += 2;
      ptr_y2 += 2;
      ptr_vu += 2;

      if (j + 1 < srcw) {
        *ptr_bgr2++ = b3;
        *ptr_bgr2++ = g3;
        *ptr_bgr2++ = r3;
        *ptr_bgr2++ = 255;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuf;
  delete[] writebuf;
}

/*
采用CV_BGR2GRAY,转换公式Gray = 0.1140*B + 0.5870*G + 0.2989*R
采用CV_RGB2GRAY,转换公式Gray = 0.1140*R + 0.5870*G + 0.2989*B
b = 0.114 *128 = 14.529 = 15
g = 0.587 * 128 = 75.136 = 75
r = 0.2989 * 127 = 38.2592 = 38
Gray = (15*B + 75*G + 38*R)/128
bgr2gray, rgb2gray
*/
void hwc3_to_hwc1(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  uint8_t b = 15;
  uint8_t g = 75;
  uint8_t r = 38;

  uint8x8_t vb = vdup_n_u8(b);
  uint8x8_t vg = vdup_n_u8(g);
  uint8x8_t vr = vdup_n_u8(r);
  uint8_t vb_array[8] = {b, b, b, b, b, b, b, b};
  uint8_t vg_array[8] = {g, g, g, g, g, g, g, g};
  uint8_t vr_array[8] = {r, r, r, r, r, r, r, r};
  int cnt_pro = srcw >> 3;
  int remain_pro = srcw % 8;
  int win = srcw * 3;
  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, srch - 3, 0, 4) {
    int j = 0;
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;
    uint8_t* outr0 = dst + i * srcw;
    uint8_t* outr1 = outr0 + srcw;
    uint8_t* outr2 = outr1 + srcw;
    uint8_t* outr3 = outr2 + srcw;
    int cnt = cnt_pro;
    if (cnt > 0) {
#ifdef __aarch64__
      asm volatile(
          "prfm   pldl1keep, [%[inptr0]]                \n"
          "prfm   pldl1keep, [%[inptr0], #128]   \n"
          "prfm   pldl1keep, [%[inptr1]]        \n"
          "prfm   pldl1keep, [%[inptr1], #128]   \n"
          "prfm   pldl1keep, [%[inptr2]]                \n"
          "prfm   pldl1keep, [%[inptr2], #128]   \n"
          "prfm   pldl1keep, [%[inptr3]]                \n"
          "prfm   pldl1keep, [%[inptr3], #128]   \n"
          "ld1 {v21.8b}, [%[vb]]                 \n"
          "ld1 {v22.8b}, [%[vg]]                 \n"
          "ld1 {v23.8b}, [%[vr]]                 \n"
          "1: \n"
          "ld3 {v0.8b - v2.8b}, [%[inptr0]], #24 \n"   // d8 = y0y3y6y9.. d9 =
                                                       // y1y4y7...
          "ld3 {v3.8b - v5.8b}, [%[inptr1]], #24 \n"   // d8 = y0y3y6y9.. d9 =
                                                       // y1y4y7...
          "ld3 {v6.8b - v8.8b}, [%[inptr2]], #24 \n"   // d8 = y0y3y6y9.. d9 =
                                                       // y1y4y7...
          "ld3 {v9.8b - v11.8b}, [%[inptr3]], #24 \n"  // d8 = y0y3y6y9.. d9 =
                                                       // y1y4y7...
          // mul b
          "umull v12.8h, v0.8b, v21.8b \n"  // v0 * vb
          "umull v13.8h, v3.8b, v21.8b \n"  // v0 * vb
          "umull v14.8h, v6.8b, v21.8b \n"  // v0 * vb
          "umull v15.8h, v9.8b, v21.8b \n"  // v0 * vb
          // mul g
          "umull v16.8h, v1.8b, v22.8b \n"   // v0 * vb
          "umull v17.8h, v4.8b, v22.8b \n"   // v0 * vb
          "umull v18.8h, v7.8b, v22.8b \n"   // v0 * vb
          "umull v19.8h, v10.8b, v22.8b \n"  // v0 * vb
          // mul r
          "umlal v12.8h, v2.8b, v23.8b \n"   // v0 * vb
          "umlal v13.8h, v5.8b, v23.8b \n"   // v0 * vb
          "umlal v14.8h, v8.8b, v23.8b \n"   // v0 * vb
          "umlal v15.8h, v11.8b, v23.8b \n"  // v0 * vb
          // 16->32
          "uaddl v0.4s, v16.4h, v12.4h \n"
          "uaddl2 v1.4s, v16.8h, v12.8h \n"
          "uaddl v2.4s, v17.4h, v13.4h \n"
          "uaddl2 v3.4s, v17.8h, v13.8h \n"
          "uaddl v4.4s, v18.4h, v14.4h \n"
          "uaddl2 v5.4s, v18.8h, v14.8h \n"
          "uaddl v6.4s, v19.4h, v15.4h \n"
          "uaddl2 v7.4s, v19.8h, v15.8h \n"
          // 32->16 v0 >> 7
          "shrn v12.4h, v0.4s, #7 \n"
          "shrn2 v12.8h, v1.4s, #7 \n"
          "shrn v13.4h, v2.4s, #7 \n"
          "shrn2 v13.8h, v3.4s, #7 \n"
          "shrn v14.4h, v4.4s, #7 \n"
          "shrn2 v14.8h, v5.4s, #7 \n"
          "shrn v15.4h, v6.4s, #7 \n"
          "shrn2 v15.8h, v7.4s, #7 \n"
          // 16->8
          "xtn v0.8b, v12.8h \n"
          "xtn v1.8b, v13.8h \n"
          "xtn v2.8b, v14.8h \n"
          "xtn v3.8b, v15.8h \n"
          "subs %w[cnt], %w[cnt], #1 \n"
          "st1 {v0.8b}, [%[outr0]], #8 \n"
          "st1 {v1.8b}, [%[outr1]], #8 \n"
          "st1 {v2.8b}, [%[outr2]], #8 \n"
          "st1 {v3.8b}, [%[outr3]], #8 \n"
          "bne 1b \n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outr0] "+r"(outr0),
            [outr1] "+r"(outr1),
            [outr2] "+r"(outr2),
            [outr3] "+r"(outr3),
            [cnt] "+r"(cnt)
          : [vb] "r"(vb_array), [vg] "r"(vg_array), [vr] "r"(vr_array)
          : "cc",
            "memory",
            "v0",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15",
            "v16",
            "v17",
            "v18",
            "v19",
            "v20",
            "v21",
            "v22",
            "v23");
#else
      asm volatile(
          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr1]]            @ preload a, 64byte\n"
          "pld [%[inptr1], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr2]]            @ preload a, 64byte\n"
          "pld [%[inptr2], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr3]]            @ preload a, 64byte\n"
          "pld [%[inptr3], #128]                         @ preload a, 64byte\n"
          "vld1.8 d0, [%[vb]] \n"
          "vld1.8 d1, [%[vg]] \n"
          "vld1.8 d2, [%[vr]] \n"
          "1: \n"
          "vld3.8 {d3, d4, d5}, [%[inptr0]]! \n"
          "vld3.8 {d6, d7, d8}, [%[inptr1]]! \n"
          "vld3.8 {d9, d10, d11}, [%[inptr2]]! \n"
          "vld3.8 {d12, d13, d14}, [%[inptr3]]! \n"
          // vb
          "vmull.u8 q8, d3, d0 \n"
          "vmull.u8 q9, d6, d0 \n"
          "vmull.u8 q10, d9, d0 \n"
          "vmull.u8 q11, d12, d0 \n"
          // vg
          "vmull.u8 q12, d4, d1 \n"
          "vmull.u8 q13, d7, d1 \n"
          "vmull.u8 q14, d10, d1 \n"
          "vmull.u8 q15, d13, d1 \n"
          // vr
          "vmlal.u8 q8, d5, d2 \n"
          "vmlal.u8 q9, d8, d2 \n"
          "vmlal.u8 q10, d11, d2 \n"
          "vmlal.u8 q11, d14, d2 \n"
          // 16->32
          "vaddl.u16 q2, d24, d16 \n"
          "vaddl.u16 q3, d25, d17 \n"
          "vaddl.u16 q4, d26, d18 \n"
          "vaddl.u16 q5, d27, d19 \n"
          "vaddl.u16 q6, d28, d20 \n"
          "vaddl.u16 q7, d29, d21 \n"
          "vaddl.u16 q8, d30, d22 \n"
          "vaddl.u16 q9, d31, d23 \n"
          // 32->16 q2 >> 7
          "vshrn.u32  d20, q2, #7 \n"
          "vshrn.u32 d21, q3, #7 \n"
          "vshrn.u32 d22, q4, #7 \n"
          "vshrn.u32 d23, q5, #7 \n"
          "vshrn.u32 d24, q6, #7 \n"
          "vshrn.u32 d25, q7, #7 \n"
          "vshrn.u32 d26, q8, #7 \n"
          "vshrn.u32 d27, q9, #7 \n"
          // 16->8
          "vmovn.u16 d4, q10 \n"
          "vmovn.u16 d5, q11 \n"
          "vmovn.u16 d6, q12 \n"
          "vmovn.u16 d7, q13 \n"
          "subs %[cnt], #1 \n"
          // store
          "vst1.8 d4, [%[outr0]]! \n"
          "vst1.8 d5, [%[outr1]]! \n"
          "vst1.8 d6, [%[outr2]]! \n"
          "vst1.8 d7, [%[outr3]]! \n"
          "bne 1b \n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outr0] "+r"(outr0),
            [outr1] "+r"(outr1),
            [outr2] "+r"(outr2),
            [outr3] "+r"(outr3),
            [cnt] "+r"(cnt)
          : [vb] "r"(vb_array), [vg] "r"(vg_array), [vr] "r"(vr_array)
          : "cc",
            "memory",
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15");
#endif
    }
    for (; j < remain_pro; j++) {
      *outr0++ = (inptr0[0] * b + inptr0[1] * g + inptr0[2] * r) >> 7;
      *outr1++ = (inptr1[0] * b + inptr1[1] * g + inptr1[2] * r) >> 7;
      *outr2++ = (inptr2[0] * b + inptr2[1] * g + inptr2[2] * r) >> 7;
      *outr3++ = (inptr3[0] * b + inptr3[1] * g + inptr3[2] * r) >> 7;
      inptr0 += 3;
      inptr1 += 3;
      inptr2 += 3;
      inptr3 += 3;
    }
  }
  LITE_PARALLEL_COMMON_END();
  for (; i < srch; i++) {
    int j = 0;
    const uint8_t* inptr0 = src + i * win;
    uint8_t* outr0 = dst + i * srcw;
    for (j = 0; j < cnt_pro; j++) {
      uint8x8x3_t y0 = vld3_u8(inptr0);  // d8 = y0y3y6y9.. d9 = y1y4y7...y
      uint16x8_t val0 = vmull_u8(y0.val[0], vb);

      uint16x8_t val0_1 = vmull_u8(y0.val[1], vg);

      val0 = vmlal_u8(val0, y0.val[2], vr);

      uint32x4_t v0_sum0 = vaddl_u16(vget_low_u16(val0_1), vget_low_u16(val0));
      uint32x4_t v0_sum1 =
          vaddl_u16(vget_high_u16(val0_1), vget_high_u16(val0));

      uint16x4_t v0_sum0_16 = vshrn_n_u32(v0_sum0, 7);
      uint16x4_t v0_sum1_16 = vshrn_n_u32(v0_sum1, 7);

      uint16x8_t v0_sum = vcombine_u16(v0_sum0_16, v0_sum1_16);

      uint8x8_t vout0 = vmovn_u16(v0_sum);

      inptr0 += 24;
      vst1_u8(outr0, vout0);
      outr0 += 8;
    }
    for (; j < srcw; j++) {
      *outr0++ = (inptr0[0] * b + inptr0[1] * g + inptr0[2] * r) >> 7;
      inptr0 += 3;
    }
  }
}
/*
采用CV_BGR2GRAY,转换公式Gray = 0.1140*B + 0.5870*G + 0.2989*R
采用CV_RGB2GRAY,转换公式Gray = 0.1140*R + 0.5870*G + 0.2989*B
b = 0.114 *128 = 14.529 = 15
g = 0.587 * 128 = 75.136 = 75
r = 0.2989 * 127 = 38.2592 = 38
Gray = (15*B + 75*G + 38*R)/128
bgra2gray, rgba2gray
*/
void hwc4_to_hwc1(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  uint8_t b = 15;
  uint8_t g = 75;
  uint8_t r = 38;

  uint8x8_t vb = vdup_n_u8(b);
  uint8x8_t vg = vdup_n_u8(g);
  uint8x8_t vr = vdup_n_u8(r);
  uint8_t vb_array[8] = {b, b, b, b, b, b, b, b};
  uint8_t vg_array[8] = {g, g, g, g, g, g, g, g};
  uint8_t vr_array[8] = {r, r, r, r, r, r, r, r};
  int cnt_pro = srcw >> 3;
  int remain_pro = srcw % 8;
  int win = srcw * 4;
  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, srch - 3, 0, 4) {
    int j = 0;
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;
    uint8_t* outr0 = dst + i * srcw;
    uint8_t* outr1 = outr0 + srcw;
    uint8_t* outr2 = outr1 + srcw;
    uint8_t* outr3 = outr2 + srcw;

    int cnt = cnt_pro;
    if (cnt > 0) {
#ifdef __aarch64__
      asm volatile(
          "prfm   pldl1keep, [%[inptr0]]                \n"
          "prfm   pldl1keep, [%[inptr0], #128]   \n"
          "prfm   pldl1keep, [%[inptr1]]        \n"
          "prfm   pldl1keep, [%[inptr1], #128]   \n"
          "prfm   pldl1keep, [%[inptr2]]                \n"
          "prfm   pldl1keep, [%[inptr2], #128]   \n"
          "prfm   pldl1keep, [%[inptr3]]                \n"
          "prfm   pldl1keep, [%[inptr3], #128]   \n"
          "ld1 {v21.8b}, [%[vb]]                 \n"
          "ld1 {v22.8b}, [%[vg]]                 \n"
          "ld1 {v23.8b}, [%[vr]]                 \n"
          "1: \n"
          "ld4 {v0.8b - v3.8b}, [%[inptr0]], #32 \n"    // d8 = y0y3y6y9.. d9 =
                                                        // y1y4y7...
          "ld4 {v4.8b - v7.8b}, [%[inptr1]], #32 \n"    // d8 = y0y3y6y9.. d9 =
                                                        // y1y4y7...
          "ld4 {v8.8b - v11.8b}, [%[inptr2]], #32 \n"   // d8 = y0y3y6y9.. d9 =
                                                        // y1y4y7...
          "ld4 {v12.8b - v15.8b}, [%[inptr3]], #32 \n"  // d8 = y0y3y6y9.. d9 =
                                                        // y1y4y7...
          // mul b
          "umull v16.8h, v0.8b, v21.8b \n"   // v0 * vb
          "umull v17.8h, v4.8b, v21.8b \n"   // v0 * vb
          "umull v18.8h, v8.8b, v21.8b \n"   // v0 * vb
          "umull v19.8h, v12.8b, v21.8b \n"  // v0 * vb
          // mul g
          "umull v20.8h, v1.8b, v22.8b \n"   // v0 * vb
          "umull v24.8h, v5.8b, v22.8b \n"   // v0 * vb
          "umull v25.8h, v9.8b, v22.8b \n"   // v0 * vb
          "umull v26.8h, v13.8b, v22.8b \n"  // v0 * vb
          // mul r
          "umlal v16.8h, v2.8b, v23.8b \n"   // v0 * vb
          "umlal v17.8h, v6.8b, v23.8b \n"   // v0 * vb
          "umlal v18.8h, v10.8b, v23.8b \n"  // v0 * vb
          "umlal v19.8h, v14.8b, v23.8b \n"  // v0 * vb
          // 16->32
          "uaddl v0.4s, v20.4h, v16.4h \n"
          "uaddl2 v1.4s, v20.8h, v16.8h \n"
          "uaddl v2.4s, v24.4h, v17.4h \n"
          "uaddl2 v3.4s, v24.8h, v17.8h \n"
          "uaddl v4.4s, v25.4h, v18.4h \n"
          "uaddl2 v5.4s, v25.8h, v18.8h \n"
          "uaddl v6.4s, v26.4h, v19.4h \n"
          "uaddl2 v7.4s, v26.8h, v19.8h \n"
          // 32->16 v0 >> 7
          "shrn v12.4h, v0.4s, #7 \n"
          "shrn2 v12.8h, v1.4s, #7 \n"
          "shrn v13.4h, v2.4s, #7 \n"
          "shrn2 v13.8h, v3.4s, #7 \n"
          "shrn v14.4h, v4.4s, #7 \n"
          "shrn2 v14.8h, v5.4s, #7 \n"
          "shrn v15.4h, v6.4s, #7 \n"
          "shrn2 v15.8h, v7.4s, #7 \n"
          // 16->8
          "xtn v0.8b, v12.8h \n"
          "xtn v1.8b, v13.8h \n"
          "xtn v2.8b, v14.8h \n"
          "xtn v3.8b, v15.8h \n"
          "subs %w[cnt], %w[cnt], #1 \n"
          "st1 {v0.8b}, [%[outr0]], #8 \n"
          "st1 {v1.8b}, [%[outr1]], #8 \n"
          "st1 {v2.8b}, [%[outr2]], #8 \n"
          "st1 {v3.8b}, [%[outr3]], #8 \n"
          "bne 1b \n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outr0] "+r"(outr0),
            [outr1] "+r"(outr1),
            [outr2] "+r"(outr2),
            [outr3] "+r"(outr3),
            [cnt] "+r"(cnt)
          : [vb] "r"(vb_array), [vg] "r"(vg_array), [vr] "r"(vr_array)
          : "cc",
            "memory",
            "v0",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15",
            "v16",
            "v17",
            "v18",
            "v19",
            "v20",
            "v21",
            "v22",
            "v23",
            "v24",
            "v25",
            "v26");
#else
      asm volatile(
          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr0], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr1]]            @ preload a, 64byte\n"
          "pld [%[inptr1], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr2]]            @ preload a, 64byte\n"
          "pld [%[inptr2], #128]                         @ preload a, 64byte\n"
          "pld [%[inptr3]]            @ preload a, 64byte\n"
          "pld [%[inptr3], #128]                         @ preload a, 64byte\n"
          "vld1.8 d0, [%[vb]] \n"
          "vld1.8 d1, [%[vg]] \n"
          "vld1.8 d2, [%[vr]] \n"
          "1: \n"
          "vld4.8 {d3, d4, d5, d6}, [%[inptr0]]! \n"
          "vld4.8 {d7, d8, d9, d10}, [%[inptr1]]! \n"
          "vld4.8 {d11, d12, d13, d14}, [%[inptr2]]! \n"
          "vld4.8 {d15, d16, d17, d18}, [%[inptr3]]! \n"
          // vb
          "vmull.u8 q10, d3, d0 \n"
          "vmull.u8 q11, d7, d0 \n"
          "vmull.u8 q12, d11, d0 \n"
          "vmull.u8 q13, d15, d0 \n"
          // vg
          "vmull.u8 q14, d4, d1 \n"
          "vmull.u8 q15, d8, d1 \n"
          "vmull.u8 q5, d12, d1 \n"
          "vmull.u8 q7, d16, d1 \n"
          // vr
          "vmlal.u8 q10, d5, d2 \n"
          "vmlal.u8 q11, d9, d2 \n"
          "vmlal.u8 q12, d13, d2 \n"
          "vmlal.u8 q13, d17, d2 \n"
          // 16->32
          "vaddl.u16 q2, d28, d20 \n"
          "vaddl.u16 q3, d29, d21 \n"
          "vaddl.u16 q4, d30, d22 \n"
          "vaddl.u16 q10, d31, d23 \n"
          "vaddl.u16 q6, d10, d24 \n"
          "vaddl.u16 q11, d11, d25 \n"
          "vaddl.u16 q8, d14, d26 \n"
          "vaddl.u16 q9, d15, d27 \n"
          // 32->16 q2 >> 7
          "vshrn.u32  d10, q2, #7 \n"
          "vshrn.u32 d11, q3, #7 \n"
          "vshrn.u32 d14, q4, #7 \n"
          "vshrn.u32 d15, q10, #7 \n"
          "vshrn.u32 d24, q6, #7 \n"
          "vshrn.u32 d25, q11, #7 \n"
          "vshrn.u32 d26, q8, #7 \n"
          "vshrn.u32 d27, q9, #7 \n"
          // 16->8
          "vmovn.u16 d4, q5 \n"
          "vmovn.u16 d5, q7 \n"
          "vmovn.u16 d6, q12 \n"
          "vmovn.u16 d7, q13 \n"
          "subs %[cnt], #1 \n"
          // store
          "vst1.8 d4, [%[outr0]]! \n"
          "vst1.8 d5, [%[outr1]]! \n"
          "vst1.8 d6, [%[outr2]]! \n"
          "vst1.8 d7, [%[outr3]]! \n"
          "bne 1b \n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outr0] "+r"(outr0),
            [outr1] "+r"(outr1),
            [outr2] "+r"(outr2),
            [outr3] "+r"(outr3),
            [cnt] "+r"(cnt)
          : [vb] "r"(vb_array), [vg] "r"(vg_array), [vr] "r"(vr_array)
          : "cc",
            "memory",
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12",
            "q13",
            "q14",
            "q15");
#endif
    }
    for (; j < remain_pro; j++) {
      *outr0++ = (inptr0[0] * b + inptr0[1] * g + inptr0[2] * r) >> 7;
      *outr1++ = (inptr1[0] * b + inptr1[1] * g + inptr1[2] * r) >> 7;
      *outr2++ = (inptr2[0] * b + inptr2[1] * g + inptr2[2] * r) >> 7;
      *outr3++ = (inptr3[0] * b + inptr3[1] * g + inptr3[2] * r) >> 7;
      inptr0 += 4;
      inptr1 += 4;
      inptr2 += 4;
      inptr3 += 4;
    }
  }
  LITE_PARALLEL_COMMON_END();
  for (; i < srch; i++) {
    int j = 0;
    const uint8_t* inptr0 = src + i * win;
    uint8_t* outr0 = dst + i * srcw;
    for (j = 0; j < cnt_pro; j++) {
      uint8x8x4_t y0 = vld4_u8(inptr0);  // d8 = y0y3y6y9.. d9 = y1y4y7...y
      uint16x8_t val0 = vmull_u8(y0.val[0], vb);

      uint16x8_t val0_1 = vmull_u8(y0.val[1], vg);

      val0 = vmlal_u8(val0, y0.val[2], vr);

      uint32x4_t v0_sum0 = vaddl_u16(vget_low_u16(val0_1), vget_low_u16(val0));
      uint32x4_t v0_sum1 =
          vaddl_u16(vget_high_u16(val0_1), vget_high_u16(val0));

      uint16x4_t v0_sum0_16 = vshrn_n_u32(v0_sum0, 7);
      uint16x4_t v0_sum1_16 = vshrn_n_u32(v0_sum1, 7);

      uint16x8_t v0_sum = vcombine_u16(v0_sum0_16, v0_sum1_16);

      uint8x8_t vout0 = vmovn_u16(v0_sum);

      inptr0 += 32;
      vst1_u8(outr0, vout0);
      outr0 += 8;
    }
    for (; j < srcw; j++) {
      *outr0++ = (inptr0[0] * b + inptr0[1] * g + inptr0[2] * r) >> 7;
      inptr0 += 4;
    }
  }
}
/*
采用CV_GRAY2BGR,转换公式B = G = R = Gray
采用CV_GRAY2RGB,转换公式R = G = B = Gray
gray2bgr, gray2rgb
*/
void hwc1_to_hwc3(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = *src;
      *dst++ = *src;
      *dst++ = *src;
      src++;
    }
  }
}
/*
采用CV_GRAY2BGRA,转换公式B = G = R = Gray A=255
采用CV_GRAY2RGBA,转换公式R = G = B = Gray A=255
gray2bgra, gray2rgba
*/
void hwc1_to_hwc4(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = *src;
      *dst++ = *src;
      *dst++ = *src;
      *dst++ = 255;
      src++;
    }
  }
}
// bgr2bgra, rgb2rgba
void hwc3_to_hwc4(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = 255;
    }
  }
}
// bgra2bgr, rgba2rgb
void hwc4_to_hwc3(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      src++;
    }
  }
}
// bgr2rgb, rgb2bgr
void hwc3_trans(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = src[2];  // r
      *dst++ = src[1];  // g
      *dst++ = src[0];  // b
      src += 3;
    }
  }
}
// bgra2rgba, rgba2bgra
void hwc4_trans(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = src[2];  // r
      *dst++ = src[1];  // g
      *dst++ = src[0];  // b
      *dst++ = src[3];  // a
      src += 4;
    }
  }
}
// bgra2rgb, rgba2bgr
void hwc4_trans_hwc3(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = src[2];  // r
      *dst++ = src[1];  // g
      *dst++ = src[0];  // b
      // *dst++ = src[4];//a
      src += 4;
    }
  }
}
// bgr2rgba, rgb2bga
void hwc3_trans_hwc4(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = src[2];  // r
      *dst++ = src[1];  // g
      *dst++ = src[0];  // b
      *dst++ = 255;     // a
      src += 3;
    }
  }
}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
