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

#include "lite/utils/cv/image_preprocess.h"
#include <arm_neon.h>
#include <math.h>
#include <algorithm>
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
// init
ImagePreprocess::ImagePreprocess(ImageFormat srcFormat,
                                 ImageFormat dstFormat,
                                 TransParam param) {
  this->srcFormat_ = srcFormat;
  this->dstFormat_ = dstFormat;
  this->transParam_ = param;
}
void ImagePreprocess::imageCovert(const uint8_t* src,
                                  uint8_t* dst,
                                  ImageFormat srcFormat,
                                  ImageFormat dstFormat) {
  ImageConvert img_convert;
  img_convert.choose(src,
                     dst,
                     srcFormat,
                     dstFormat,
                     this->transParam_.iw,
                     this->transParam_.ih);
}

void compute_xy(int srcw,
                int srch,
                int dstw,
                int dsth,
                double scale_x,
                double scale_y,
                int* xofs,
                int* yofs,
                int16_t* ialpha,
                int16_t* ibeta);

void ImagePreprocess::imageResize(const uint8_t* src,
                                  uint8_t* dst,
                                  ImageFormat srcFormat,
                                  int srcw,
                                  int srch,
                                  int dstw,
                                  int dsth) {
  int size = srcw * srch;
  if (srcw == dstw && srch == dsth) {
    if (srcFormat == NV12 || srcFormat == NV21) {
      size = srcw * (floor(1.5 * srch));
    } else if (srcFormat == BGR || srcFormat == RGB) {
      size = 3 * srcw * srch;
    } else if (srcFormat == BGRA || srcFormat == RGBA) {
      size = 4 * srcw * srch;
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

  compute_xy(
      srcw, srch, dstw, dsth, scale_x, scale_y, xofs, yofs, ialpha, ibeta);

  int w_out = dstw;
  int w_in = srcw;
  int num = 1;
  int orih = dsth;
  if (srcFormat == GRAY) {
    num = 1;
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    num = 1;
    int hout = static_cast<int>(0.5 * dsth);
    dsth += hout;
  } else if (srcFormat == BGR || srcFormat == RGB) {
    w_in = srcw * 3;
    w_out = dstw * 3;
    num = 3;

  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    w_in = srcw * 4;
    w_out = dstw * 4;
    num = 4;
  }

  int* xofs1 = nullptr;
  int* yofs1 = nullptr;
  int16_t* ialpha1 = nullptr;
  if (orih < dsth) {  // uv
    int tmp = dsth - orih;
    int w = dstw / 2;
    xofs1 = new int[w];
    yofs1 = new int[tmp];
    ialpha1 = new int16_t[srcw];
    compute_xy(srcw / 2,
               srch / 2,
               w,
               tmp,
               scale_x,
               scale_y,
               xofs1,
               yofs1,
               ialpha1,
               ibeta + orih);
  }
  int cnt = w_out >> 3;
  int remain = w_out % 8;
  int32x4_t _v2 = vdupq_n_s32(2);
#pragma omp parallel for
  for (int dy = 0; dy < dsth; dy++) {
    int16_t* rowsbuf0 = new int16_t[w_out];
    int16_t* rowsbuf1 = new int16_t[w_out];
    int sy = yofs[dy];
    if (dy >= orih) {
      xofs = xofs1;
      yofs = yofs1;
      ialpha = ialpha1;
    }
    if (sy < 0) {
      memset(rowsbuf0, 0, sizeof(uint16_t) * w_out);
      const uint8_t* S1 = src + srcw * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rowsbuf1;
      for (int dx = 0; dx < dstw; dx++) {
        int sx = xofs[dx] * num;  // num = 4
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1pl = S1 + sx;
        const uint8_t* S1pr = S1 + sx + num;
        if (sx < 0) {
          S1pl = S1;
        }
        for (int i = 0; i < num; i++) {
          if (sx < 0) {
            *rows1p++ = ((*S1pl++) * a1) >> 4;
          } else {
            *rows1p++ = ((*S1pl++) * a0 + (*S1pr++) * a1) >> 4;
          }
        }
        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + w_in * (sy);
      const uint8_t* S1 = src + w_in * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rowsbuf0;
      int16_t* rows1p = rowsbuf1;
      for (int dx = 0; dx < dstw; dx++) {
        int sx = xofs[dx] * num;  // num = 4
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0pl = S0 + sx;
        const uint8_t* S0pr = S0 + sx + num;
        const uint8_t* S1pl = S1 + sx;
        const uint8_t* S1pr = S1 + sx + num;
        if (sx < 0) {
          S0pl = S0;
          S1pl = S1;
        }
        for (int i = 0; i < num; i++) {
          if (sx < 0) {
            *rows0p = ((*S0pl++) * a1) >> 4;
            *rows1p = ((*S1pl++) * a1) >> 4;
            rows0p++;
            rows1p++;
          } else {
            *rows0p++ = ((*S0pl++) * a0 + (*S0pr++) * a1) >> 4;
            *rows1p++ = ((*S1pl++) * a0 + (*S1pr++) * a1) >> 4;
          }
        }
        ialphap += 2;
      }
    }
    int ind = dy * 2;
    int16_t b0 = ibeta[ind];
    int16_t b1 = ibeta[ind + 1];
    int16x8_t _b0 = vdupq_n_s16(b0);
    int16x8_t _b1 = vdupq_n_s16(b1);
    uint8_t* dp_ptr = dst + dy * w_out;
    int16_t* rows0p = rowsbuf0;
    int16_t* rows1p = rowsbuf1;
    int re_cnt = cnt;
    if (re_cnt > 0) {
#ifdef __aarch64__
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
            [cnt] "+r"(re_cnt),
            [dp] "+r"(dp_ptr)
          : [_b0] "w"(_b0), [_b1] "w"(_b1), [_v2] "w"(_v2)
          : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
      asm volatile(
          "mov        r4, #2          \n"
          "vdup.s32   q12, r4         \n"
          "0:                         \n"
          "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
          "vld1.s16   {d6-d7}, [%[rows1p]]!\n"
          "vorr.s32   q10, q12, q12   \n"
          "vorr.s32   q11, q12, q12   \n"

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
          "subs       %[cnt], #1          \n"
          "vqmovun.s16 d20, q10        \n"
          "vst1.8     {d20}, [%[dp]]!    \n"
          "bne        0b              \n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [cnt] "+r"(re_cnt),
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

#endif  // __aarch64__
    }
    for (int i = 0; i < remain; i++) {
      //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >>
      //             INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
  }
  delete[] buf;
}
void ImagePreprocess::imageResize(const uint8_t* src,
                                  uint8_t* dst,
                                  ImageFormat srcFormat) {
  int srcw = this->transParam_.iw;
  int srch = this->transParam_.ih;
  int dstw = this->transParam_.ow;
  int dsth = this->transParam_.oh;
  ImagePreprocess::imageResize(src, dst, srcw, srch, dstw, dsth);
}
void ImagePreprocess::imageFlip(const uint8_t* src,
                                uint8_t* dst,
                                ImageFormat srcFormat,
                                int srcw,
                                int srch,
                                FlipParam flip_param) {
  if (srcFormat == GRAY) {
    flip_hwc1(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    flip_hwc3(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    flip_hwc4(src, dst, srcw, srch, flip_param);
  }
}
void ImagePreprocess::imageFlip(const uint8_t* src,
                                uint8_t* dst,
                                ImageFormat srcFormat,
                                int srcw,
                                int srch,
                                FlipParam flip_param) {
  if (srcFormat == GRAY) {
    flip_hwc1(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    flip_hwc3(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    flip_hwc4(src, dst, srcw, srch, flip_param);
  }
}
void ImagePreprocess::imageFlip(const uint8_t* src, uint8_t* dst) {
  auto srcw = this->transParam_.ow;
  auto srch = this->transParam_.oh, auto srcFormat = this->dstFormat_;
  auto flip_param = this->transParam_.flip_param;
  if (srcFormat == GRAY) {
    flip_hwc1(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    flip_hwc3(src, dst, srcw, srch, flip_param);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    flip_hwc4(src, dst, srcw, srch, flip_param);
  }
}

void ImagePreprocess::imageRotate(const uint8_t* src,
                                  uint8_t* dst,
                                  ImageFormat srcFormat,
                                  int srcw,
                                  int srch,
                                  float degree) {
  if (srcFormat == GRAY) {
    rotate_hwc1(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    rotate_hwc3(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    rotate_hwc4(src, dst, srcw, srch, degree);
  }
}

void ImagePreprocess::imageRotate(const uint8_t* src, uint8_t* dst) {
  auto srcw = this->transParam_.ow;
  auto srch = this->transParam_.oh, auto srcFormat = this->dstFormat_;
  auto rotate_param = this->transParam_.rotate_param;
  if (srcFormat == GRAY) {
    rotate_hwc1(src, dst, srcw, srch, rotate_param);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    rotate_hwc3(src, dst, srcw, srch, rotate_param);
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    rotate_hwc4(src, dst, srcw, srch, rotate_param);
  }
}

void ImagePreprocess::image2Tensor(const uint8_t* src,
                                   Tensor* dstTensor,
                                   LayOut layout,
                                   float* means,
                                   float* scales) {
  Image2Tensor img2tensor;
  img2tensor.choose(src,
                    dstTensor,
                    this->dstFormat_,
                    layout,
                    this->transParam_.ow,
                    this->transParam_.oh,
                    means,
                    scales);
}
// compute xofs, yofs, alpha, beta
void compute_xy(int srcw,
                int srch,
                int dstw,
                int dsth,
                double scale_x,
                double scale_y,
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

  for (int dx = 0; dx < dstw; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= srcw - 1) {
      sx = srcw - 2;
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
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
