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

#include "lite/core/parallel_defines.h"
#include "lite/utils/cv/paddle_image_preprocess.h"

typedef paddle::lite::utils::cv::ImageFormat ImageFormat;
typedef paddle::lite::utils::cv::FlipParam FlipParam;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite_api::DataLayoutType LayoutType;

void nv2bgr(const uint8_t* in_data,
            uint8_t* out_data,
            int srcw,
            int srch,
            int v_num,
            int u_num) {
  int size = srch * srcw;
  const uint8_t* y_ptr = in_data;
  const uint8_t* uv_ptr = in_data + size;
  for (int i = 0; i < srch; i++) {
    int j = 0;
    const uint8_t* ptr_y1 = y_ptr + i * srcw;
    const uint8_t* ptr_vu = uv_ptr + (i / 2) * srcw;
    uint8_t* ptr_bgr1 = out_data + i * 3 * srcw;
    for (; j < srcw; j += 2) {
      uint8_t _y0 = ptr_y1[0];
      uint8_t _y1 = ptr_y1[1];
      uint8_t _v = ptr_vu[v_num];
      uint8_t _u = ptr_vu[u_num];

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

      *ptr_bgr1++ = b1;
      *ptr_bgr1++ = g1;
      *ptr_bgr1++ = r1;

      ptr_y1 += 2;
      ptr_vu += 2;
    }
    if (j < srcw) {
      uint8_t _y = ptr_y1[0];
      uint8_t _v = ptr_vu[v_num];
      uint8_t _u = ptr_vu[u_num];

      int r = _y + ((179 * (_v - 128)) >> 7);
      int g = _y - ((44 * (_u - 128) - 91 * (_v - 128)) >> 7);
      int b = _y + ((227 * (_u - 128)) >> 7);

      r = r < 0 ? 0 : (r > 255) ? 255 : r;
      g = g < 0 ? 0 : (g > 255) ? 255 : g;
      b = b < 0 ? 0 : (b > 255) ? 255 : b;

      ptr_bgr1[0] = b;
      ptr_bgr1[1] = g;
      ptr_bgr1[2] = r;
    }
  }
}
void nv2bgra(const uint8_t* in_data,
             uint8_t* out_data,
             int srcw,
             int srch,
             int v_num,
             int u_num) {
  int size = srch * srcw;
  const uint8_t* y_ptr = in_data;
  const uint8_t* uv_ptr = in_data + size;
  for (int i = 0; i < srch; i++) {
    int j = 0;
    const uint8_t* ptr_y1 = y_ptr + i * srcw;
    const uint8_t* ptr_vu = uv_ptr + (i / 2) * srcw;
    uint8_t* ptr_bgr1 = out_data + i * 4 * srcw;
    for (; j < srcw; j += 2) {
      uint8_t _y0 = ptr_y1[0];
      uint8_t _y1 = ptr_y1[1];
      uint8_t _v = ptr_vu[v_num];
      uint8_t _u = ptr_vu[u_num];

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

      *ptr_bgr1++ = b1;
      *ptr_bgr1++ = g1;
      *ptr_bgr1++ = r1;
      *ptr_bgr1++ = 255;

      ptr_y1 += 2;
      ptr_vu += 2;
    }
    if (j < srcw) {
      uint8_t _y = ptr_y1[0];
      uint8_t _v = ptr_vu[v_num];
      uint8_t _u = ptr_vu[u_num];

      int r = _y + ((179 * (_v - 128)) >> 7);
      int g = _y - ((44 * (_u - 128) - 91 * (_v - 128)) >> 7);
      int b = _y + ((227 * (_u - 128)) >> 7);

      r = r < 0 ? 0 : (r > 255) ? 255 : r;
      g = g < 0 ? 0 : (g > 255) ? 255 : g;
      b = b < 0 ? 0 : (b > 255) ? 255 : b;

      ptr_bgr1[0] = b;
      ptr_bgr1[1] = g;
      ptr_bgr1[2] = r;
      ptr_bgr1[3] = 255;
    }
  }
}

void nv12_bgr_basic(const uint8_t* in_data,
                    uint8_t* out_data,
                    int srcw,
                    int srch) {
  nv2bgr(in_data, out_data, srcw, srch, 1, 0);
}

void nv21_bgr_basic(const uint8_t* in_data,
                    uint8_t* out_data,
                    int srcw,
                    int srch) {
  nv2bgr(in_data, out_data, srcw, srch, 0, 1);
}
void nv12_bgra_basic(const uint8_t* in_data,
                     uint8_t* out_data,
                     int srcw,
                     int srch) {
  nv2bgra(in_data, out_data, srcw, srch, 1, 0);
}

void nv21_bgra_basic(const uint8_t* in_data,
                     uint8_t* out_data,
                     int srcw,
                     int srch) {
  nv2bgra(in_data, out_data, srcw, srch, 0, 1);
}

/*
采用CV_BGR2GRAY,转换公式Gray = 0.1140*B + 0.5870*G + 0.2989*R
采用CV_RGB2GRAY,转换公式Gray = 0.1140*R + 0.5870*G + 0.2989*B
b = 0.114 *128 = 14.529 = 15
g = 0.587 * 128 = 75.136 = 75
r = 0.2989 * 128 = 38.2592 = 38
Gray = (15*B + 75*G + 38*R)/128
bgr2gray, rgb2gray
*/
void bgr_gray_basic(const uint8_t* in_data,
                    uint8_t* out_data,
                    int srcw,
                    int srch) {
  for (int i = 0; i < srch; i++) {
    const uint8_t* din_ptr = in_data + i * 3 * srcw;
    uint8_t* dout_ptr = out_data + i * srcw;
    for (int j = 0; j < srcw; j++) {
      int sum = din_ptr[0] * 15 + din_ptr[1] * 75 + din_ptr[2] * 38;
      sum = sum >> 7;
      *dout_ptr++ = sum;
      din_ptr += 3;
    }
  }
}
void bgra_gray_basic(const uint8_t* in_data,
                     uint8_t* out_data,
                     int srcw,
                     int srch) {
  for (int i = 0; i < srch; i++) {
    const uint8_t* din_ptr = in_data + i * 4 * srcw;
    uint8_t* dout_ptr = out_data + i * srcw;
    for (int j = 0; j < srcw; j++) {
      int sum = din_ptr[0] * 15 + din_ptr[1] * 75 + din_ptr[2] * 38;
      sum = sum >> 7;
      *dout_ptr++ = sum;
      din_ptr += 4;
    }
  }
}

void gray_bgr_basic(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
  for (int i = 0; i < srch; i++) {
    for (int j = 0; j < srcw; j++) {
      *dst++ = *src;
      *dst++ = *src;
      *dst++ = *src;
      src++;
    }
  }
}
void gray_bgra_basic(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
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
void hwc3_to_hwc4_basic(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
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
void hwc4_to_hwc3_basic(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
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
void hwc3_trans_basic(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
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
void hwc4_trans_basic(const uint8_t* src, uint8_t* dst, int srcw, int srch) {
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
void hwc4_trans_hwc3_basic(const uint8_t* src,
                           uint8_t* dst,
                           int srcw,
                           int srch) {
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
void hwc3_trans_hwc4_basic(const uint8_t* src,
                           uint8_t* dst,
                           int srcw,
                           int srch) {
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
void image_convert_basic(const uint8_t* in_data,
                         uint8_t* out_data,
                         ImageFormat srcFormat,
                         ImageFormat dstFormat,
                         int srcw,
                         int srch,
                         int out_size) {
  if (srcFormat == dstFormat) {
    // copy
    memcpy(out_data, in_data, sizeof(uint8_t) * out_size);
    return;
  } else {
    if (srcFormat == ImageFormat::NV12 &&
        (dstFormat == ImageFormat::BGR || dstFormat == ImageFormat::RGB)) {
      nv12_bgr_basic(in_data, out_data, srcw, srch);
    } else if (srcFormat == ImageFormat::NV21 &&
               (dstFormat == ImageFormat::BGR ||
                dstFormat == ImageFormat::RGB)) {
      nv21_bgr_basic(in_data, out_data, srcw, srch);
    } else if (srcFormat == ImageFormat::NV12 &&
               (dstFormat == ImageFormat::BGRA ||
                dstFormat == ImageFormat::RGBA)) {
      nv12_bgra_basic(in_data, out_data, srcw, srch);
    } else if (srcFormat == ImageFormat::NV21 &&
               (dstFormat == ImageFormat::BGRA ||
                dstFormat == ImageFormat::RGBA)) {
      nv21_bgra_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGB &&
                dstFormat == ImageFormat::GRAY) ||
               (srcFormat == ImageFormat::BGR &&
                dstFormat == ImageFormat::GRAY)) {
      bgr_gray_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::GRAY &&
                dstFormat == ImageFormat::RGB) ||
               (srcFormat == ImageFormat::GRAY &&
                dstFormat == ImageFormat::BGR)) {
      gray_bgr_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGBA &&
                dstFormat == ImageFormat::GRAY) ||
               (srcFormat == ImageFormat::BGRA &&
                dstFormat == ImageFormat::GRAY)) {
      bgra_gray_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::GRAY &&
                dstFormat == ImageFormat::RGBA) ||
               (srcFormat == ImageFormat::GRAY &&
                dstFormat == ImageFormat::BGRA)) {
      gray_bgra_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGBA &&
                dstFormat == ImageFormat::RGB) ||
               (srcFormat == ImageFormat::BGRA &&
                dstFormat == ImageFormat::BGR)) {
      hwc4_to_hwc3_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGB &&
                dstFormat == ImageFormat::RGBA) ||
               (srcFormat == ImageFormat::BGR &&
                dstFormat == ImageFormat::BGRA)) {
      hwc3_to_hwc4_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGB &&
                dstFormat == ImageFormat::BGR) ||
               (srcFormat == ImageFormat::BGR &&
                dstFormat == ImageFormat::RGB)) {
      hwc3_trans_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGBA &&
                dstFormat == ImageFormat::BGRA) ||
               (srcFormat == ImageFormat::BGRA &&
                dstFormat == ImageFormat::RGBA)) {
      hwc4_trans_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGBA &&
                dstFormat == ImageFormat::BGR) ||
               (srcFormat == ImageFormat::BGRA &&
                dstFormat == ImageFormat::RGB)) {
      hwc4_trans_hwc3_basic(in_data, out_data, srcw, srch);
    } else if ((srcFormat == ImageFormat::RGB &&
                dstFormat == ImageFormat::BGRA) ||
               (srcFormat == ImageFormat::BGR &&
                dstFormat == ImageFormat::RGBA)) {
      hwc3_trans_hwc4_basic(in_data, out_data, srcw, srch);
    } else {
      printf("srcFormat: %d, dstFormat: %d does not support! \n",
             srcFormat,
             dstFormat);
    }
    // for (int i = 0; i < out_size; i++){
    //     printf("%d  ", *out_data++);
    //     if ((i+1) % 10 == 0){
    //         printf("\n");
    //     }
    // }
  }
}

void compute_xy(int srcw,
                int srch,
                int dstw,
                int dsth,
                double scale_x,
                double scale_y,
                int* xofs,
                int* yofs,
                float* ialpha,
                float* ibeta) {
  float fy = 0.f;
  float fx = 0.f;
  int sy = 0;
  int sx = 0;
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;
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

    float a0 = (1.f - fx);
    float a1 = fx;

    ialpha[dx * 2] = a0;
    ialpha[dx * 2 + 1] = a1;
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

    float b0 = (1.f - fy);
    float b1 = fy;

    ibeta[dy * 2] = b0;
    ibeta[dy * 2 + 1] = b1;
  }
}
void image_resize_basic(const uint8_t* in_data,
                        uint8_t* out_data,
                        ImageFormat srcFormat,
                        int srcw,
                        int srch,
                        int dstw,
                        int dsth) {
  int size = srcw * srch;
  if (srcw == dstw && srch == dsth) {
    if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
      size = srcw * (static_cast<int>(1.5 * srch));
    } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
      size = 3 * srcw * srch;
    } else if (srcFormat == ImageFormat::BGRA ||
               srcFormat == ImageFormat::RGBA) {
      size = 4 * srcw * srch;
    }
    memcpy(out_data, in_data, sizeof(uint8_t) * size);
    return;
  }
  double scale_x = static_cast<double>(srcw) / dstw;
  double scale_y = static_cast<double>(srch) / dsth;

  int* buf = new int[dstw + dsth];

  int* xofs = buf;
  int* yofs = buf + dstw;
  float* ialpha = new float[dstw * 2];
  float* ibeta = new float[dsth * 3];

  int w_in = srcw;
  int w_out = dstw;
  int num = 1;
  int orih = dsth;

  compute_xy(
      srcw, srch, dstw, dsth, scale_x, scale_y, xofs, yofs, ialpha, ibeta);
  if (srcFormat == ImageFormat::GRAY) {
    num = 1;
  } else if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
    int hout = static_cast<int>(0.5 * dsth);
    // uv todo
    w_out = dstw;
    num = 1;
    dsth += hout;
  } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    w_in = srcw * 3;
    w_out = dstw * 3;
    num = 3;
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    w_in = srcw * 4;
    w_out = dstw * 4;
    num = 4;
  }
  float* ialpha1 = nullptr;
  int* xofs1 = nullptr;
  int* yofs1 = nullptr;
  if (orih < dsth) {
    int tmp = dsth - orih;
    ialpha1 = new float[dstw];
    xofs1 = new int[dstw];
    yofs1 = new int[tmp];
    compute_xy(srcw,
               srch / 2,
               dstw / 2,
               tmp,
               scale_x,
               scale_y,
               xofs1,
               yofs1,
               ialpha1,
               ibeta + orih * 2);
  }
  LITE_PARALLEL_BEGIN(dy, tid, dsth) {
    uint8_t* out_ptr = out_data + dy * w_out;
    int y_in_start = yofs[dy];
    int y_flag = 0;

    float b0 = ibeta[dy * 2];
    float b1 = ibeta[dy * 2 + 1];
    if (dy >= orih) {
      num = 2;  // uv
      ialpha = ialpha1;
      xofs = xofs1;
      yofs = yofs1;
      y_in_start = yofs[dy - orih] + srch;
    }
    int y_in_end = y_in_start + 1;
    if (y_in_start < 0) {
      y_flag = 1;
      y_in_end = 0;
    }
    for (int dx = 0; dx < w_out; dx += num) {
      int tmp = dx / num;
      int x_in_start = xofs[tmp] * num;  // 0
      int x_in_end = x_in_start + num;   // 2
      int x_flag = 0;
      if (x_in_start < 0) {
        x_flag = 1;
        x_in_end = 0;
      }
      float a0 = ialpha[tmp * 2];
      float a1 = ialpha[tmp * 2 + 1];
      int tl_index = y_in_start * w_in + x_in_start;  // 0
      int tr_index = y_in_start * w_in + x_in_end;    // 2
      int bl_index = y_in_end * w_in + x_in_start;
      int br_index = y_in_end * w_in + x_in_end;
      int ind = dx;
      for (int i = 0; i < num; i++) {
        int tl = in_data[tl_index];
        int tr = in_data[tr_index];
        int bl = in_data[bl_index];
        int br = in_data[br_index];
        if (y_flag == 1) {
          tl = 0;
          tr = 0;
        }
        if (x_flag == 1) {
          tl = 0;
          bl = 0;
        }
        tl_index++;
        tr_index++;
        bl_index++;
        br_index++;
        float outval = (tl * a0 + tr * a1) * b0 + (bl * a0 + br * a1) * b1;
        out_ptr[ind++] = ceil(outval);
      }
    }
  }
  LITE_PARALLEL_END();
}

void rotate90_basic(const uint8_t* in_data,
                    int h_in,
                    int w_in,
                    uint8_t* out_data,
                    int h_out,
                    int w_out,
                    int num) {
  int win = w_in * num;
  int wout = w_out * num;
  for (int x = 0; x < h_in; x++) {
    for (int y = 0; y < w_in; y++) {
      int tmpy = y * num;
      int tmpx = (w_out - 1 - x) * num;  // x
      for (int i = 0; i < num; i++) {
        out_data[y * wout + tmpx] = in_data[x * win + tmpy];
        tmpx++;
        tmpy++;
      }
    }
  }
}

void rotate180_basic(const uint8_t* in_data,
                     int h_in,
                     int w_in,
                     uint8_t* out_data,
                     int h_out,
                     int w_out,
                     int num) {
  int win = w_in * num;
  int h = h_in - 1;
  int w = win - 1;
  for (int x = 0; x < h_in; x++) {
    for (int y = 0; y < w_in; y++) {
      int tmpy = y * num;
      int tmp = tmpy + (num - 1);
      for (int i = 0; i < num; i++) {
        out_data[(h - x) * win + w - tmp] = in_data[x * win + tmpy];
        tmpy++;
        tmp--;
      }
    }
  }
}
void rotate270_basic(const uint8_t* in_data,
                     int h_in,
                     int w_in,
                     uint8_t* out_data,
                     int h_out,
                     int w_out,
                     int num) {
  int win = w_in * num;
  int wout = w_out * num;
  int h = h_out - 1;
  for (int x = 0; x < h_in; x++) {
    for (int y = 0; y < w_in; y++) {
      int tmpy = y * num;
      int tmpx = x * num;
      for (int i = 0; i < num; i++) {
        out_data[(h - y) * wout + tmpx] =
            in_data[x * win + tmpy];  // (y,x) = in(x,y)
        tmpx++;
        tmpy++;
      }
    }
  }
}

void image_rotate_basic(const uint8_t* in_data,
                        uint8_t* out_data,
                        ImageFormat srcFormat,
                        int srcw,
                        int srch,
                        float rotate) {
  int num = 1;
  if (srcFormat == ImageFormat::GRAY) {
    num = 1;
  } else if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
    num = 1;  // todo
    return;
  } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    num = 3;
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    num = 4;
  }
  if (rotate == 90) {
    rotate90_basic(in_data, srch, srcw, out_data, srcw, srch, num);
  } else if (rotate == 180) {
    rotate180_basic(in_data, srch, srcw, out_data, srch, srcw, num);
  } else if (rotate == 270) {
    rotate270_basic(in_data, srch, srcw, out_data, srcw, srch, num);
  }
}

void flipx_basic(
    const uint8_t* in_data, int h_in, int w_in, uint8_t* out_data, int num) {
  int h = h_in - 1;
  int w = w_in * num;
  for (int x = 0; x < h_in; x++) {
    for (int y = 0; y < w_in; y++) {
      int tmpy = y * num;
      for (int i = 0; i < num; i++) {
        out_data[(h - x) * w + tmpy] =
            in_data[x * w + tmpy];  // (y,x) = in(x,y)
        tmpy++;
      }
    }
  }
}

void flipy_basic(
    const uint8_t* in_data, int h_in, int w_in, uint8_t* out_data, int num) {
  int w = w_in * num - 1;
  for (int x = 0; x < h_in; x++) {
    for (int y = 0; y < w_in; y++) {
      int tmpy = y * num;
      int tmp = tmpy + (num - 1);
      for (int i = 0; i < num; i++) {
        out_data[x * w_in * num + w - tmp] =
            in_data[x * w_in * num + tmpy];  // (y,x) = in(x,y)
        tmpy++;
        tmp--;
      }
    }
  }
}

void flipxy_basic(
    const uint8_t* in_data, int h_in, int w_in, uint8_t* out_data, int num) {
  int win = w_in * num;
  int h = h_in - 1;
  int w = win - 1;
  for (int x = 0; x < h_in; x++) {
    for (int y = 0; y < w_in; y++) {
      int tmpy = y * num;
      int tmp = tmpy + (num - 1);
      for (int i = 0; i < num; i++) {
        out_data[(h - x) * win + w - tmp] =
            in_data[x * win + tmpy];  // (h-y,w-x) = in(x,y)
        tmpy++;
        tmp--;
      }
    }
  }
}

void image_flip_basic(const uint8_t* in_data,
                      uint8_t* out_data,
                      ImageFormat srcFormat,
                      int srcw,
                      int srch,
                      FlipParam flip) {
  int num = 1;
  if (srcFormat == ImageFormat::GRAY) {
    num = 1;
  } else if (srcFormat == ImageFormat::NV12 || srcFormat == ImageFormat::NV21) {
    num = 1;  // todo
    return;
  } else if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    num = 3;
  } else if (srcFormat == ImageFormat::BGRA || srcFormat == ImageFormat::RGBA) {
    num = 4;
  }
  // printf("image_flip_basic: %d \n", flip);
  if (flip == FlipParam::X) {
    flipx_basic(in_data, srch, srcw, out_data, num);
  } else if (flip == FlipParam::Y) {
    flipy_basic(in_data, srch, srcw, out_data, num);
  } else if (flip == FlipParam::XY) {
    flipxy_basic(in_data, srch, srcw, out_data, num);
  }
}
void gray_to_tensor_basic(const uint8_t* bgr,
                          float* output,
                          int width,
                          int height,
                          float* means,
                          float* scales,
                          int num) {
  int size = width * height;
  float mean_val = means[0];
  float scale_val = scales[0];

  for (int h = 0; h < height; h++) {
    const uint8_t* ptr_bgr = bgr + h * width * num;
    float* ptr_h = output + h * width;
    for (int i = 0; i < width; i++) {
      *ptr_h++ = (ptr_bgr[0] - mean_val) * scale_val;
      ptr_bgr += num;
    }
  }
}

void bgr_to_tensor_chw_basic(const uint8_t* bgr,
                             float* output,
                             int width,
                             int height,
                             float* means,
                             float* scales,
                             int num) {
  int size = width * height;
  float r_means = means[0];
  float g_means = means[1];
  float b_means = means[2];
  float r_scales = scales[0];
  float g_scales = scales[1];
  float b_scales = scales[2];

  for (int h = 0; h < height; h++) {
    const uint8_t* ptr_bgr = bgr + h * width * num;
    float* ptr_b = output + h * width;
    float* ptr_g = ptr_b + size;
    float* ptr_r = ptr_g + size;
    for (int i = 0; i < width; i++) {
      *ptr_b++ = (ptr_bgr[0] - b_means) * b_scales;
      *ptr_g++ = (ptr_bgr[1] - g_means) * g_scales;
      *ptr_r++ = (ptr_bgr[2] - r_means) * r_scales;
      ptr_bgr += num;
    }
  }
}

void bgr_to_tensor_hwc_basic(const uint8_t* bgr,
                             float* output,
                             int width,
                             int height,
                             float* means,
                             float* scales,
                             int num) {
  int size = width * height;
  float r_means = means[0];
  float g_means = means[1];
  float b_means = means[2];
  float r_scales = scales[0];
  float g_scales = scales[1];
  float b_scales = scales[2];

  for (int h = 0; h < height; h++) {
    const uint8_t* ptr_bgr = bgr + h * width * num;
    float* out_bgr = output + h * width * num;
    for (int i = 0; i < width; i++) {
      *out_bgr++ = (ptr_bgr[0] - b_means) * b_scales;
      *out_bgr++ = (ptr_bgr[1] - g_means) * g_scales;
      *out_bgr++ = (ptr_bgr[2] - r_means) * r_scales;
      ptr_bgr += num;
    }
  }
}

void image_to_tensor_basic(const uint8_t* in_data,
                           Tensor* dst,
                           ImageFormat srcFormat,
                           LayoutType layout,
                           int srcw,
                           int srch,
                           float* means,
                           float* scales) {
  float* output = dst->mutable_data<float>();
  if (layout == LayoutType::kNCHW &&
      (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB)) {
    bgr_to_tensor_chw_basic(in_data, output, srcw, srch, means, scales, 3);
  } else if (layout == LayoutType::kNHWC &&
             (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB)) {
    bgr_to_tensor_hwc_basic(in_data, output, srcw, srch, means, scales, 3);
  } else if (layout == LayoutType::kNCHW && (srcFormat == ImageFormat::BGRA ||
                                             srcFormat == ImageFormat::RGBA)) {
    bgr_to_tensor_chw_basic(in_data, output, srcw, srch, means, scales, 4);
  } else if (layout == LayoutType::kNHWC && (srcFormat == ImageFormat::BGRA ||
                                             srcFormat == ImageFormat::RGBA)) {
    bgr_to_tensor_hwc_basic(in_data, output, srcw, srch, means, scales, 4);
  } else if (srcFormat == ImageFormat::GRAY &&
             (layout == LayoutType::kNHWC || layout == LayoutType::kNCHW)) {
    gray_to_tensor_basic(in_data, output, srcw, srch, means, scales, 1);
  }
}
