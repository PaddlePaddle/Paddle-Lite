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
#include "lite/utils/cv/cv_enum.h"
typedef paddle::lite::utils::cv::ImageFormat ImageFormat;

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
      unsigned char _y0 = ptr_y1[0];
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
      unsigned char _y = ptr_y1[0];
      unsigned char _v = ptr_vu[v_num];
      unsigned char _u = ptr_vu[u_num];

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
    uint8_t* ptr_bgr1 = out_data + i * 3 * srcw;
    for (; j < srcw; j += 2) {
      unsigned char _y0 = ptr_y1[0];
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
      unsigned char _y = ptr_y1[0];
      unsigned char _v = ptr_vu[v_num];
      unsigned char _u = ptr_vu[u_num];

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
/*
采用CV_BGR2GRAY,转换公式Gray = 0.1140*B + 0.5870*G + 0.2989*R
采用CV_RGB2GRAY,转换公式Gray = 0.1140*R + 0.5870*G + 0.2989*B
b = 0.114 *128 = 14.529 = 15
g = 0.587 * 128 = 75.136 = 75
r = 0.2989 * 127 = 38.2592 = 38
Gray = (15*B + 75*G + 38*R)/128
bgr2gray, rgb2gray
*/
void bgr_gray_basic(const uint8_t* in_data,
                    uint8_t* out_data,
                    int srcw,
                    int srch) {
  int size = srch * srcw;
  const uint8_t* y_ptr = in_data;
  for (int i = 0; i < srch; i++) {
    const uint8_t* din_ptr = in_data + i * 3 * srcw;
    uint8_t* dout_ptr = out_data + i * srcw;
    for (int j = 0; j < srcw; j += 3) {
      int sum = din_ptr[j] * 15 + din_ptr[j + 1] * 75 + din_ptr[j + 2] * 38;
      sum = sum >> 7;
      *dout_ptr++ = sum;
      din_ptr += 3;
    }
  }
}

void gray_bgr_basic(const uint8_t* in_data,
                    uint8_t* out_data,
                    int srcw,
                    int srch) {
  int size = srch * srcw;
  const uint8_t* y_ptr = in_data;
  for (int i = 0; i < srch; i++) {
    const uint8_t* din_ptr = in_data + i * srcw;
    uint8_t* dout_ptr = out_data + i * 3 * srcw;
    for (int j = 0; j < srcw; j++) {
      *dout_ptr++ = *din_ptr;
      *dout_ptr++ = *din_ptr;
      *dout_ptr++ = *din_ptr;
      din_ptr++;
    }
  }
}

void image_convert_basic(const uint8_t* in_data,
                         uint8_t* out_data,
                         ImageFormat srcFormat,
                         ImageFormat dstFormat,
                         int srcw,
                         int srch) {
  if (srcFormat == dstFormat) {
    // copy
    memcpy(out_data, in_data, sizeof(uint8_t) * srch * srcw);
  } else {
    if (srcFormat == ImageFormat::NV12 && dstFormat == ImageFormat::BGR) {
      nv12_bgr_basic(in_data, out_data, srcw, srch);
    } else if (srcFormat == ImageFormat::NV21 &&
               dstFormat == ImageFormat::BGR) {
      nv21_bgr_basic(in_data, out_data, srcw, srch);
    } else if (srcFormat == ImageFormat::NV12 &&
               dstFormat == ImageFormat::BGRA) {
      nv12_bgra_basic(in_data, out_data, srcw, srch);
    } else if (srcFormat == ImageFormat::NV21 &&
               dstFormat == ImageFormat::BGRA) {
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
    } else {
      printf("srcFormat: %d, dstFormat: %d pass! \n", srcFormat, dstFormat);
    }
    /*
    else if ((srcFormat == RGBA && dstFormat == RGB) ||
               (srcFormat == BGRA && dstFormat == BGR)) {
      impl_ = hwc4_to_hwc3;
    } else if ((srcFormat == RGB && dstFormat == RGBA) ||
               (srcFormat == BGR && dstFormat == BGRA)) {
      impl_ = hwc4_to_hwc3;
    } else if ((srcFormat == RGB && dstFormat == BGR) ||
               (srcFormat == BGR && dstFormat == RGB)) {
      impl_ = hwc3_trans;
    } else if ((srcFormat == RGBA && dstFormat == BGRA) ||
               (srcFormat == BGRA && dstFormat == RGBA)) {
      impl_ = hwc4_trans;
    } else if ((srcFormat == GRAY && dstFormat == RGB) ||
               (srcFormat == GRAY && dstFormat == BGR)) {
      impl_ = hwc1_to_hwc3;
    } else if ((srcFormat == RGBA && dstFormat == BGR) ||
               (srcFormat == BGRA && dstFormat == RGB)) {
      impl_ = hwc4_trans_hwc3;
    } else if ((srcFormat == RGB && dstFormat == BGRA) ||
               (srcFormat == BGR && dstFormat == RGBA)) {
      impl_ = hwc3_trans_hwc4;
    } else {
      printf("srcFormat: %d, dstFormat: %d does not support! \n",
             srcFormat,
             dstFormat);
    }
    */
  }
}
