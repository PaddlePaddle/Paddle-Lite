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

#include "lite/tests/cv/anakin/cv_utils.h"

void rotate90_hwc_bgra(const uint8_t* src, uint8_t* dst, int w_in, int h_in);

void rotate270_hwc_bgra(const uint8_t* src, uint8_t* dst, int w_in, int h_in);

void rotate180_hwc_bgra(const uint8_t* src, uint8_t* dst, int w_in, int h_in);

void bgra_rotate_hwc(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int angle) {
  if (angle == 90) {
    rotate90_hwc_bgra(src, dst, w_in, h_in);
  }
  if (angle == 270) {
    rotate270_hwc_bgra(src, dst, w_in, h_in);
  }
  if (angle == 180) {
    rotate180_hwc_bgra(src, dst, w_in, h_in);
  }
}

/*
bgr1 bgr2 bgr3
bgr4 bgr5 bgr6
bgr7 bgr8 bgr9
rotate:
bgr7 bgr4 bgr1
bgr8 bgr5 bgr2
bgr9 bgr6 bgr3
*/
void rotate90_hwc_bgra(const uint8_t* src, uint8_t* dst, int w_in, int h_in) {
  int w_out = h_in;
  int h_out = w_in;
  int win = w_in * 4;
  int wout = w_out * 4;
  int hremain = h_in % 8;
  int stride_h = 4 * win;
  int stride_h_w = 4 * win - 32;
  int ww = w_out - 8;
  // block 8*8. -- 8*8
  int i = 0;
  for (i = 0; i < h_in - 7; i += 8) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;
    const uint8_t* inptr4 = inptr3 + win;
    const uint8_t* inptr5 = inptr4 + win;
    const uint8_t* inptr6 = inptr5 + win;
    const uint8_t* inptr7 = inptr6 + win;
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr4], #64]   \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr5], #64]   \n"
        "prfm   pldl1keep, [%[ptr6]]        \n"
        "prfm   pldl1keep, [%[ptr6], #64]   \n"
        "prfm   pldl1keep, [%[ptr7]]        \n"
        "prfm   pldl1keep, [%[ptr7], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
        "pld [%[ptr0], #64]            @ preload a, 64byte\n"
        "pld [%[ptr1]]            @ preload a, 64byte\n"
        "pld [%[ptr1], #64]            @ preload a, 64byte\n"
        "pld [%[ptr2]]            @ preload a, 64byte\n"
        "pld [%[ptr2], #64]            @ preload a, 64byte\n"
        "pld [%[ptr3]]            @ preload a, 64byte\n"
        "pld [%[ptr3], #64]            @ preload a, 64byte\n"
        "pld [%[ptr4]]            @ preload a, 64byte\n"
        "pld [%[ptr4], #64]            @ preload a, 64byte\n"
        "pld [%[ptr5]]            @ preload a, 64byte\n"
        "pld [%[ptr5], #64]            @ preload a, 64byte\n"
        "pld [%[ptr6]]            @ preload a, 64byte\n"
        "pld [%[ptr6], #64]            @ preload a, 64byte\n"
        "pld [%[ptr7]]            @ preload a, 64byte\n"
        "pld [%[ptr7], #64]            @ preload a, 64byte\n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7)
        : "memory");
#endif
    int j = 0;
    for (; j < w_in; j++) {
      int tmpx = (ww - i) * 4;
      uint8_t* outptr = dst + j * wout + tmpx;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;

      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;

      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;

      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;

      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;

      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;

      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;

      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
    }
  }
  ww = w_out - 1;
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * win;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + j * wout + (ww - i) * 4;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
    }
  }
}
/*
bgr1 bgr2 bgr3
bgr4 bgr5 bgr6
bgr7 bgr8 bgr9
rotate:
bgr3 bgr6 bgr9
bgr2 bgr5 bgr8
bgr1 bgr4 bgr7
*/
// dst = (h_out - 1) * w_out
// 类似rotate90，将输出结果倒着输出 或者先rotate90,然后沿Y轴翻转
void rotate270_hwc_bgra(const uint8_t* src, uint8_t* dst, int w_in, int h_in) {
  int w_out = h_in;
  int h_out = w_in;
  int win = w_in * 4;
  int wout = w_out * 4;
  int hremain = h_in % 8;
  int stride_h = 4 * win;
  int stride_h_w = 4 * win - 32;
  int hout = h_out - 1;
  // block 8*8. -- 8*8
  int i = 0;
  for (; i < h_in - 7; i += 8) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;
    const uint8_t* inptr4 = inptr3 + win;
    const uint8_t* inptr5 = inptr4 + win;
    const uint8_t* inptr6 = inptr5 + win;
    const uint8_t* inptr7 = inptr6 + win;
    int j = 0;
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr4], #64]   \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr5], #64]   \n"
        "prfm   pldl1keep, [%[ptr6]]        \n"
        "prfm   pldl1keep, [%[ptr6], #64]   \n"
        "prfm   pldl1keep, [%[ptr7]]        \n"
        "prfm   pldl1keep, [%[ptr7], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
        "pld [%[ptr0], #64]            @ preload a, 64byte\n"
        "pld [%[ptr1]]            @ preload a, 64byte\n"
        "pld [%[ptr1], #64]            @ preload a, 64byte\n"
        "pld [%[ptr2]]            @ preload a, 64byte\n"
        "pld [%[ptr2], #64]            @ preload a, 64byte\n"
        "pld [%[ptr3]]            @ preload a, 64byte\n"
        "pld [%[ptr3], #64]            @ preload a, 64byte\n"
        "pld [%[ptr4]]            @ preload a, 64byte\n"
        "pld [%[ptr4], #64]            @ preload a, 64byte\n"
        "pld [%[ptr5]]            @ preload a, 64byte\n"
        "pld [%[ptr5], #64]            @ preload a, 64byte\n"
        "pld [%[ptr6]]            @ preload a, 64byte\n"
        "pld [%[ptr6], #64]            @ preload a, 64byte\n"
        "pld [%[ptr7]]            @ preload a, 64byte\n"
        "pld [%[ptr7], #64]            @ preload a, 64byte\n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5),
          [ptr6] "r"(inptr6),
          [ptr7] "r"(inptr7)
        : "memory");
#endif
    for (; j < w_in; j++) {
      int tmpx = i * 4;
      uint8_t* outptr = dst + (hout - j) * wout + tmpx;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;

      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;

      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;

      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;

      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;

      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;

      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;

      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
    }
  }

  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * win;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + (hout - j) * wout + i * 4;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
    }
  }
}
/*
bgr1 bgr2 bgr3
bgr4 bgr5 bgr6
bgr7 bgr8 bgr9
rotate:
bgr9 bgr8 bgr7
bgr6 bgr5 bgr4
bgr3 bgr2 bgr1
*/
// filp y
void rotate180_hwc_bgra(const uint8_t* src, uint8_t* dst, int w, int h_in) {
  int w_in = w * 4;
  uint8_t zerobuff[w_in];  // NOLINT
  memset(zerobuff, 0, w_in * sizeof(uint8_t));
  int stride_w = 4;
  // 4*8
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    uint8_t* outptr0 = dst + (h_in - i) * w_in - stride_w;  // last
    uint8_t* outptr1 = outptr0 - w_in;
    uint8_t* outptr2 = outptr1 - w_in;
    uint8_t* outptr3 = outptr2 - w_in;

    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
        case 2:
          inptr1 = zerobuff;
        case 1:
          inptr2 = zerobuff;
        case 0:
          inptr3 = zerobuff;
        default:
          break;
      }
    }
#ifdef __aarch64__
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
#else
    asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
        "pld [%[ptr0], #64]            @ preload a, 64byte\n"
        "pld [%[ptr1]]            @ preload a, 64byte\n"
        "pld [%[ptr1], #64]            @ preload a, 64byte\n"
        "pld [%[ptr2]]            @ preload a, 64byte\n"
        "pld [%[ptr2], #64]            @ preload a, 64byte\n"
        "pld [%[ptr3]]            @ preload a, 64byte\n"
        "pld [%[ptr3], #64]            @ preload a, 64byte\n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
#endif
    int j = 0;
    for (; j < w; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            outptr2 -= 8;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            outptr1 -= 8;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            outptr0 -= 8;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        outptr3 -= 8;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        outptr2 -= 8;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        outptr1 -= 8;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        outptr0 -= 8;
      }
    }
  }
}
