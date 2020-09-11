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

void flip_x_hwc(const uint8_t* src, uint8_t* dst, int w_in, int h_in);

void flip_y_hwc(const uint8_t* src, uint8_t* dst, int w_in, int h_in);

void flip_xy_hwc(const uint8_t* src, uint8_t* dst, int w_in, int h_in);

// x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void bgr_flip_hwc(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int flip_num) {
  if (flip_num == 1) {  // x
    flip_x_hwc(src, dst, w_in, h_in);
  }
  if (flip_num == -1) {  // y
    flip_y_hwc(src, dst, w_in, h_in);
  }
  if (flip_num == 0) {  // xy
    flip_xy_hwc(src, dst, w_in, h_in);
  }
}
/*
bgr1 bgr2 bgr3
bgr4 bgr5 bgr6
bgr7 bgr8 bgr9
rotate:
bgr7 bgr8 bgr9
bgr4 bgr5 bgr6
bgr1 bgr2 bgr3
*/
#ifdef __aarch64__
void flip_x_hwc(const uint8_t* src, uint8_t* dst, int w_in, int h_in) {
  int h = h_in - 1;
  int win = w_in * 3;
  uint8_t zerobuff[win];  // NOLINT
  memset(zerobuff, 0, win * sizeof(uint8_t));
  uint8_t zerobuff2[win];  // NOLINT
  memset(zerobuff2, 0, win * sizeof(uint8_t));
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    uint8_t* outptr0 = dst + (h - i) * win;  // last
    uint8_t* outptr1 = outptr0 - win;
    uint8_t* outptr2 = outptr1 - win;
    uint8_t* outptr3 = outptr2 - win;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = zerobuff2;
        case 2:
          inptr1 = zerobuff;
          outptr1 = zerobuff2;
        case 1:
          inptr2 = zerobuff;
          outptr2 = zerobuff2;
        case 0:
          inptr3 = zerobuff;
          outptr3 = zerobuff2;
        default:
          break;
      }
    }
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      asm volatile(
          "ld3  {v0.8b, v1.8b, v2.8b}, [%[inptr0]], #24    \n"  // v0={00,01,02,
                                                                // 03, 04, 05,
                                                                // 06, 07}"
          "ld3  {v3.8b, v4.8b, v5.8b}, [%[inptr1]], #24     \n"  // v0={10,11,12,
                                                                 // 13, 14, 15,
                                                                 // 16, 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]], #24    \n"  // v0={20,21,22,
                                                                // 23, 24, 25,
                                                                // 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]], #24    \n"  // v0={30,31,32,
                                                                  // 33, 34, 35,
                                                                  // 36, 37}"

          "prfm   pldl1keep, [%[inptr0]]        \n"
          "prfm   pldl1keep, [%[inptr1]]        \n"
          "prfm   pldl1keep, [%[inptr2]]        \n"
          "prfm   pldl1keep, [%[inptr3]]        \n"

          "st3 {v0.8b, v1.8b, v2.8b}, [%[outptr0]], #24             \n"   // 00
                                                                          // 10
                                                                          // 20
                                                                          // 30
                                                                          // 04
                                                                          // 14
                                                                          // 24
                                                                          // 34
          "st3 {v3.8b, v4.8b, v5.8b}, [%[outptr1]], #24              \n"  // 02
                                                                          // 12
                                                                          // 22
                                                                          // 32
          "st3 {v6.8b, v7.8b, v8.8b}, [%[outptr2]], #24             \n"   // 01
                                                                          // 11
                                                                          // 21
                                                                          // 31
          "st3 {v9.8b, v10.8b, v11.8b}, [%[outptr3]], #24              \n"  // 03 13 23 33

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3)
          :
          : "v0",
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
            "v11");
    }
    for (; j < w_in; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
      }
    }
  }
}
#else
void flip_x_hwc(const uint8_t* src, uint8_t* dst, int w_in, int h_in) {
  int win = w_in * 3;
  uint8_t zerobuff[win];  // NOLINT
  memset(zerobuff, 0, win * sizeof(uint8_t));
  uint8_t zerobuff2[win];  // NOLINT
  memset(zerobuff2, 0, win * sizeof(uint8_t));
  int h = h_in - 1;
  // 4*8
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    uint8_t* outptr0 = dst + (h - i) * win;  // last
    uint8_t* outptr1 = outptr0 - win;
    uint8_t* outptr2 = outptr1 - win;
    uint8_t* outptr3 = outptr2 - win;
    asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
        "pld [%[ptr1]]            @ preload a, 64byte\n"
        "pld [%[ptr2]]            @ preload a, 64byte\n"
        "pld [%[ptr3]]            @ preload a, 64byte\n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = zerobuff2;
        case 2:
          inptr1 = zerobuff;
          outptr1 = zerobuff2;
        case 1:
          inptr2 = zerobuff;
          outptr2 = zerobuff2;
        case 0:
          inptr3 = zerobuff;
          outptr3 = zerobuff2;
        default:
          break;
      }
    }
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      asm volatile(
          "vld3.8  {d0, d1, d2}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 "
          "04 05 06 07\n"
          "vld3.8  {d3, d4, d5}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 "
          "14 15 16 17\n"
          "vld3.8  {d6, d7, d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 "
          "24 25 26 27\n"
          "vld3.8  {d9, d10, d11}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 "
          "33 34 35 36 37\n"

          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr1]]                         @ preload a, 64byte\n"
          "pld [%[inptr2]]                         @ preload a, 64byte\n"
          "pld [%[inptr3]]                         @ preload a, 64byte\n"

          "vst3.8  {d0, d1, d2},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d3, d4, d5},    [%[outptr1]]!   @ write d4(q0,low),r01,r11 "
          "21 31\n"
          "vst3.8  {d6, d7, d8},    [%[outptr2]]!   @ write d4(q0,low),r01,r11 "
          "21 31\n"
          "vst3.8  {d9, d10, d11},    [%[outptr3]]!   @ write "
          "d4(q0,low),r01,r11 21 31\n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3)
          :
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
    }
    for (; j < w_in; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
      }
    }
  }
}
#endif
/*
bgr1 bgr2 bgr3
bgr4 bgr5 bgr6
bgr7 bgr8 bgr9
flip:
bgr3 bgr2 bgr1
bgr6 bgr5 bgr4
bgr9 bgr8 bgr7
*/
#ifdef __aarch64__
void flip_y_hwc(const uint8_t* src, uint8_t* dst, int w, int h_in) {
  int w_in = w * 3;
  uint8_t zerobuff[w_in];  // NOLINT
  memset(zerobuff, 0, w_in * sizeof(uint8_t));
  uint8_t zerobuff2[w_in];  // NOLINT
  memset(zerobuff2, 0, w_in * sizeof(uint8_t));
  int stride_w = 24;
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    uint8_t* outptr0 = dst + (i + 1) * w_in - stride_w;  // last col
    uint8_t* outptr1 = outptr0 + w_in;
    uint8_t* outptr2 = outptr1 + w_in;
    uint8_t* outptr3 = outptr2 + w_in;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    int j = 0;
    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = zerobuff2;
        case 2:
          inptr1 = zerobuff;
          outptr1 = zerobuff2;
        case 1:
          inptr2 = zerobuff;
          outptr2 = zerobuff2;
        case 0:
          inptr3 = zerobuff;
          outptr3 = zerobuff2;
        default:
          break;
      }
    }
    for (; j < w - 7; j += 8) {
      asm volatile(
          "ld3  {v0.8b, v1.8b, v2.8b}, [%[inptr0]], #24    \n"  // v0={00,01,02,
                                                                // 03, 04, 05,
                                                                // 06, 07}"
          "ld3  {v3.8b, v4.8b, v5.8b}, [%[inptr1]], #24     \n"  // v0={10,11,12,
                                                                 // 13, 14, 15,
                                                                 // 16, 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]], #24    \n"  // v0={20,21,22,
                                                                // 23, 24, 25,
                                                                // 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]], #24    \n"  // v0={30,31,32,
                                                                  // 33, 34, 35,
                                                                  // 36, 37}"

          "rev64  v12.8b, v0.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 b
          "rev64  v13.8b, v1.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 g
          "rev64  v14.8b, v2.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 r

          "rev64  v15.8b, v3.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v16.8b, v4.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v17.8b, v5.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00

          "rev64  v18.8b, v6.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v19.8b, v7.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v20.8b, v8.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00

          "rev64  v21.8b, v9.8b                \n"   //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v22.8b, v10.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v23.8b, v11.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00

          "prfm   pldl1keep, [%[inptr0]]        \n"
          "prfm   pldl1keep, [%[inptr1]]        \n"
          "prfm   pldl1keep, [%[inptr2]]        \n"
          "prfm   pldl1keep, [%[inptr3]]        \n"

          "st3 {v12.8b, v13.8b, v14.8b}, [%[outptr0]]             \n"   // 00 10
                                                                        // 20 30
                                                                        // 04 14
                                                                        // 24 34
          "st3 {v15.8b, v16.8b, v17.8b}, [%[outptr1]]              \n"  // 02 12
                                                                        // 22 32
          "st3 {v18.8b, v19.8b, v20.8b}, [%[outptr2]]             \n"   // 01 11
                                                                        // 21 31
          "st3 {v21.8b, v22.8b, v23.8b}, [%[outptr3]]              \n"  // 03 13
                                                                        // 23 33

          "sub %[outptr0], %[outptr0], %[stride_w]       \n"  //@ ptr - stride_w
          "sub %[outptr1], %[outptr1], %[stride_w]       \n"
          "sub %[outptr2], %[outptr2], %[stride_w]       \n"
          "sub %[outptr3], %[outptr3], %[stride_w]       \n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [stride_w] "+r"(stride_w)
          :
          : "v0",
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
    }
    outptr3 += stride_w - 3;
    outptr2 += stride_w - 3;
    outptr1 += stride_w - 3;
    outptr0 += stride_w - 3;
    for (; j < w; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            outptr2 -= 6;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            outptr1 -= 6;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            outptr0 -= 6;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        outptr3 -= 6;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        outptr2 -= 6;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        outptr1 -= 6;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        outptr0 -= 6;
      }
    }
  }
}
#else
void flip_y_hwc(const uint8_t* src, uint8_t* dst, int w, int h_in) {
  int w_in = w * 3;
  uint8_t zerobuff[w_in];  // NOLINT
  memset(zerobuff, 0, w_in * sizeof(uint8_t));
  uint8_t zerobuff2[w_in];  // NOLINT
  memset(zerobuff2, 0, w_in * sizeof(uint8_t));
  int stride_w = 24;
  // 4*8
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    uint8_t* outptr0 = dst + (i + 1) * w_in - stride_w;  // last
    uint8_t* outptr1 = outptr0 + w_in;
    uint8_t* outptr2 = outptr1 + w_in;
    uint8_t* outptr3 = outptr2 + w_in;
    asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
        "pld [%[ptr1]]            @ preload a, 64byte\n"
        "pld [%[ptr2]]            @ preload a, 64byte\n"
        "pld [%[ptr3]]            @ preload a, 64byte\n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = zerobuff2;
        case 2:
          inptr1 = zerobuff;
          outptr1 = zerobuff2;
        case 1:
          inptr2 = zerobuff;
          outptr2 = zerobuff2;
        case 0:
          inptr3 = zerobuff;
          outptr3 = zerobuff2;
        default:
          break;
      }
    }
    int j = 0;
    for (; j < w - 7; j += 8) {
      asm volatile(
          "vld3.8  {d0, d1, d2}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 "
          "04 05 06 07\n"
          "vld3.8  {d3, d4, d5}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 "
          "14 15 16 17\n"
          "vld3.8  {d6, d7, d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 "
          "24 25 26 27\n"
          "vld3.8  {d9, d10, d11}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 "
          "33 34 35 36 37\n"

          "vrev64.8  d12, d0               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d13, d1               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d14, d2               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d15, d3               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d16, d4               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d17, d5               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d18, d6               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d19, d7               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d20, d8               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d21, d9               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d22, d10               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"
          "vrev64.8  d23, d11               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"

          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr1]]                         @ preload a, 64byte\n"
          "pld [%[inptr2]]                         @ preload a, 64byte\n"
          "pld [%[inptr3]]                         @ preload a, 64byte\n"

          "vst3.8  {d12, d13, d14},    [%[outptr0]]   @ write "
          "d0(q0,low),r00,r10 20 30\n"
          "vst3.8  {d15, d16, d17},    [%[outptr1]]   @ write "
          "d4(q0,low),r01,r11 21 31\n"
          "vst3.8  {d18, d19, d20},    [%[outptr2]]   @ write "
          "d4(q0,low),r01,r11 21 31\n"
          "vst3.8  {d21, d22, d23},    [%[outptr3]]   @ write "
          "d4(q0,low),r01,r11 21 31\n"

          "sub %[outptr0], %[stride_w]       @ ptr - stride_w \n"
          "sub %[outptr1], %[stride_w]       @ ptr - stride_w \n"
          "sub %[outptr2], %[stride_w]       @ ptr - stride_w \n"
          "sub %[outptr3], %[stride_w]       @ ptr - stride_w \n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [stride_w] "+r"(stride_w)
          :
          : "q0",
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
            "q12");
    }
    outptr3 += stride_w - 3;
    outptr2 += stride_w - 3;
    outptr1 += stride_w - 3;
    outptr0 += stride_w - 3;
    for (; j < w; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            outptr2 -= 6;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            outptr1 -= 6;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            outptr0 -= 6;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        outptr3 -= 6;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        outptr2 -= 6;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        outptr1 -= 6;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        outptr0 -= 6;
      }
    }
  }
}
#endif
/*
bgr1 bgr2 bgr3
bgr4 bgr5 bgr6
bgr7 bgr8 bgr9
flip:
bgr9 bgr8 bgr7
bgr6 bgr5 bgr4
bgr3 bgr2 bgr1
*/
#ifdef __aarch64__
void flip_xy_hwc(const uint8_t* src, uint8_t* dst, int w, int h_in) {
  int stride_w = 24;
  int w_in = w * 3;
  uint8_t zerobuff[w_in];  // NOLINT
  memset(zerobuff, 0, w_in * sizeof(uint8_t));
  uint8_t zerobuff2[w_in];  // NOLINT
  memset(zerobuff2, 0, w_in * sizeof(uint8_t));
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    uint8_t* outptr0 = dst + (h_in - i) * w_in - stride_w;  // last col
    uint8_t* outptr1 = outptr0 - w_in;
    uint8_t* outptr2 = outptr1 - w_in;
    uint8_t* outptr3 = outptr2 - w_in;

    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = zerobuff2;
        case 2:
          inptr1 = zerobuff;
          outptr1 = zerobuff2;
        case 1:
          inptr2 = zerobuff;
          outptr2 = zerobuff2;
        case 0:
          inptr3 = zerobuff;
          outptr3 = zerobuff2;
        default:
          break;
      }
    }
    int j = 0;
    for (; j < w - 7; j += 8) {
      asm volatile(
          "ld3  {v0.8b, v1.8b, v2.8b}, [%[inptr0]], #24    \n"  // v0={00,01,02,
                                                                // 03, 04, 05,
                                                                // 06, 07}"
          "ld3  {v3.8b, v4.8b, v5.8b}, [%[inptr1]], #24     \n"  // v0={10,11,12,
                                                                 // 13, 14, 15,
                                                                 // 16, 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]], #24    \n"  // v0={20,21,22,
                                                                // 23, 24, 25,
                                                                // 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]], #24    \n"  // v0={30,31,32,
                                                                  // 33, 34, 35,
                                                                  // 36, 37}"

          "rev64  v12.8b, v0.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 b
          "rev64  v13.8b, v1.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 g
          "rev64  v14.8b, v2.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 r

          "rev64  v15.8b, v3.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v16.8b, v4.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v17.8b, v5.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00

          "rev64  v18.8b, v6.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v19.8b, v7.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v20.8b, v8.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00

          "rev64  v21.8b, v9.8b                \n"   //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v22.8b, v10.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v23.8b, v11.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00

          "prfm   pldl1keep, [%[inptr0]]        \n"
          "prfm   pldl1keep, [%[inptr1]]        \n"
          "prfm   pldl1keep, [%[inptr2]]        \n"
          "prfm   pldl1keep, [%[inptr3]]        \n"

          "st3 {v12.8b, v13.8b, v14.8b}, [%[outptr0]]             \n"   // 00 10
                                                                        // 20 30
                                                                        // 04 14
                                                                        // 24 34
          "st3 {v15.8b, v16.8b, v17.8b}, [%[outptr1]]              \n"  // 02 12
                                                                        // 22 32
          "st3 {v18.8b, v19.8b, v20.8b}, [%[outptr2]]             \n"   // 01 11
                                                                        // 21 31
          "st3 {v21.8b, v22.8b, v23.8b}, [%[outptr3]]              \n"  // 03 13
                                                                        // 23 33

          "sub %[outptr0], %[outptr0], %[stride_w]       \n"  //@ ptr - stride_w
          "sub %[outptr1], %[outptr1], %[stride_w]       \n"
          "sub %[outptr2], %[outptr2], %[stride_w]       \n"
          "sub %[outptr3], %[outptr3], %[stride_w]       \n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [stride_w] "+r"(stride_w)
          :
          : "v0",
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
    }
    outptr3 += stride_w - 3;
    outptr2 += stride_w - 3;
    outptr1 += stride_w - 3;
    outptr0 += stride_w - 3;
    for (; j < w; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            outptr2 -= 6;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            outptr1 -= 6;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            outptr0 -= 6;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        outptr3 -= 6;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        outptr2 -= 6;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        outptr1 -= 6;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        outptr0 -= 6;
      }
    }
  }
}
#else
void flip_xy_hwc(const uint8_t* src, uint8_t* dst, int w, int h_in) {
  int w_in = w * 3;
  uint8_t zerobuff[w_in];  // NOLINT
  memset(zerobuff, 0, w_in * sizeof(uint8_t));
  uint8_t zerobuff2[w_in];  // NOLINT
  memset(zerobuff2, 0, w_in * sizeof(uint8_t));
  int stride_w = 24;
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
    asm volatile(
        "pld [%[ptr0]]                         @ preload a, 64byte\n"
        "pld [%[ptr1]]            @ preload a, 64byte\n"
        "pld [%[ptr2]]            @ preload a, 64byte\n"
        "pld [%[ptr3]]            @ preload a, 64byte\n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3)
        : "memory");
    if (i + 3 >= h_in) {
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = zerobuff2;
        case 2:
          inptr1 = zerobuff;
          outptr1 = zerobuff2;
        case 1:
          inptr2 = zerobuff;
          outptr2 = zerobuff2;
        case 0:
          inptr3 = zerobuff;
          outptr3 = zerobuff2;
        default:
          break;
      }
    }
    int j = 0;
    for (; j < w - 7; j += 8) {
      asm volatile(
          "vld3.8  {d0, d1, d2}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 "
          "04 05 06 07\n"
          "vld3.8  {d3, d4, d5}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 "
          "14 15 16 17\n"
          "vld3.8  {d6, d7, d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 "
          "24 25 26 27\n"
          "vld3.8  {d9, d10, d11}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 "
          "33 34 35 36 37\n"

          "vrev64.8  d12, d0               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d13, d1               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d14, d2               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d15, d3               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d16, d4               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d17, d5               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d18, d6               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d19, d7               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d20, d8               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d21, d9               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d22, d10               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"
          "vrev64.8  d23, d11               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"

          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr1]]                         @ preload a, 64byte\n"
          "pld [%[inptr2]]                         @ preload a, 64byte\n"
          "pld [%[inptr3]]                         @ preload a, 64byte\n"

          "vst3.8  {d12, d13, d14},    [%[outptr0]]   @ write "
          "d0(q0,low),r00,r10 20 30\n"
          "vst3.8  {d15, d16, d17},    [%[outptr1]]   @ write "
          "d4(q0,low),r01,r11 21 31\n"
          "vst3.8  {d18, d19, d20},    [%[outptr2]]   @ write "
          "d4(q0,low),r01,r11 21 31\n"
          "vst3.8  {d21, d22, d23},    [%[outptr3]]   @ write "
          "d4(q0,low),r01,r11 21 31\n"

          "sub %[outptr0], %[stride_w]       @ ptr - stride_w \n"
          "sub %[outptr1], %[stride_w]       @ ptr - stride_w \n"
          "sub %[outptr2], %[stride_w]       @ ptr - stride_w \n"
          "sub %[outptr3], %[stride_w]       @ ptr - stride_w \n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [stride_w] "+r"(stride_w)
          :
          : "q0",
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
            "q12");
    }
    outptr3 += stride_w - 3;
    outptr2 += stride_w - 3;
    outptr1 += stride_w - 3;
    outptr0 += stride_w - 3;
    for (; j < w; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            *outptr2++ = *inptr2++;
            outptr2 -= 6;
          case 1:
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            *outptr1++ = *inptr1++;
            outptr1 -= 6;
          case 2:
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            *outptr0++ = *inptr0++;
            outptr0 -= 6;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        *outptr3++ = *inptr3++;
        outptr3 -= 6;

        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        *outptr2++ = *inptr2++;
        outptr2 -= 6;

        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        *outptr1++ = *inptr1++;
        outptr1 -= 6;

        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        *outptr0++ = *inptr0++;
        outptr0 -= 6;
      }
    }
  }
}
#endif
