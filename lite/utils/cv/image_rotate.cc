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

#include "lite/utils/cv/image_rotate.h"
#include <math.h>
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
/*
1 2 3
4 5 6
7 8 9
rotate:
1 4 7
2 5 8
3 6 9
*/
// transpose
void rotate_hwc1_90(const uint8_t* src,
                    uint8_t* dst,
                    int w_in,
                    int h_in,
                    int w_out,
                    int h_out) {
  uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // block 4*8. -- 8*4
  int i = 0;
#ifdef __aarch64__
  int64_t stride_h = 4 * w_in;
  int64_t stride_h_w = 4 * w_in - 8;
  for (i = 0; i < h_in - 7; i += 8) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      uint8_t* outptr0 = dst + j * w_out + i;
      uint8_t* outptr1 = outptr0 + w_out;
      uint8_t* outptr2 = outptr1 + w_out;
      uint8_t* outptr3 = outptr2 + w_out;
      uint8_t* outptr4 = outptr3 + w_out;
      uint8_t* outptr5 = outptr4 + w_out;
      uint8_t* outptr6 = outptr5 + w_out;
      uint8_t* outptr7 = outptr6 + w_out;
      asm volatile(
          "ld1  {v0.8b}, [%[inptr0]]    \n"  // v0={00,01,02, 03, 04, 05, 06,
                                             // 07}"
          "ld1  {v1.8b}, [%[inptr1]]    \n"  // v0={10,11,12, 13, 14, 15, 16,
                                             // 17}"
          "ld1  {v2.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23, 24, 25, 26,
                                             // 27}"
          "ld1  {v3.8b}, [%[inptr3]]    \n"  // v0={30,31,32, 33, 34, 35, 36,
                                             // 37}"

          "add %[inptr0], %[inptr0], %[stride_h] \n"  // 4 +4*w_in
          "add %[inptr1], %[inptr1], %[stride_h] \n"  // 5
          "add %[inptr2], %[inptr2], %[stride_h] \n"  // 6
          "add %[inptr3], %[inptr3], %[stride_h] \n"  // 7

          "trn1 v4.8b, v0.8b, v1.8b             \n"  // v4={00 10 02 12 04 14 06
                                                     // 16 }
          "trn1 v6.8b, v2.8b, v3.8b             \n"  // v4={20 30 22 32 24 34 26
                                                     // 36 }

          "trn2 v5.8b, v0.8b, v1.8b             \n"  // v5={01 11 03 13 05 15 07
                                                     // 17 }
          "trn2 v7.8b, v2.8b, v3.8b             \n"  // v7={21 31 23 33 25 35 27
                                                     // 37 }

          "ld1  {v12.8b}, [%[inptr0]]    \n"  // v0={00,01,02, 03, 04, 05, 06,
                                              // 07}"
          "ld1  {v13.8b}, [%[inptr1]]    \n"  // v0={10,11,12, 13, 14, 15, 16,
                                              // 17}"
          "ld1  {v14.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23, 24, 25, 26,
                                              // 27}"
          "ld1  {v15.8b}, [%[inptr3]]    \n"  // v0={30,31,32, 33, 34, 35, 36,
                                              // 37}"

          "trn1 v0.4h, v4.4h, v6.4h             \n"  // v0={00 10 20 30 04 14 24
                                                     // 34}
          "trn1 v2.4h, v5.4h, v7.4h             \n"  // v2={01 11 21 31 05 15 25
                                                     // 35}

          "trn2 v1.4h, v4.4h, v6.4h             \n"  // v1={02 12 22 32 06 16 26
                                                     // 36}
          "trn2 v3.4h, v5.4h, v7.4h             \n"  // v3={03 13 23 33 07 17 27
                                                     // 37}

          "trn1 v9.8b, v12.8b, v13.8b             \n"   // v9={40 50 42 52 44 54
                                                        // 46 56 }
          "trn1 v11.8b, v14.8b, v15.8b             \n"  // v11={60 70 62 72 64
                                                        // 74 66 76 }

          "trn2 v10.8b, v12.8b, v13.8b             \n"  // v10={01 11 03 13 05
                                                        // 15 07 17 }
          "trn2 v12.8b, v14.8b, v15.8b             \n"  // v12={21 31 23 33 25
                                                        // 35 27 37 }

          "sub %[inptr0], %[inptr0], %[stride_h_w] \n"  // 4 - 4*w_in + 8
          "sub %[inptr1], %[inptr1], %[stride_h_w] \n"  // 5
          "sub %[inptr2], %[inptr2], %[stride_h_w] \n"  // 6
          "sub %[inptr3], %[inptr3], %[stride_h_w] \n"  // 7

          "trn1 v4.4h, v9.4h, v11.4h             \n"   // v4={40 50 60 70 44 54
                                                       // 64 74}
          "trn1 v6.4h, v10.4h, v12.4h             \n"  // v6={41 51 61 71 45 55
                                                       // 65 75}

          "trn2 v5.4h, v9.4h, v11.4h             \n"  // v5={42 52 62 72 46 56
                                                      // 66 76}
          "trn2 v7.4h, v10.4h, v12.4h            \n"  // v7={43 53 63 73 47 57
                                                      // 67 77}

          "trn1 v8.2s, v0.2s, v4.2s             \n"  // v8={00 10 20 30 40 50 60
                                                     // 70}
          "trn1 v9.2s, v2.2s, v6.2s             \n"  // v6={01 11 21 31 41 51 61
                                                     // 71}
          "trn1 v10.2s, v1.2s, v5.2s             \n"  // v10={02 12 22 32 42 52
                                                      // 62 72}
          "trn1 v11.2s, v3.2s, v7.2s             \n"  // v11={03 13 23 33 43 53
                                                      // 63 73}

          "trn2 v12.2s, v0.2s, v4.2s             \n"  // v12={04 14 24 34 44 54
                                                      // 64 74}
          "trn2 v13.2s, v2.2s, v6.2s             \n"  // v13={05 15 25 35 45 55
                                                      // 65 75}
          "trn2 v14.2s, v1.2s, v5.2s             \n"  // v14={06 16 26 36 46 56
                                                      // 66 76}
          "trn2 v15.2s, v3.2s, v7.2s             \n"  // v15={07 17 27 37 47 57
                                                      // 67 77}

          "st1 {v8.8b}, [%[outptr0]], #8             \n"    // 00 10 20 30 04 14
                                                            // 24 34
          "st1 {v9.8b}, [%[outptr1]], #8              \n"   // 02 12 22 32
          "st1 {v10.8b}, [%[outptr2]], #8              \n"  // 01 11 21 31
          "st1 {v11.8b}, [%[outptr3]], #8              \n"  // 03 13 23 33

          "st1 {v12.8b}, [%[outptr4]], #8             \n"   // 00 10 20 30 04 14
                                                            // 24 34
          "st1 {v13.8b}, [%[outptr5]], #8              \n"  // 02 12 22 32
          "st1 {v14.8b}, [%[outptr6]], #8              \n"  // 01 11 21 31
          "st1 {v15.8b}, [%[outptr7]], #8              \n"  // 03 13 23 33

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7),
            [stride_h] "+r"(stride_h),
            [stride_h_w] "+r"(stride_h_w)
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
            "v15");
    }
    const uint8_t* inptr4 = inptr3 + w_in;
    const uint8_t* inptr5 = inptr4 + w_in;
    const uint8_t* inptr6 = inptr5 + w_in;
    const uint8_t* inptr7 = inptr6 + w_in;
    for (; j < w_in; j++) {
      uint8_t* outptr = dst + j * w_out + i;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr7++;
    }
  }
#else
  for (i = 0; i < h_in - 3; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    int j = 0;
    for (j = 0; j < w_in - 7; j += 8) {
      // printf("j: %d, inptr0: %x \n", j, inptr0);
      uint8_t* outptr0 = dst + j * w_out + i;
      uint8_t* outptr1 = outptr0 + w_out;
      uint8_t* outptr2 = outptr1 + w_out;
      uint8_t* outptr3 = outptr2 + w_out;
      uint8_t* outptr4 = outptr3 + w_out;
      uint8_t* outptr5 = outptr4 + w_out;
      uint8_t* outptr6 = outptr5 + w_out;
      uint8_t* outptr7 = outptr6 + w_out;
      asm volatile(
          "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 "
          "06 07\n"
          "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 "
          "16 17\n"
          "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 "
          "26 27\n"
          "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 "
          "36 37\n"

          "vtrn.8  d0, d4                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"
          "vtrn.8  d8, d12                  @ trans data: d8=r20,r30,r12,r32 "
          "24 34 26 36  d12=21 31 23 33 \n"
          // d0=r00,r10,r20,r30 04 14 24 34 d8 = 02 12 22 32 06 16 26 36
          "vtrn.16 d0, d8                  @ trans data: \n"
          // d4=01 11 21 31 05 15 25 35; d12 =r03 13 23 33 07 17 27 37
          "vtrn.16  d4, d12                  @ trans data:\n"
          // d0=00,10,20,30 01 11 21 31 d4 = 04 14 24 34 05 15 25 35
          "vtrn.32 d0, d4                  @ trans data: \n"
          // d8= 02 12 22 32 03 13 23 33; d12 =r06 16 26 36 07 17 27 37
          "vtrn.32  d8, d12                  @ trans data: \n"
          // "vtrn.16  d1, d9                   @ trans data: d1=r04,r14,r24,r34
          // d9=06 16 26 36 \n"

          "vst1.32  {d0[0]},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 20 "
          "30\n"
          "vst1.32  {d0[1]},    [%[outptr1]]!   @ write d4(q0,low),r01,r11 21 "
          "31\n"
          "vst1.32  {d4[0]},    [%[outptr4]]!   @ write d2(q0,low),r02,r12\n"
          "vst1.32  {d4[1]},    [%[outptr5]]!   @ write d6(q0,low),r06,r16\n"
          "vst1.32  {d8[0]},    [%[outptr2]]!   @ write d1(q0,high),r02,r12 22 "
          "32\n"
          "vst1.32  {d8[1]},    [%[outptr3]]!   @ write d5(q0,low),r03,r13 23 "
          "33\n"
          "vst1.32 {d12[0]},     [%[outptr6]]!   @ write d3(q0,low),r03,r13\n"
          "vst1.32  {d12[1]},    [%[outptr7]]!   @ write d7(q0,low),r07,r17\n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7)
          :
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
    }
    for (; j < w_in; j++) {
      uint8_t* outptr = dst + j * w_out + i;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
    }
  }
#endif
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * w_in;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + j * w_out + i;
      *outptr0 = *inptr0++;
    }
  }
}
/*
1 2 3
4 5 6
7 8 9
rotate:
3 2 1
6 5 4
9 8 7
*/
void rotate_hwc1_180(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t stride_w = 8;
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    uint8_t* outptr0 = dst + (i + 1) * w_out - stride_w;  // last
    uint8_t* outptr1 = outptr0 + w_out;
    uint8_t* outptr2 = outptr1 + w_out;
    uint8_t* outptr3 = outptr2 + w_out;

    int j = 0;
    for (; j < w_in - 7; j += 8) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 3:
            inptr0 = zerobuff;
            outptr0 = zerobuff;
          case 2:
            inptr1 = zerobuff;
            outptr1 = zerobuff;
          case 1:
            inptr2 = zerobuff;
            outptr2 = zerobuff;
          case 0:
            inptr3 = zerobuff;
            outptr3 = zerobuff;
          default:
            break;
        }
      }
#ifdef __aarch64__
      asm volatile(
          "ld1  {v0.8b}, [%[inptr0]], #8    \n"   // v0={00,01,02, 03, 04, 05,
                                                  // 06, 07}"
          "ld1  {v1.8b}, [%[inptr1]], #8     \n"  // v0={10,11,12, 13, 14, 15,
                                                  // 16, 17}"
          "ld1  {v2.8b}, [%[inptr2]], #8    \n"   // v0={20,21,22, 23, 24, 25,
                                                  // 26, 27}"
          "ld1  {v3.8b}, [%[inptr3]], #8    \n"   // v0={30,31,32, 33, 34, 35,
                                                  // 36, 37}"

          "rev64  v4.8b, v0.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00
          "rev64  v5.8b, v1.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00
          "rev64  v6.8b, v2.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00
          "rev64  v7.8b, v3.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00

          "prfm   pldl1keep, [%[inptr0]]        \n"
          "prfm   pldl1keep, [%[inptr1]]        \n"
          "prfm   pldl1keep, [%[inptr2]]        \n"
          "prfm   pldl1keep, [%[inptr3]]        \n"

          "st1 {v4.8b}, [%[outptr0]]             \n"  // 00 10 20 30 04 14 24 34
          "st1 {v5.8b}, [%[outptr1]]              \n"  // 02 12 22 32
          "st1 {v6.8b}, [%[outptr2]]             \n"   // 01 11 21 31
          "st1 {v7.8b}, [%[outptr3]]              \n"  // 03 13 23 33

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
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
      asm volatile(
          "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 "
          "06 07\n"
          "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 "
          "16 17\n"
          "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 "
          "26 27\n"
          "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 "
          "36 37\n"

          "vrev64.8  d1, d0               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d5, d4              @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d9, d8               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d13, d12               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"

          "pld [%[inptr0]]                         @ preload a, 64byte\n"
          "pld [%[inptr1]]                         @ preload a, 64byte\n"
          "pld [%[inptr2]]                         @ preload a, 64byte\n"
          "pld [%[inptr3]]                         @ preload a, 64byte\n"

          "vst1.32  {d1},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n"
          "vst1.32  {d5},    [%[outptr1]]   @ write d4(q0,low),r01,r11 21 31\n"
          "vst1.32  {d9},    [%[outptr2]]   @ write d4(q0,low),r01,r11 21 31\n"
          "vst1.32  {d13},    [%[outptr3]]   @ write d4(q0,low),r01,r11 21 31\n"

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
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif
    }
    outptr3 += stride_w - 1;
    outptr2 += stride_w - 1;
    outptr1 += stride_w - 1;
    outptr0 += stride_w - 1;
    for (; j < w_in; j++) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 0:
            *outptr2-- = *inptr2++;
          case 1:
            *outptr1-- = *inptr1++;
          // inptr1 = zerobuff;
          case 2:
            *outptr0-- = *inptr0++;
          case 3:
          // inptr3 = zerobuff;
          default:
            break;
        }
      } else {
        *outptr3-- = *inptr3++;
        *outptr2-- = *inptr2++;
        *outptr1-- = *inptr1++;
        *outptr0-- = *inptr0++;
      }
      // *outptr0-- = *inptr0++;
      // *outptr1-- = *inptr1++;
      // *outptr2-- = *inptr2++;
      // *outptr3-- = *inptr3++;
    }
  }
}
/*
1 2 3
4 5 6
7 8 9
rotate:
3 6 9
2 5 8
1 4 7
*/
// dst = (h_out - 1) * w_out
// 类似rotate90，将输出结果倒着输出 或者先rotate90,然后沿Y轴翻转
void rotate_hwc1_270(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t stride_h = 4 * w_in;
  int64_t stride_h_w = 4 * w_in - 8;
  int hout = h_out - 1;

  int i = 0;
// block 8*8. -- 8*8
#ifdef __aarch64__
  for (; i < h_in - 7; i += 8) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      uint8_t* outptr0 = dst + (hout - j) * w_out + i;
      uint8_t* outptr1 = outptr0 - w_out;
      uint8_t* outptr2 = outptr1 - w_out;
      uint8_t* outptr3 = outptr2 - w_out;
      uint8_t* outptr4 = outptr3 - w_out;
      uint8_t* outptr5 = outptr4 - w_out;
      uint8_t* outptr6 = outptr5 - w_out;
      uint8_t* outptr7 = outptr6 - w_out;
      asm volatile(
          "ld1  {v0.8b}, [%[inptr0]]    \n"  // v0={00,01,02, 03, 04, 05, 06,
                                             // 07}"
          "ld1  {v1.8b}, [%[inptr1]]    \n"  // v0={10,11,12, 13, 14, 15, 16,
                                             // 17}"
          "ld1  {v2.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23, 24, 25, 26,
                                             // 27}"
          "ld1  {v3.8b}, [%[inptr3]]    \n"  // v0={30,31,32, 33, 34, 35, 36,
                                             // 37}"

          "add %[inptr0], %[inptr0], %[stride_h] \n"  // 4 +4*w_in
          "add %[inptr1], %[inptr1], %[stride_h] \n"  // 5
          "add %[inptr2], %[inptr2], %[stride_h] \n"  // 6
          "add %[inptr3], %[inptr3], %[stride_h] \n"  // 7

          "trn1 v4.8b, v0.8b, v1.8b             \n"  // v4={00 10 02 12 04 14 06
                                                     // 16 }
          "trn1 v6.8b, v2.8b, v3.8b             \n"  // v4={20 30 22 32 24 34 26
                                                     // 36 }

          "trn2 v5.8b, v0.8b, v1.8b             \n"  // v5={01 11 03 13 05 15 07
                                                     // 17 }
          "trn2 v7.8b, v2.8b, v3.8b             \n"  // v7={21 31 23 33 25 35 27
                                                     // 37 }

          "ld1  {v12.8b}, [%[inptr0]]    \n"  // v0={00,01,02, 03, 04, 05, 06,
                                              // 07}"
          "ld1  {v13.8b}, [%[inptr1]]    \n"  // v0={10,11,12, 13, 14, 15, 16,
                                              // 17}"
          "ld1  {v14.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23, 24, 25, 26,
                                              // 27}"
          "ld1  {v15.8b}, [%[inptr3]]    \n"  // v0={30,31,32, 33, 34, 35, 36,
                                              // 37}"

          "trn1 v0.4h, v4.4h, v6.4h             \n"  // v0={00 10 20 30 04 14 24
                                                     // 34}
          "trn1 v2.4h, v5.4h, v7.4h             \n"  // v2={01 11 21 31 05 15 25
                                                     // 35}

          "trn2 v1.4h, v4.4h, v6.4h             \n"  // v1={02 12 22 32 06 16 26
                                                     // 36}
          "trn2 v3.4h, v5.4h, v7.4h             \n"  // v3={03 13 23 33 07 17 27
                                                     // 37}

          "trn1 v9.8b, v12.8b, v13.8b             \n"   // v9={40 50 42 52 44 54
                                                        // 46 56 }
          "trn1 v11.8b, v14.8b, v15.8b             \n"  // v11={60 70 62 72 64
                                                        // 74 66 76 }

          "trn2 v10.8b, v12.8b, v13.8b             \n"  // v10={01 11 03 13 05
                                                        // 15 07 17 }
          "trn2 v12.8b, v14.8b, v15.8b             \n"  // v12={21 31 23 33 25
                                                        // 35 27 37 }

          "sub %[inptr0], %[inptr0], %[stride_h_w] \n"  // 4 - 4*w_in + 8
          "sub %[inptr1], %[inptr1], %[stride_h_w] \n"  // 5
          "sub %[inptr2], %[inptr2], %[stride_h_w] \n"  // 6
          "sub %[inptr3], %[inptr3], %[stride_h_w] \n"  // 7

          "trn1 v4.4h, v9.4h, v11.4h             \n"   // v4={40 50 60 70 44 54
                                                       // 64 74}
          "trn1 v6.4h, v10.4h, v12.4h             \n"  // v6={41 51 61 71 45 55
                                                       // 65 75}

          "trn2 v5.4h, v9.4h, v11.4h             \n"  // v5={42 52 62 72 46 56
                                                      // 66 76}
          "trn2 v7.4h, v10.4h, v12.4h            \n"  // v7={43 53 63 73 47 57
                                                      // 67 77}

          "trn1 v8.2s, v0.2s, v4.2s             \n"  // v8={00 10 20 30 40 50 60
                                                     // 70}
          "trn1 v9.2s, v2.2s, v6.2s             \n"  // v6={01 11 21 31 41 51 61
                                                     // 71}
          "trn1 v10.2s, v1.2s, v5.2s             \n"  // v10={02 12 22 32 42 52
                                                      // 62 72}
          "trn1 v11.2s, v3.2s, v7.2s             \n"  // v11={03 13 23 33 43 53
                                                      // 63 73}

          "trn2 v12.2s, v0.2s, v4.2s             \n"  // v12={04 14 24 34 44 54
                                                      // 64 74}
          "trn2 v13.2s, v2.2s, v6.2s             \n"  // v13={05 15 25 35 45 55
                                                      // 65 75}
          "trn2 v14.2s, v1.2s, v5.2s             \n"  // v14={06 16 26 36 46 56
                                                      // 66 76}
          "trn2 v15.2s, v3.2s, v7.2s             \n"  // v15={07 17 27 37 47 57
                                                      // 67 77}

          "st1 {v8.8b}, [%[outptr0]], #8             \n"    // 00 10 20 30 04 14
                                                            // 24 34
          "st1 {v9.8b}, [%[outptr1]], #8              \n"   // 02 12 22 32
          "st1 {v10.8b}, [%[outptr2]], #8              \n"  // 01 11 21 31
          "st1 {v11.8b}, [%[outptr3]], #8              \n"  // 03 13 23 33

          "st1 {v12.8b}, [%[outptr4]], #8             \n"   // 00 10 20 30 04 14
                                                            // 24 34
          "st1 {v13.8b}, [%[outptr5]], #8              \n"  // 02 12 22 32
          "st1 {v14.8b}, [%[outptr6]], #8              \n"  // 01 11 21 31
          "st1 {v15.8b}, [%[outptr7]], #8              \n"  // 03 13 23 33

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7),
            [stride_h] "+r"(stride_h),
            [stride_h_w] "+r"(stride_h_w)
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
            "v15");
    }
    const uint8_t* inptr4 = inptr3 + w_in;
    const uint8_t* inptr5 = inptr4 + w_in;
    const uint8_t* inptr6 = inptr5 + w_in;
    const uint8_t* inptr7 = inptr6 + w_in;
    for (; j < w_in; j++) {
      uint8_t* outptr = dst + (hout - j) * w_out + i;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr7++;
    }
  }
#else
  // block 4*8. -- 8*4
  for (; i < h_in - 3; i += 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      uint8_t* outptr0 = dst + (hout - j) * w_out + i;
      uint8_t* outptr1 = outptr0 - w_out;
      uint8_t* outptr2 = outptr1 - w_out;
      uint8_t* outptr3 = outptr2 - w_out;
      uint8_t* outptr4 = outptr3 - w_out;
      uint8_t* outptr5 = outptr4 - w_out;
      uint8_t* outptr6 = outptr5 - w_out;
      uint8_t* outptr7 = outptr6 - w_out;
      asm volatile(
          "vld1.8  {d0}, [%[inptr0]]!   @ zip load r0, d0 =00 01 02 03 04 05 "
          "06 07\n"
          "vld1.8  {d4}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 14 15 "
          "16 17\n"
          "vld1.8  {d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 24 25 "
          "26 27\n"
          "vld1.8  {d12}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 33 34 35 "
          "36 37\n"

          "vtrn.8  d0, d4                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"
          "vtrn.8  d8, d12                  @ trans data: d8=r20,r30,r12,r32 "
          "24 34 26 36  d12=21 31 23 33 \n"

          "vtrn.16 d0, d8                  @ trans data: d0=r00,r10,r20,r30 04 "
          "14 24 34 \n"
          "vtrn.16  d4, d12                  @ trans data: d4=01 11 21 31 05 "
          "15 25 35 \n"
          "vtrn.32 d0, d4                  @ trans data: d0=00,10,20,30 01 11 "
          "21 31 \n"
          "vtrn.32  d8, d12                  @ trans data: d8= 02 12 22 32 03 "
          "13 23 33\n"

          "vst1.32  {d0[0]},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 20 "
          "30\n"
          "vst1.32  {d0[1]},    [%[outptr1]]!   @ write d4(q0,low),r01,r11 21 "
          "31\n"
          "vst1.32  {d4[0]},    [%[outptr4]]!   @ write d2(q0,low),r02,r12\n"
          "vst1.32  {d4[1]},    [%[outptr5]]!   @ write d6(q0,low),r06,r16\n"
          "vst1.32  {d8[0]},    [%[outptr2]]!   @ write d1(q0,high),r02,r12 22 "
          "32\n"
          "vst1.32  {d8[1]},    [%[outptr3]]!   @ write d5(q0,low),r03,r13 23 "
          "33\n"
          "vst1.32 {d12[0]},     [%[outptr6]]!   @ write d3(q0,low),r03,r13\n"
          "vst1.32  {d12[1]},    [%[outptr7]]!   @ write d7(q0,low),r07,r17\n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7)
          :
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
    }
    for (; j < w_in; j++) {
      uint8_t* outptr = dst + (hout - j) * w_out + i;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
    }
  }
#endif
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * w_in;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + (hout - j) * w_out + i;
      *outptr0 = *inptr0++;
    }
  }
}

void rotate_hwc3_90(const uint8_t* src,
                    uint8_t* dst,
                    int w_in,
                    int h_in,
                    int w_out,
                    int h_out) {
  int win = w_in * 3;
  int wout = w_out * 3;
  int64_t stride_h = 4 * win;
  int64_t stride_h_w = 4 * win - 24;
  int ww = w_out - 8;
  uint8_t zerobuff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // block 4*8. -- 8*4
  int i = 0;
  for (i = 0; i < h_in - 7; i += 8) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      uint8_t* outptr0 = dst + j * wout + (ww - i) * 3;
      uint8_t* outptr1 = outptr0 + wout;
      uint8_t* outptr2 = outptr1 + wout;
      uint8_t* outptr3 = outptr2 + wout;
      uint8_t* outptr4 = outptr3 + wout;
      uint8_t* outptr5 = outptr4 + wout;
      uint8_t* outptr6 = outptr5 + wout;
      uint8_t* outptr7 = outptr6 + wout;
#ifdef __aarch64__
      asm volatile(
          "ld3  {v0.8b, v1.8b, v2.8b}, [%[inptr0]]    \n"  // v0={00,01,02, 03,
                                                           // 04, 05, 06, 07}"
          "ld3  {v3.8b, v4.8b, v5.8b}, [%[inptr1]]    \n"  // v0={10,11,12, 13,
                                                           // 14, 15, 16, 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23,
                                                           // 24, 25, 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]]    \n"  // v0={30,31,32,
                                                             // 33, 34, 35, 36,
                                                             // 37}"

          "add %[inptr0], %[inptr0], %[stride_h] \n"  // 4 + 4*w_in
          "add %[inptr1], %[inptr1], %[stride_h] \n"  // 5
          "add %[inptr2], %[inptr2], %[stride_h] \n"  // 6
          "add %[inptr3], %[inptr3], %[stride_h] \n"  // 7

          // b
          "trn1 v12.8b, v0.8b, v3.8b             \n"  // v4={00 10 02 12 04 14
                                                      // 06 16 }
          "trn1 v15.8b, v6.8b, v9.8b             \n"  // v4={20 30 22 32 24 34
                                                      // 26 36 }

          "trn2 v18.8b, v0.8b, v3.8b             \n"  // v5={01 11 03 13 05 15
                                                      // 07 17 }
          "trn2 v21.8b, v6.8b, v9.8b             \n"  // v7={21 31 23 33 25 35
                                                      // 27 37 }

          // g
          "trn1 v13.8b, v1.8b, v4.8b             \n"   // v4={00 10 02 12 04 14
                                                       // 06 16 }
          "trn1 v16.8b, v7.8b, v10.8b             \n"  // v4={20 30 22 32 24 34
                                                       // 26 36 }

          "trn2 v19.8b, v1.8b, v4.8b             \n"   // v5={01 11 03 13 05 15
                                                       // 07 17 }
          "trn2 v22.8b, v7.8b, v10.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }

          // r
          "trn1 v14.8b, v2.8b, v5.8b             \n"   // v4={00 10 02 12 04 14
                                                       // 06 16 }
          "trn1 v17.8b, v8.8b, v11.8b             \n"  // v4={20 30 22 32 24 34
                                                       // 26 36 }

          "trn2 v20.8b, v2.8b, v5.8b             \n"   // v5={01 11 03 13 05 15
                                                       // 07 17 }
          "trn2 v23.8b, v8.8b, v11.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }

          // b1
          "trn1 v24.4h, v12.4h, v15.4h             \n"  // v0={00 10 20 30 04 14
                                                        // 24 34}
          "trn1 v27.4h, v18.4h, v21.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          "trn2 v0.4h, v12.4h, v15.4h             \n"  // v1={02 12 22 32 06 16
                                                       // 26 36}
          "trn2 v3.4h, v18.4h, v21.4h             \n"  // v3={03 13 23 33 07 17
                                                       // 27 37}

          // g1
          "trn1 v25.4h, v13.4h, v16.4h             \n"  // v0={00 10 20 30 04 14
                                                        // 24 34}
          "trn1 v28.4h, v19.4h, v22.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          "trn2 v1.4h, v13.4h, v16.4h             \n"  // v1={02 12 22 32 06 16
                                                       // 26 36}
          "trn2 v4.4h, v19.4h, v22.4h             \n"  // v3={03 13 23 33 07 17
                                                       // 27 37}

          // r1
          "trn1 v26.4h, v14.4h, v17.4h             \n"  // v0={00 10 20 30 04 14
                                                        // 24 34}
          "trn1 v29.4h, v20.4h, v23.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          "trn2 v2.4h, v14.4h, v17.4h             \n"  // v1={02 12 22 32 06 16
                                                       // 26 36}
          "trn2 v5.4h, v20.4h, v23.4h             \n"  // v3={03 13 23 33 07 17
                                                       // 27 37}

          "ld3  {v12.8b, v13.8b, v14.8b}, [%[inptr0]]    \n"  // v0={00,01,02,
                                                              // 03, 04, 05, 06,
                                                              // 07}"
          "ld3  {v15.8b, v16.8b, v17.8b}, [%[inptr1]]    \n"  // v0={10,11,12,
                                                              // 13, 14, 15, 16,
                                                              // 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23,
                                                           // 24, 25, 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]]    \n"  // v0={30,31,32,
                                                             // 33, 34, 35, 36,
                                                             // 37}"

          "sub %[inptr0], %[inptr0], %[stride_h_w] \n"  // 4 - 4*w_in + 8
          "sub %[inptr1], %[inptr1], %[stride_h_w] \n"  // 5
          "sub %[inptr2], %[inptr2], %[stride_h_w] \n"  // 6
          "sub %[inptr3], %[inptr3], %[stride_h_w] \n"  // 7

          // b2
          "trn1 v18.8b, v12.8b, v15.8b             \n"  // v4={00 10 02 12 04 14
                                                        // 06 16 }
          "trn1 v21.8b, v6.8b, v9.8b             \n"    // v4={20 30 22 32 24 34
                                                        // 26 36 }
          // g2
          "trn1 v19.8b, v13.8b, v16.8b             \n"  // v4={00 10 02 12 04 14
                                                        // 06 16 }
          "trn1 v22.8b, v7.8b, v10.8b             \n"   // v4={20 30 22 32 24 34
                                                        // 26 36 }
          // r2
          "trn1 v20.8b, v14.8b, v17.8b             \n"  // v4={00 10 02 12 04 14
                                                        // 06 16 }
          "trn1 v23.8b, v8.8b, v11.8b             \n"   // v4={20 30 22 32 24 34
                                                        // 26 36 }

          "trn2 v12.8b, v12.8b, v15.8b             \n"  // v5={01 11 03 13 05 15
                                                        // 07 17 }
          "trn2 v13.8b, v13.8b, v16.8b             \n"  // v5={01 11 03 13 05 15
                                                        // 07 17 }
          "trn2 v14.8b, v14.8b, v17.8b             \n"  // v5={01 11 03 13 05 15
                                                        // 07 17 }

          "trn2 v15.8b, v6.8b, v9.8b             \n"   // v7={21 31 23 33 25 35
                                                       // 27 37 }
          "trn2 v16.8b, v7.8b, v10.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }
          "trn2 v17.8b, v8.8b, v11.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }

          // b2
          "trn1 v6.4h, v18.4h, v21.4h             \n"  // v0={00 10 20 30 04 14
                                                       // 24 34}
          // g2
          "trn1 v7.4h, v19.4h, v22.4h             \n"  // v0={00 10 20 30 04 14
                                                       // 24 34}
          // r2
          "trn1 v8.4h, v20.4h, v23.4h             \n"  // v0={00 10 20 30 04 14
                                                       // 24 34}

          // bgr
          "trn1 v9.4h, v12.4h, v15.4h             \n"   // v2={01 11 21 31 05 15
                                                        // 25 35}
          "trn1 v10.4h, v13.4h, v16.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}
          "trn1 v11.4h, v14.4h, v17.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          // bgr
          "trn2 v18.4h, v18.4h, v21.4h             \n"  // v1={02 12 22 32 06 16
                                                        // 26 36}
          "trn2 v19.4h, v19.4h, v22.4h             \n"  // v1={02 12 22 32 06 16
                                                        // 26 36}
          "trn2 v20.4h, v20.4h, v23.4h             \n"  // v1={02 12 22 32 06 16
                                                        // 26 36}

          // bgr
          "trn2 v21.4h, v12.4h, v15.4h             \n"  // v3={03 13 23 33 07 17
                                                        // 27 37}
          "trn2 v22.4h, v13.4h, v16.4h             \n"  // v3={03 13 23 33 07 17
                                                        // 27 37}
          "trn2 v23.4h, v14.4h, v17.4h             \n"  // v3={03 13 23 33 07 17
                                                        // 27 37}

          // b1 b2
          "trn1 v12.2s, v24.2s, v6.2s             \n"  // v8={00 10 20 30 40 50
                                                       // 60 70} b
          "trn1 v13.2s, v25.2s, v7.2s             \n"  // v6={00 10 20 30 40 50
                                                       // 60 70} g
          "trn1 v14.2s, v26.2s, v8.2s             \n"  // v6={00 10 20 30 40 50
                                                       // 60 70} r

          // b1 b2
          "trn2 v15.2s, v24.2s, v6.2s             \n"  // v8={04 14 24 34 44 54
                                                       // 64 74} b
          "trn2 v16.2s, v25.2s, v7.2s             \n"  // v6={04 14 24 34 44 54
                                                       // 64 74} g
          "trn2 v17.2s, v26.2s, v8.2s             \n"  // v6={04 14 24 34 44 54
                                                       // 64 74} r

          // b1 b2
          "trn1 v6.2s, v27.2s, v9.2s             \n"   // v8={01 11 20 30 40 50
                                                       // 60 70} b
          "trn1 v7.2s, v28.2s, v10.2s             \n"  // v6={01 10 20 30 40 50
                                                       // 60 70} g
          "trn1 v8.2s, v29.2s, v11.2s             \n"  // v6={01 10 20 30 40 50
                                                       // 60 70} r

          "rev64  v12.8b, v12.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 b
          "rev64  v13.8b, v13.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 g
          "rev64  v14.8b, v14.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 r
          "rev64  v15.8b, v15.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 b
          "rev64  v16.8b, v16.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 g
          "rev64  v17.8b, v17.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 r

          // b1 b2
          "trn2 v24.2s, v27.2s, v9.2s             \n"   // v8={05 10 20 30 40 50
                                                        // 60 70} b
          "trn2 v25.2s, v28.2s, v10.2s             \n"  // v6={05 10 20 30 40 50
                                                        // 60 70} g
          "trn2 v26.2s, v29.2s, v11.2s             \n"  // v6={05 10 20 30 40 50
                                                        // 60 70} r

          "st3 {v12.8b, v13.8b, v14.8b}, [%[outptr0]], #24             \n"  // 00 10 20 30 04 14 24 34
          "st3 {v15.8b, v16.8b, v17.8b}, [%[outptr4]], #24             \n"  // 02 12 22 32
          // b1 b2
          "trn1 v9.2s, v0.2s, v18.2s             \n"   // v8={02 11 20 30 40 50
                                                       // 60 70} b
          "trn1 v10.2s, v1.2s, v19.2s             \n"  // v6={02 10 20 30 40 50
                                                       // 60 70} g
          "trn1 v11.2s, v2.2s, v20.2s             \n"  // v6={02 10 20 30 40 50
                                                       // 60 70} r

          "trn2 v27.2s, v0.2s, v18.2s             \n"  // v8={06 11 20 30 40 50
                                                       // 60 70} b
          "trn2 v28.2s, v1.2s, v19.2s             \n"  // v6={06 10 20 30 40 50
                                                       // 60 70} g
          "trn2 v29.2s, v2.2s, v20.2s             \n"  // v6={06 10 20 30 40 50
                                                       // 60 70} r

          // b1 b2
          "trn1 v0.2s, v3.2s, v21.2s             \n"  // v8={03 11 20 30 40 50
                                                      // 60 70} b
          "trn1 v1.2s, v4.2s, v22.2s             \n"  // v6={03 10 20 30 40 50
                                                      // 60 70} g
          "trn1 v2.2s, v5.2s, v23.2s             \n"  // v6={03 10 20 30 40 50
                                                      // 60 70} r

          "trn2 v18.2s, v3.2s, v21.2s             \n"  // v8={07 11 20 30 40 50
                                                       // 60 70} b
          "trn2 v19.2s, v4.2s, v22.2s             \n"  // v6={07 10 20 30 40 50
                                                       // 60 70} g
          "trn2 v20.2s, v5.2s, v23.2s             \n"  // v6={07 10 20 30 40 50
                                                       // 60 70} r

          "rev64  v6.8b, v6.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 b
          "rev64  v7.8b, v7.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 g
          "rev64  v8.8b, v8.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 r

          "rev64  v24.8b, v24.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 b
          "rev64  v25.8b, v25.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 g
          "rev64  v26.8b, v26.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 r

          "rev64  v9.8b, v9.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 b
          "rev64  v10.8b, v10.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 g
          "rev64  v11.8b, v11.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 r

          "rev64  v27.8b, v27.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 b
          "rev64  v28.8b, v28.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 g
          "rev64  v29.8b, v29.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 r

          "rev64  v0.8b, v0.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 b
          "rev64  v1.8b, v1.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 g
          "rev64  v2.8b, v2.8b                \n"  //@ reverse 07 06 05 04 03 02
                                                   // 01 00 r

          "rev64  v18.8b, v18.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 b
          "rev64  v19.8b, v19.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 g
          "rev64  v20.8b, v20.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00 r

          "st3 {v6.8b, v7.8b, v8.8b}, [%[outptr1]], #24             \n"  // 00
                                                                         // 10
                                                                         // 20
                                                                         // 30
                                                                         // 04
                                                                         // 14
                                                                         // 24
                                                                         // 34
          "st3 {v24.8b, v25.8b, v26.8b}, [%[outptr5]], #24             \n"  // 02 12 22 32

          "st3 {v9.8b, v10.8b, v11.8b}, [%[outptr2]], #24             \n"  // 00
                                                                           // 10
                                                                           // 20
                                                                           // 30
                                                                           // 04
                                                                           // 14
                                                                           // 24
                                                                           // 34
          "st3 {v27.8b, v28.8b, v29.8b}, [%[outptr6]], #24             \n"  // 02 12 22 32

          "st3 {v0.8b, v1.8b, v2.8b}, [%[outptr3]], #24             \n"  // 00
                                                                         // 10
                                                                         // 20
                                                                         // 30
                                                                         // 04
                                                                         // 14
                                                                         // 24
                                                                         // 34
          "st3 {v18.8b, v19.8b, v20.8b}, [%[outptr7]], #24             \n"  // 02 12 22 32

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7),
            [stride_h] "+r"(stride_h),
            [stride_h_w] "+r"(stride_h_w)
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
            "v23",
            "v24",
            "v25",
            "v26",
            "v27",
            "v28",
            "v29",
            "v30");
#else
      asm volatile(
          "vld3.8  {d0, d1, d2}, [%[inptr0]]   @ zip load r0, d0 =00 01 02 03 "
          "04 05 06 07\n"
          "vld3.8  {d3, d4, d5}, [%[inptr1]]   @ zip load r1, d2 =10 11 12 13 "
          "14 15 16 17\n"
          "vld3.8  {d6, d7, d8}, [%[inptr2]]   @ zip load r1, d4 =20 21 22 23 "
          "24 25 26 27\n"
          "vld3.8  {d9, d10, d11}, [%[inptr3]]   @ zip load r1, d6 = 30 31 32 "
          "33 34 35 36 37\n"

          "add %[inptr0], %[stride_h]             @ add\n"  // 4 + 4*w_in
          "add %[inptr1], %[stride_h]             @ add\n"  // 5
          "add %[inptr2], %[stride_h]             @ add\n"  // 6
          "add %[inptr3], %[stride_h]             @ add\n"  // 7

          "vtrn.8  d0, d3                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"
          "vtrn.8  d1, d4                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"
          "vtrn.8  d2, d5                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"

          "vtrn.8  d6, d9                  @ trans data: d8=r20,r30,r12,r32 24 "
          "34 26 36  d12=21 31 23 33 \n"
          "vtrn.8  d7, d10                  @ trans data: d8=r20,r30,r12,r32 "
          "24 34 26 36  d12=21 31 23 33 \n"
          "vtrn.8  d8, d11                  @ trans data: d8=r20,r30,r12,r32 "
          "24 34 26 36  d12=21 31 23 33 \n"

          "vld3.8  {d12, d13, d14}, [%[inptr0]]   @ zip load r0, d0 = 40 41 02 "
          "03 04 05 06 07\n"
          "vld3.8  {d15, d16, d17}, [%[inptr1]]   @ zip load r1, d2 = 50 51 12 "
          "13 14 15 16 17\n"
          "vld3.8  {d18, d19, d20}, [%[inptr2]]   @ zip load r1, d4 = 60 61 22 "
          "23 24 25 26 27\n"
          "vld3.8  {d21, d22, d23}, [%[inptr3]]   @ zip load r1, d6 = 70 71 32 "
          "33 34 35 36 37\n"

          // d0=r00,r10,r20,r30 04 14 24 34 d8 = 02 12 22 32 06 16 26 36
          "vtrn.16 d0, d6                  @ trans data: \n"
          "vtrn.16 d1, d7                  @ trans data: \n"
          "vtrn.16 d2, d8                  @ trans data: \n"
          // d4=01 11 21 31 05 15 25 35; d12 =r03 13 23 33 07 17 27 37
          "vtrn.16  d3, d9                  @ trans data:\n"
          "vtrn.16  d4, d10                  @ trans data:\n"
          "vtrn.16  d5, d11                  @ trans data:\n"

          "vtrn.8  d12, d15                  @ trans data: d0=r40,r50,r42,r52 "
          "04 14 06 16  d4=41 51 03 13 \n"
          "vtrn.8  d13, d16                  @ trans data: d0=r40,r50,r42,r52 "
          "04 14 06 16  d4=41 51 03 13 \n"
          "vtrn.8  d14, d17                  @ trans data: d0=r40,r50,r42,r52 "
          "04 14 06 16  d4=41 51 03 13 \n"

          "vtrn.8  d18, d21                  @ trans data: d8=r60,r70,r12,r32 "
          "24 34 26 36  d12=61 71 23 33 \n"
          "vtrn.8  d19, d22                  @ trans data: d8=r60,r70,r12,r32 "
          "24 34 26 36  d12=61 71 23 33 \n"
          "vtrn.8  d20, d23                  @ trans data: d8=r60,r70,r12,r32 "
          "24 34 26 36  d12=61 71 23 33 \n"

          // d0=r40,r50,r60,r70 44 54 64 74 d8 = 42 52 62 72 46 56 66 76
          "vtrn.16 d12, d18                  @ trans data: \n"
          "vtrn.16 d13, d19                  @ trans data: \n"
          "vtrn.16 d14, d20                  @ trans data: \n"
          // d4=41 51 61 71 45 55 65 75; d12 = 43 53 63 73 47 57 67 77
          "vtrn.16  d15, d21                  @ trans data:\n"
          "vtrn.16  d16, d22                  @ trans data:\n"
          "vtrn.16  d17, d23                  @ trans data:\n"

          // d0=00,10,20,30 40 50 60 70 d4 = 04 14 24 34 44 54 64 74
          "vtrn.32 d0, d12                  @ trans data: \n"
          "vtrn.32 d1, d13                  @ trans data: \n"
          "vtrn.32 d2, d14                  @ trans data: \n"

          // d0=02,12,22,32 d4 = 06 26 16 36
          "vtrn.32 d6, d18                  @ trans data: \n"
          "vtrn.32 d7, d19                  @ trans data: \n"
          "vtrn.32 d8, d20                  @ trans data: \n"

          // d4=01 11 21 31 41 51 61 71; d12 = 05 15 25 35 45 55 65 75
          "vtrn.32  d3, d15                  @ trans data: \n"
          "vtrn.32  d4, d16                  @ trans data: \n"
          "vtrn.32  d5, d17                  @ trans data: \n"

          // d0= 03 13 23 33  d4 = 07 17 27 37
          "vtrn.32 d9, d21                  @ trans data: \n"
          "vtrn.32 d10, d22                  @ trans data: \n"
          "vtrn.32 d11, d23                  @ trans data: \n"

          "sub %[inptr0], %[inptr0], %[stride_h_w] \n"  // 4 - 4*w_in + 8
          "sub %[inptr1], %[inptr1], %[stride_h_w] \n"  // 5
          "sub %[inptr2], %[inptr2], %[stride_h_w] \n"  // 6
          "sub %[inptr3], %[inptr3], %[stride_h_w] \n"  // 7

          "vst3.8  {d0, d1, d2},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d12, d13, d14},    [%[outptr4]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          "vst3.8  {d6, d7, d8},    [%[outptr2]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d18, d19, d20},    [%[outptr6]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          "vst3.8  {d3, d4, d5},    [%[outptr1]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d15, d16, d17},    [%[outptr5]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          "vst3.8  {d9, d10, d11},    [%[outptr3]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"
          "vst3.8  {d21, d22, d23},    [%[outptr7]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7)
          : [stride_h] "r"(stride_h), [stride_h_w] "r"(stride_h_w)
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
#endif
    }
    const uint8_t* inptr4 = inptr3 + win;
    const uint8_t* inptr5 = inptr4 + win;
    const uint8_t* inptr6 = inptr5 + win;
    const uint8_t* inptr7 = inptr6 + win;
    for (; j < w_in; j++) {
      int tmpx = (ww - i) * 3;
      uint8_t* outptr = dst + j * wout + tmpx;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;

      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;

      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;

      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;

      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;

      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;

      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;

      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
    }
  }
  // remain
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * w_in;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + j * w_out + i;
      *outptr0 = *inptr0++;
    }
  }
}

void rotate_hwc3_180(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  int win = w_in * 3;
  uint8_t zerobuff[30000];
  memset(zerobuff, 0, win * sizeof(uint8_t));
  int64_t stride_w = 24;
  // printf("src: %x, dst: %x \n", src, dst);
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    uint8_t* outptr0 = dst + (h_in - i) * win - stride_w;  // last col
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
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      if (i + 3 >= h_in) {
        switch ((i + 3) - h_in) {
          case 3:
            inptr0 = zerobuff;
            outptr0 = zerobuff;
          case 2:
            inptr1 = zerobuff;
            outptr1 = zerobuff;
          case 1:
            inptr2 = zerobuff;
            outptr2 = zerobuff;
          case 0:
            inptr3 = zerobuff;
            outptr3 = zerobuff;
          default:
            break;
        }
      }
#ifdef __aarch64__
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
#else
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
#endif
    }
    outptr3 += stride_w - 3;
    outptr2 += stride_w - 3;
    outptr1 += stride_w - 3;
    outptr0 += stride_w - 3;
    for (; j < w_in; j++) {
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

void rotate_hwc3_270(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  int win = w_in * 3;
  int wout = w_out * 3;
  int64_t stride_h = 4 * win;
  int64_t stride_h_w = 4 * win - 24;
  int hout = h_out - 1;
  // block 8*8. -- 8*8
  int i = 0;
  for (; i < h_in - 7; i += 8) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    int j = 0;
    for (; j < w_in - 7; j += 8) {
      uint8_t* outptr0 = dst + (hout - j) * wout + i * 3;
      uint8_t* outptr1 = outptr0 - wout;
      uint8_t* outptr2 = outptr1 - wout;
      uint8_t* outptr3 = outptr2 - wout;
      uint8_t* outptr4 = outptr3 - wout;
      uint8_t* outptr5 = outptr4 - wout;
      uint8_t* outptr6 = outptr5 - wout;
      uint8_t* outptr7 = outptr6 - wout;
#ifdef __aarch64__
      asm volatile(
          "ld3  {v0.8b, v1.8b, v2.8b}, [%[inptr0]]    \n"  // v0={00,01,02, 03,
                                                           // 04, 05, 06, 07}"
          "ld3  {v3.8b, v4.8b, v5.8b}, [%[inptr1]]    \n"  // v0={10,11,12, 13,
                                                           // 14, 15, 16, 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23,
                                                           // 24, 25, 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]]    \n"  // v0={30,31,32,
                                                             // 33, 34, 35, 36,
                                                             // 37}"

          "add %[inptr0], %[inptr0], %[stride_h] \n"  // 4 + 4*w_in
          "add %[inptr1], %[inptr1], %[stride_h] \n"  // 5
          "add %[inptr2], %[inptr2], %[stride_h] \n"  // 6
          "add %[inptr3], %[inptr3], %[stride_h] \n"  // 7

          // b
          "trn1 v12.8b, v0.8b, v3.8b             \n"  // v4={00 10 02 12 04 14
                                                      // 06 16 }
          "trn1 v15.8b, v6.8b, v9.8b             \n"  // v4={20 30 22 32 24 34
                                                      // 26 36 }

          "trn2 v18.8b, v0.8b, v3.8b             \n"  // v5={01 11 03 13 05 15
                                                      // 07 17 }
          "trn2 v21.8b, v6.8b, v9.8b             \n"  // v7={21 31 23 33 25 35
                                                      // 27 37 }

          // g
          "trn1 v13.8b, v1.8b, v4.8b             \n"   // v4={00 10 02 12 04 14
                                                       // 06 16 }
          "trn1 v16.8b, v7.8b, v10.8b             \n"  // v4={20 30 22 32 24 34
                                                       // 26 36 }

          "trn2 v19.8b, v1.8b, v4.8b             \n"   // v5={01 11 03 13 05 15
                                                       // 07 17 }
          "trn2 v22.8b, v7.8b, v10.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }

          // r
          "trn1 v14.8b, v2.8b, v5.8b             \n"   // v4={00 10 02 12 04 14
                                                       // 06 16 }
          "trn1 v17.8b, v8.8b, v11.8b             \n"  // v4={20 30 22 32 24 34
                                                       // 26 36 }

          "trn2 v20.8b, v2.8b, v5.8b             \n"   // v5={01 11 03 13 05 15
                                                       // 07 17 }
          "trn2 v23.8b, v8.8b, v11.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }

          // b1
          "trn1 v24.4h, v12.4h, v15.4h             \n"  // v0={00 10 20 30 04 14
                                                        // 24 34}
          "trn1 v27.4h, v18.4h, v21.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          "trn2 v0.4h, v12.4h, v15.4h             \n"  // v1={02 12 22 32 06 16
                                                       // 26 36}
          "trn2 v3.4h, v18.4h, v21.4h             \n"  // v3={03 13 23 33 07 17
                                                       // 27 37}

          // g1
          "trn1 v25.4h, v13.4h, v16.4h             \n"  // v0={00 10 20 30 04 14
                                                        // 24 34}
          "trn1 v28.4h, v19.4h, v22.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          "trn2 v1.4h, v13.4h, v16.4h             \n"  // v1={02 12 22 32 06 16
                                                       // 26 36}
          "trn2 v4.4h, v19.4h, v22.4h             \n"  // v3={03 13 23 33 07 17
                                                       // 27 37}

          // r1
          "trn1 v26.4h, v14.4h, v17.4h             \n"  // v0={00 10 20 30 04 14
                                                        // 24 34}
          "trn1 v29.4h, v20.4h, v23.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          "trn2 v2.4h, v14.4h, v17.4h             \n"  // v1={02 12 22 32 06 16
                                                       // 26 36}
          "trn2 v5.4h, v20.4h, v23.4h             \n"  // v3={03 13 23 33 07 17
                                                       // 27 37}

          "ld3  {v12.8b, v13.8b, v14.8b}, [%[inptr0]]    \n"  // v0={00,01,02,
                                                              // 03, 04, 05, 06,
                                                              // 07}"
          "ld3  {v15.8b, v16.8b, v17.8b}, [%[inptr1]]    \n"  // v0={10,11,12,
                                                              // 13, 14, 15, 16,
                                                              // 17}"
          "ld3  {v6.8b, v7.8b, v8.8b}, [%[inptr2]]    \n"  // v0={20,21,22, 23,
                                                           // 24, 25, 26, 27}"
          "ld3  {v9.8b, v10.8b, v11.8b}, [%[inptr3]]    \n"  // v0={30,31,32,
                                                             // 33, 34, 35, 36,
                                                             // 37}"

          "sub %[inptr0], %[inptr0], %[stride_h_w] \n"  // 4 - 4*w_in + 8
          "sub %[inptr1], %[inptr1], %[stride_h_w] \n"  // 5
          "sub %[inptr2], %[inptr2], %[stride_h_w] \n"  // 6
          "sub %[inptr3], %[inptr3], %[stride_h_w] \n"  // 7

          // b2
          "trn1 v18.8b, v12.8b, v15.8b             \n"  // v4={00 10 02 12 04 14
                                                        // 06 16 }
          "trn1 v21.8b, v6.8b, v9.8b             \n"    // v4={20 30 22 32 24 34
                                                        // 26 36 }
          // g2
          "trn1 v19.8b, v13.8b, v16.8b             \n"  // v4={00 10 02 12 04 14
                                                        // 06 16 }
          "trn1 v22.8b, v7.8b, v10.8b             \n"   // v4={20 30 22 32 24 34
                                                        // 26 36 }
          // r2
          "trn1 v20.8b, v14.8b, v17.8b             \n"  // v4={00 10 02 12 04 14
                                                        // 06 16 }
          "trn1 v23.8b, v8.8b, v11.8b             \n"   // v4={20 30 22 32 24 34
                                                        // 26 36 }

          "trn2 v12.8b, v12.8b, v15.8b             \n"  // v5={01 11 03 13 05 15
                                                        // 07 17 }
          "trn2 v13.8b, v13.8b, v16.8b             \n"  // v5={01 11 03 13 05 15
                                                        // 07 17 }
          "trn2 v14.8b, v14.8b, v17.8b             \n"  // v5={01 11 03 13 05 15
                                                        // 07 17 }

          "trn2 v15.8b, v6.8b, v9.8b             \n"   // v7={21 31 23 33 25 35
                                                       // 27 37 }
          "trn2 v16.8b, v7.8b, v10.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }
          "trn2 v17.8b, v8.8b, v11.8b             \n"  // v7={21 31 23 33 25 35
                                                       // 27 37 }

          // b2
          "trn1 v6.4h, v18.4h, v21.4h             \n"  // v0={00 10 20 30 04 14
                                                       // 24 34}
          // g2
          "trn1 v7.4h, v19.4h, v22.4h             \n"  // v0={00 10 20 30 04 14
                                                       // 24 34}
          // r2
          "trn1 v8.4h, v20.4h, v23.4h             \n"  // v0={00 10 20 30 04 14
                                                       // 24 34}

          // bgr
          "trn1 v9.4h, v12.4h, v15.4h             \n"   // v2={01 11 21 31 05 15
                                                        // 25 35}
          "trn1 v10.4h, v13.4h, v16.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}
          "trn1 v11.4h, v14.4h, v17.4h             \n"  // v2={01 11 21 31 05 15
                                                        // 25 35}

          // bgr
          "trn2 v18.4h, v18.4h, v21.4h             \n"  // v1={02 12 22 32 06 16
                                                        // 26 36}
          "trn2 v19.4h, v19.4h, v22.4h             \n"  // v1={02 12 22 32 06 16
                                                        // 26 36}
          "trn2 v20.4h, v20.4h, v23.4h             \n"  // v1={02 12 22 32 06 16
                                                        // 26 36}

          // bgr
          "trn2 v21.4h, v12.4h, v15.4h             \n"  // v3={03 13 23 33 07 17
                                                        // 27 37}
          "trn2 v22.4h, v13.4h, v16.4h             \n"  // v3={03 13 23 33 07 17
                                                        // 27 37}
          "trn2 v23.4h, v14.4h, v17.4h             \n"  // v3={03 13 23 33 07 17
                                                        // 27 37}

          // b1 b2
          "trn1 v12.2s, v24.2s, v6.2s             \n"  // v8={00 10 20 30 40 50
                                                       // 60 70} b
          "trn1 v13.2s, v25.2s, v7.2s             \n"  // v6={00 10 20 30 40 50
                                                       // 60 70} g
          "trn1 v14.2s, v26.2s, v8.2s             \n"  // v6={00 10 20 30 40 50
                                                       // 60 70} r

          // b1 b2
          "trn2 v15.2s, v24.2s, v6.2s             \n"  // v8={04 14 24 34 44 54
                                                       // 64 74} b
          "trn2 v16.2s, v25.2s, v7.2s             \n"  // v6={04 14 24 34 44 54
                                                       // 64 74} g
          "trn2 v17.2s, v26.2s, v8.2s             \n"  // v6={04 14 24 34 44 54
                                                       // 64 74} r

          // b1 b2
          "trn1 v6.2s, v27.2s, v9.2s             \n"   // v8={01 11 20 30 40 50
                                                       // 60 70} b
          "trn1 v7.2s, v28.2s, v10.2s             \n"  // v6={01 10 20 30 40 50
                                                       // 60 70} g
          "trn1 v8.2s, v29.2s, v11.2s             \n"  // v6={01 10 20 30 40 50
                                                       // 60 70} r

          // b1 b2
          "trn2 v24.2s, v27.2s, v9.2s             \n"   // v8={05 10 20 30 40 50
                                                        // 60 70} b
          "trn2 v25.2s, v28.2s, v10.2s             \n"  // v6={05 10 20 30 40 50
                                                        // 60 70} g
          "trn2 v26.2s, v29.2s, v11.2s             \n"  // v6={05 10 20 30 40 50
                                                        // 60 70} r

          "st3 {v12.8b, v13.8b, v14.8b}, [%[outptr0]], #24             \n"  // 00 10 20 30 04 14 24 34
          "st3 {v15.8b, v16.8b, v17.8b}, [%[outptr4]], #24             \n"  // 02 12 22 32
          // b1 b2
          "trn1 v9.2s, v0.2s, v18.2s             \n"   // v8={02 11 20 30 40 50
                                                       // 60 70} b
          "trn1 v10.2s, v1.2s, v19.2s             \n"  // v6={02 10 20 30 40 50
                                                       // 60 70} g
          "trn1 v11.2s, v2.2s, v20.2s             \n"  // v6={02 10 20 30 40 50
                                                       // 60 70} r

          "trn2 v27.2s, v0.2s, v18.2s             \n"  // v8={06 11 20 30 40 50
                                                       // 60 70} b
          "trn2 v28.2s, v1.2s, v19.2s             \n"  // v6={06 10 20 30 40 50
                                                       // 60 70} g
          "trn2 v29.2s, v2.2s, v20.2s             \n"  // v6={06 10 20 30 40 50
                                                       // 60 70} r

          // b1 b2
          "trn1 v0.2s, v3.2s, v21.2s             \n"  // v8={03 11 20 30 40 50
                                                      // 60 70} b
          "trn1 v1.2s, v4.2s, v22.2s             \n"  // v6={03 10 20 30 40 50
                                                      // 60 70} g
          "trn1 v2.2s, v5.2s, v23.2s             \n"  // v6={03 10 20 30 40 50
                                                      // 60 70} r

          "trn2 v18.2s, v3.2s, v21.2s             \n"  // v8={07 11 20 30 40 50
                                                       // 60 70} b
          "trn2 v19.2s, v4.2s, v22.2s             \n"  // v6={07 10 20 30 40 50
                                                       // 60 70} g
          "trn2 v20.2s, v5.2s, v23.2s             \n"  // v6={07 10 20 30 40 50
                                                       // 60 70} r

          "st3 {v6.8b, v7.8b, v8.8b}, [%[outptr1]], #24             \n"  // 00
                                                                         // 10
                                                                         // 20
                                                                         // 30
                                                                         // 04
                                                                         // 14
                                                                         // 24
                                                                         // 34
          "st3 {v24.8b, v25.8b, v26.8b}, [%[outptr5]], #24             \n"  // 02 12 22 32

          "st3 {v9.8b, v10.8b, v11.8b}, [%[outptr2]], #24             \n"  // 00
                                                                           // 10
                                                                           // 20
                                                                           // 30
                                                                           // 04
                                                                           // 14
                                                                           // 24
                                                                           // 34
          "st3 {v27.8b, v28.8b, v29.8b}, [%[outptr6]], #24             \n"  // 02 12 22 32

          "st3 {v0.8b, v1.8b, v2.8b}, [%[outptr3]], #24             \n"  // 00
                                                                         // 10
                                                                         // 20
                                                                         // 30
                                                                         // 04
                                                                         // 14
                                                                         // 24
                                                                         // 34
          "st3 {v18.8b, v19.8b, v20.8b}, [%[outptr7]], #24             \n"  // 02 12 22 32

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7),
            [stride_h] "+r"(stride_h),
            [stride_h_w] "+r"(stride_h_w)
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
            "v23",
            "v24",
            "v25",
            "v26",
            "v27",
            "v28",
            "v29");
#else
      asm volatile(
          "vld3.8  {d0, d1, d2}, [%[inptr0]]   @ zip load r0, d0 =00 01 02 03 "
          "04 05 06 07\n"
          "vld3.8  {d3, d4, d5}, [%[inptr1]]   @ zip load r1, d2 =10 11 12 13 "
          "14 15 16 17\n"
          "vld3.8  {d6, d7, d8}, [%[inptr2]]   @ zip load r1, d4 =20 21 22 23 "
          "24 25 26 27\n"
          "vld3.8  {d9, d10, d11}, [%[inptr3]]   @ zip load r1, d6 = 30 31 32 "
          "33 34 35 36 37\n"

          "add %[inptr0], %[stride_h]             @ add\n"  // 4 + 4*w_in
          "add %[inptr1], %[stride_h]             @ add\n"  // 5
          "add %[inptr2], %[stride_h]             @ add\n"  // 6
          "add %[inptr3], %[stride_h]             @ add\n"  // 7

          "vtrn.8  d0, d3                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"
          "vtrn.8  d1, d4                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"
          "vtrn.8  d2, d5                  @ trans data: d0=r00,r10,r02,r12 04 "
          "14 06 16  d4=01 11 03 13 \n"

          "vtrn.8  d6, d9                  @ trans data: d8=r20,r30,r12,r32 24 "
          "34 26 36  d12=21 31 23 33 \n"
          "vtrn.8  d7, d10                  @ trans data: d8=r20,r30,r12,r32 "
          "24 34 26 36  d12=21 31 23 33 \n"
          "vtrn.8  d8, d11                  @ trans data: d8=r20,r30,r12,r32 "
          "24 34 26 36  d12=21 31 23 33 \n"

          "vld3.8  {d12, d13, d14}, [%[inptr0]]   @ zip load r0, d0 = 40 41 02 "
          "03 04 05 06 07\n"
          "vld3.8  {d15, d16, d17}, [%[inptr1]]   @ zip load r1, d2 = 50 51 12 "
          "13 14 15 16 17\n"
          "vld3.8  {d18, d19, d20}, [%[inptr2]]   @ zip load r1, d4 = 60 61 22 "
          "23 24 25 26 27\n"
          "vld3.8  {d21, d22, d23}, [%[inptr3]]   @ zip load r1, d6 = 70 71 32 "
          "33 34 35 36 37\n"

          // d0=r00,r10,r20,r30 04 14 24 34 d8 = 02 12 22 32 06 16 26 36
          "vtrn.16 d0, d6                  @ trans data: \n"
          "vtrn.16 d1, d7                  @ trans data: \n"
          "vtrn.16 d2, d8                  @ trans data: \n"
          // d4=01 11 21 31 05 15 25 35; d12 =r03 13 23 33 07 17 27 37
          "vtrn.16  d3, d9                  @ trans data:\n"
          "vtrn.16  d4, d10                  @ trans data:\n"
          "vtrn.16  d5, d11                  @ trans data:\n"

          "vtrn.8  d12, d15                  @ trans data: d0=r40,r50,r42,r52 "
          "04 14 06 16  d4=41 51 03 13 \n"
          "vtrn.8  d13, d16                  @ trans data: d0=r40,r50,r42,r52 "
          "04 14 06 16  d4=41 51 03 13 \n"
          "vtrn.8  d14, d17                  @ trans data: d0=r40,r50,r42,r52 "
          "04 14 06 16  d4=41 51 03 13 \n"

          "vtrn.8  d18, d21                  @ trans data: d8=r60,r70,r12,r32 "
          "24 34 26 36  d12=61 71 23 33 \n"
          "vtrn.8  d19, d22                  @ trans data: d8=r60,r70,r12,r32 "
          "24 34 26 36  d12=61 71 23 33 \n"
          "vtrn.8  d20, d23                  @ trans data: d8=r60,r70,r12,r32 "
          "24 34 26 36  d12=61 71 23 33 \n"

          // d0=r40,r50,r60,r70 44 54 64 74 d8 = 42 52 62 72 46 56 66 76
          "vtrn.16 d12, d18                  @ trans data: \n"
          "vtrn.16 d13, d19                  @ trans data: \n"
          "vtrn.16 d14, d20                  @ trans data: \n"
          // d4=41 51 61 71 45 55 65 75; d12 = 43 53 63 73 47 57 67 77
          "vtrn.16  d15, d21                  @ trans data:\n"
          "vtrn.16  d16, d22                  @ trans data:\n"
          "vtrn.16  d17, d23                  @ trans data:\n"

          // d0=00,10,20,30 40 50 60 70 d4 = 04 14 24 34 44 54 64 74
          "vtrn.32 d0, d12                  @ trans data: \n"
          "vtrn.32 d1, d13                  @ trans data: \n"
          "vtrn.32 d2, d14                  @ trans data: \n"

          // d0=02,12,22,32 d4 = 06 26 16 36
          "vtrn.32 d6, d18                  @ trans data: \n"
          "vtrn.32 d7, d19                  @ trans data: \n"
          "vtrn.32 d8, d20                  @ trans data: \n"

          // d4=01 11 21 31 41 51 61 71; d12 = 05 15 25 35 45 55 65 75
          "vtrn.32  d3, d15                  @ trans data: \n"
          "vtrn.32  d4, d16                  @ trans data: \n"
          "vtrn.32  d5, d17                  @ trans data: \n"

          // d0= 03 13 23 33  d4 = 07 17 27 37
          "vtrn.32 d9, d21                  @ trans data: \n"
          "vtrn.32 d10, d22                  @ trans data: \n"
          "vtrn.32 d11, d23                  @ trans data: \n"

          "sub %[inptr0], %[inptr0], %[stride_h_w] \n"  // 4 - 4*w_in + 8
          "sub %[inptr1], %[inptr1], %[stride_h_w] \n"  // 5
          "sub %[inptr2], %[inptr2], %[stride_h_w] \n"  // 6
          "sub %[inptr3], %[inptr3], %[stride_h_w] \n"  // 7

          "vst3.8  {d0, d1, d2},    [%[outptr0]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d12, d13, d14},    [%[outptr4]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          "vst3.8  {d6, d7, d8},    [%[outptr2]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d18, d19, d20},    [%[outptr6]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          "vst3.8  {d3, d4, d5},    [%[outptr1]]!   @ write d0(q0,low),r00,r10 "
          "20 30\n"
          "vst3.8  {d15, d16, d17},    [%[outptr5]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          "vst3.8  {d9, d10, d11},    [%[outptr3]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"
          "vst3.8  {d21, d22, d23},    [%[outptr7]]!   @ write "
          "d0(q0,low),r00,r10 20 30\n"

          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [outptr0] "+r"(outptr0),
            [outptr1] "+r"(outptr1),
            [outptr2] "+r"(outptr2),
            [outptr3] "+r"(outptr3),
            [outptr4] "+r"(outptr4),
            [outptr5] "+r"(outptr5),
            [outptr6] "+r"(outptr6),
            [outptr7] "+r"(outptr7)
          : [stride_h] "r"(stride_h), [stride_h_w] "r"(stride_h_w)
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
#endif
    }
    const uint8_t* inptr4 = inptr3 + win;
    const uint8_t* inptr5 = inptr4 + win;
    const uint8_t* inptr6 = inptr5 + win;
    const uint8_t* inptr7 = inptr6 + win;
    for (; j < w_in; j++) {
      int tmpx = i * 3;
      uint8_t* outptr = dst + (hout - j) * wout + tmpx;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;
      *outptr++ = *inptr0++;

      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr1++;

      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr2++;

      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr3++;

      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr4++;

      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr5++;

      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr6++;

      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr7++;
    }
  }
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * win;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + (hout - j) * wout + i * 3;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
    }
  }
}

void rotate_hwc4_90(const uint8_t* src,
                    uint8_t* dst,
                    int w_in,
                    int h_in,
                    int w_out,
                    int h_out) {
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

void rotate_hwc4_180(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  int win = w_in * 4;
  uint8_t zerobuff[40000];
  memset(zerobuff, 0, win * sizeof(uint8_t));
  int stride_w = 4;
  // 4*8
  for (int i = 0; i < h_in; i += 4) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    uint8_t* outptr0 = dst + (h_in - i) * win - stride_w;  // last
    uint8_t* outptr1 = outptr0 - win;
    uint8_t* outptr2 = outptr1 - win;
    uint8_t* outptr3 = outptr2 - win;

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
    int j = 0;
    for (; j < w_in; j++) {
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

void rotate_hwc4_270(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
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

void rotate_hwc1(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree) {
  if (degree == 90) {
    rotate_hwc1_90(src, dst, srcw, srch, srch, srcw);
  } else if (degree == 180) {
    rotate_hwc1_180(src, dst, srcw, srch, srcw, srch);
  } else if (degree == 270) {
    rotate_hwc1_270(src, dst, srcw, srch, srch, srcw);
  }
}
void rotate_hwc2(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree) {
  // if (degree == 90){
  //     rotate_hwc2_90(src, dst, srcw, srch, srch, srcw);
  // }else if (degree == 180){
  //     rotate_hwc2_180(src, dst, srcw, srch, srcw, srch);
  // }else if (degree == 270){
  //     rotate_hwc2_270(src, dst, srcw, srch, srch, srcw);
  // }
}

void rotate_hwc3(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree) {
  if (degree == 90) {
    rotate_hwc3_90(src, dst, srcw, srch, srch, srcw);
  } else if (degree == 180) {
    rotate_hwc3_180(src, dst, srcw, srch, srcw, srch);
  } else if (degree == 270) {
    rotate_hwc3_270(src, dst, srcw, srch, srch, srcw);
  }
}

void rotate_hwc4(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree) {
  if (degree == 90) {
    rotate_hwc4_90(src, dst, srcw, srch, srch, srcw);
  } else if (degree == 180) {
    rotate_hwc4_180(src, dst, srcw, srch, srcw, srch);
  } else if (degree == 270) {
    rotate_hwc4_270(src, dst, srcw, srch, srch, srcw);
  }
}
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
