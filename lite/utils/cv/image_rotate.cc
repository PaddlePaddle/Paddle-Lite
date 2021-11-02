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
#include <string.h>
#include "lite/core/parallel_defines.h"
#include "lite/utils/cv/bgr_rotate.h"
namespace paddle {
namespace lite {
namespace utils {
namespace cv {
void ImageRotate::choose(const uint8_t* src,
                         uint8_t* dst,
                         ImageFormat srcFormat,
                         int srcw,
                         int srch,
                         float degree) {
  if (degree != 90 && degree != 180 && degree != 270) {
    printf("this degree: %f not support \n", degree);
  }
  if (srcFormat == GRAY) {
    rotate_hwc1(src, dst, srcw, srch, degree);
  } else if (srcFormat == BGR || srcFormat == RGB) {
    // rotate_hwc3(src, dst, srcw, srch, degree);
    bgr_rotate_hwc(src, dst, srcw, srch, static_cast<int>(degree));
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    rotate_hwc4(src, dst, srcw, srch, degree);
  } else {
    printf("this srcFormat: %d does not support! \n", srcFormat);
    return;
  }
}
// gray
void rotate_hwc1_90(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
void rotate_hwc1_180(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
void rotate_hwc1_270(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);

// bgr rgb
void rotate_hwc3_90(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
void rotate_hwc3_180(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
void rotate_hwc3_270(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
// rgba bgra
void rotate_hwc4_90(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
void rotate_hwc4_180(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);
void rotate_hwc4_270(
    const uint8_t* src, uint8_t* dst, int w_in, int h_in, int w_out, int h_out);

void rotate_hwc1(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree) {
  if (degree == 90) {
    rotate_hwc1_90(src, dst, srcw, srch, srch, srcw);
  } else if (degree == 180) {
    rotate_hwc1_180(src, dst, srcw, srch, srcw, srch);
  } else if (degree == 270) {
    rotate_hwc1_270(src, dst, srcw, srch, srch, srcw);
  } else {
    printf("this degree: %f does not support! \n", degree);
    return;
  }
}

void rotate_hwc3(
    const uint8_t* src, uint8_t* dst, int srcw, int srch, float degree) {
  if (degree == 90) {
    rotate_hwc3_90(src, dst, srcw, srch, srch, srcw);
  } else if (degree == 180) {
    rotate_hwc3_180(src, dst, srcw, srch, srcw, srch);
  } else if (degree == 270) {
    rotate_hwc3_270(src, dst, srcw, srch, srch, srcw);
  } else {
    printf("this degree: %f does not support! \n", degree);
    return;
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
  } else {
    printf("this degree: %f does not support! \n", degree);
    return;
  }
}
#ifdef __aarch64__

#define INPUT_C1                    \
  "ld1  {v0.8b}, [%[inptr0]]    \n" \
  "ld1  {v4.8b}, [%[inptr1]]    \n" \
  "ld1  {v8.8b}, [%[inptr2]]    \n" \
  "ld1  {v12.8b}, [%[inptr3]]    \n"

#define INPUT_C3                                   \
  "ld3  {v0.8b, v1.8b, v2.8b}, [%[inptr0]]    \n"  \
  "ld3  {v4.8b, v5.8b, v6.8b}, [%[inptr1]]    \n"  \
  "ld3  {v8.8b, v9.8b, v10.8b}, [%[inptr2]]    \n" \
  "ld3  {v12.8b, v13.8b, v14.8b}, [%[inptr3]]    \n"

#define ADD_INPUT                            \
  "add %[inptr0], %[inptr0], %[stride_h] \n" \
  "add %[inptr1], %[inptr1], %[stride_h] \n" \
  "add %[inptr2], %[inptr2], %[stride_h] \n" \
  "add %[inptr3], %[inptr3], %[stride_h] \n"

#define SUB_INPUT                                                    \
  "sub %[inptr0], %[inptr0], %[stride_h_w] \n" /*  4 - 4*w_in + 8 */ \
  "sub %[inptr1], %[inptr1], %[stride_h_w] \n" /*  5 - 4*w_in + 8 */ \
  "sub %[inptr2], %[inptr2], %[stride_h_w] \n" /*  6 - 4*w_in + 8 */ \
  "sub %[inptr3], %[inptr3], %[stride_h_w] \n" /*  7 - 4*w_in + 8 */

#define TRANS_C1_8                                                             \
  "trn1 v1.8b, v0.8b, v4.8b             \n"  /* b v4=00 10 02 12 04 14 06 16*/ \
  "trn1 v5.8b, v8.8b, v12.8b             \n" /* v4={20 30 22 32 24 34 26 36 */ \
  "trn2 v2.8b, v0.8b, v4.8b             \n"  /* v5={01 11 03 13 05 15 07 17 */ \
  "trn2 v6.8b, v8.8b, v12.8b             \n" /* v7={21 31 23 33 25 35 27 37 */

#define TRANS_C1_16                                                            \
  "trn1 v9.4h, v1.4h, v5.4h             \n"                                    \
  "trn1 v13.4h, v2.4h, v6.4h             \n" /* v22=01 11 21 31 05 15 25 35 */ \
  "trn2 v10.4h, v1.4h, v5.4h             \n" /* v21=02 12 22 32 06 16 26 36*/  \
  "trn2 v14.4h, v2.4h, v6.4h             \n" /* v23=03 13 23 33 07 17 27 37 */

#define TRANS_C1                                                               \
  "trn1 v0.4h, v1.4h, v5.4h             \n" /* b v4=40 50 60 70 04 14 24 34 */ \
  "trn1 v8.4h, v2.4h, v6.4h             \n" /* v4=41 11 21 31 05 15 25 35 */   \
  "trn2 v4.4h, v1.4h, v5.4h             \n" /* v5=42 12 22 32 06 16 26 36*/    \
  "trn2 v12.4h, v2.4h, v6.4h             \n" /* v7=43 13 23 33 07 17 27 37 */  \
  "trn1 v1.2s, v9.2s, v0.2s             \n"  /* b v1=00 10 20 30 40 50 60 */   \
  "trn1 v2.2s, v13.2s, v8.2s             \n" /* v2= 01 11 21 31 41 51 61 71 */ \
  "trn1 v5.2s, v10.2s, v4.2s             \n" /* b v5=02 12 22 32 42 52 62 */   \
  "trn1 v6.2s, v14.2s, v12.2s             \n" /* v6=03 13 23 33 43 53 63 73*/  \
                                                                               \
  "trn2 v3.2s, v9.2s, v0.2s             \n"    /* v3=04 14 24 34 44 54 64 74*/ \
  "trn2 v7.2s, v13.2s, v8.2s             \n"   /* v7=05 15 25 35 45 55 65 75*/ \
  "trn2 v11.2s, v10.2s, v4.2s             \n"  /* v11=06 16 26 36 46 56 66 */  \
  "trn2 v15.2s, v14.2s, v12.2s             \n" /* v15=07 17 27 37 47 57 67 */

#define REVERSE_C1                         \
  "rev64  v0.8b, v1.8b                \n"  \
  "rev64  v4.8b, v2.8b                \n"  \
  "rev64  v8.8b, v5.8b                \n"  \
  "rev64  v12.8b, v6.8b                \n" \
                                           \
  "rev64  v1.8b, v3.8b                \n"  \
  "rev64  v5.8b, v7.8b                \n"  \
  "rev64  v9.8b, v11.8b                \n" \
  "rev64  v13.8b, v15.8b                \n"

#define STORE_C1_R                                                             \
  "st1 {v0.8b}, [%[outptr0]]             \n"  /* b v1=00 10 20 30 40 50 60*/   \
  "st1 {v4.8b}, [%[outptr1]]             \n"  /* v2=01 11 21 31 41 51 61 71*/  \
  "st1 {v8.8b}, [%[outptr2]]             \n"  /* b v5=02 12 22 32 42 52 62 */  \
  "st1 {v12.8b}, [%[outptr3]]             \n" /* v6=03 13 23 33 43 53 63 73*/  \
                                                                               \
  "st1 {v1.8b}, [%[outptr4]]             \n"  /* v3=04 14 24 34 44 54 64 74}*/ \
  "st1 {v5.8b}, [%[outptr5]]             \n"  /* v7=05 15 25 35 45 55 65 75}*/ \
  "st1 {v9.8b}, [%[outptr6]]             \n"  /* v11=06 16 26 36 46 56 66 */   \
  "st1 {v13.8b}, [%[outptr7]]             \n" /* v15=07 17 27 37 47 57 67 */

#define STORE_C1                                                              \
  "st1 {v1.8b}, [%[outptr0]]             \n" /* b v1=00 10 20 30 40 50 60 */  \
  "st1 {v2.8b}, [%[outptr1]]             \n" /* v2=01 11 21 31 41 51 61 71*/  \
  "st1 {v5.8b}, [%[outptr2]]             \n" /* b v5=02 12 22 32 42 52 62 */  \
  "st1 {v6.8b}, [%[outptr3]]             \n" /* v6=03 13 23 33 43 53 63 73}*/ \
                                                                              \
  "st1 {v3.8b}, [%[outptr4]]             \n"  /* v3=04 14 24 34 44 54 64 74*/ \
  "st1 {v7.8b}, [%[outptr5]]             \n"  /* v7=05 15 25 35 45 55 65 75*/ \
  "st1 {v11.8b}, [%[outptr6]]             \n" /* v11=06 16 26 36 46 56 66  */ \
  "st1 {v15.8b}, [%[outptr7]]             \n" /* v15=07 17 27 37 47 57 67*/

#define TRANS_C3_8                                                             \
  "trn1 v3.8b, v0.8b, v4.8b             \n" /* b v4=00 10 02 12 04 14 06 16 */ \
  "trn1 v7.8b, v8.8b, v12.8b             \n"  /* v4=20 30 22 32 24 34 26 36 */ \
  "trn2 v11.8b, v0.8b, v4.8b             \n"  /* v5=01 11 03 13 05 15 07 17 */ \
  "trn2 v15.8b, v8.8b, v12.8b             \n" /* v7=21 31 23 33 25 35 27 37*/  \
                                                                               \
  "trn1 v16.8b, v1.8b, v5.8b             \n"                                   \
  "trn1 v18.8b, v9.8b, v13.8b             \n" /* v4=20 30 22 32 24 34 26 36 */ \
  "trn2 v17.8b, v1.8b, v5.8b             \n" /* v5={01 11 03 13 05 15 07 17 */ \
  "trn2 v19.8b, v9.8b, v13.8b             \n" /* v7=21 31 23 33 25 35 27 37 */ \
                                                                               \
  "trn1 v20.8b, v2.8b, v6.8b             \n"                                   \
  "trn1 v22.8b, v10.8b, v14.8b             \n"                                 \
  "trn2 v21.8b, v2.8b, v6.8b             \n" /* v5=01 11 03 13 05 15 07 17 */  \
  "trn2 v23.8b, v10.8b, v14.8b             \n"

#define TRANS_C3_16                                                            \
  "trn1 v24.4h, v3.4h, v7.4h             \n"                                   \
  "trn1 v26.4h, v11.4h, v15.4h             \n" /* v4=01 11 21 31 05 15 25 35*/ \
  "trn2 v25.4h, v3.4h, v7.4h             \n"   /* v5=02 12 22 32 06 16 26 36*/ \
  "trn2 v27.4h, v11.4h, v15.4h             \n"                                 \
                                                                               \
  "trn1 v28.4h, v16.4h, v18.4h             \n" /* g v4=00 10 20 30 04 14 24 */ \
  "trn1 v30.4h, v17.4h, v19.4h             \n"                                 \
  "trn2 v29.4h, v16.4h, v18.4h             \n" /* v5=02 12 22 32 06 16 26 */   \
  "trn2 v31.4h, v17.4h, v19.4h             \n"                                 \
                                                                               \
  "trn1 v16.4h, v20.4h, v22.4h             \n" /* r v4=00 10 20 30 04 14 24 */ \
  "trn1 v18.4h, v21.4h, v23.4h             \n"                                 \
  "trn2 v17.4h, v20.4h, v22.4h             \n" /* v5=02 12 22 32 06 16 26 */   \
  "trn2 v19.4h, v21.4h, v23.4h             \n"

#define TRANS_C3                                                               \
  "trn1 v3.8b, v0.8b, v4.8b             \n" /* b v4=40 50 42 52 04 14 06 16 */ \
  "trn1 v7.8b, v8.8b, v12.8b             \n"  /* v4=60 70 62 72 24 34 26 36 */ \
  "trn2 v11.8b, v0.8b, v4.8b             \n"  /* v5=41 51 03 13 05 15 07 17 */ \
  "trn2 v15.8b, v8.8b, v12.8b             \n" /* v7=61 71 23 33 25 35 27 37 */ \
                                                                               \
  "trn1 v20.8b, v2.8b, v6.8b             \n"                                   \
  "trn1 v22.8b, v10.8b, v14.8b             \n"                                 \
  "trn2 v21.8b, v2.8b, v6.8b             \n" /* v5=41 51 03 13 05 15 07 17 */  \
  "trn2 v23.8b, v10.8b, v14.8b             \n"                                 \
                                                                               \
  "trn1 v0.4h, v3.4h, v7.4h             \n" /* b v4=40 50 60 70 04 14 24 34 */ \
  "trn1 v4.4h, v11.4h, v15.4h             \n" /* v4=41 51 61 71 05 15 25 35 */ \
  "trn2 v8.4h, v3.4h, v7.4h             \n"   /* v5=42 52 62 72 06 16 26 36*/  \
  "trn2 v12.4h, v11.4h, v15.4h             \n"                                 \
                                                                               \
  "trn1 v2.4h, v20.4h, v22.4h             \n"  /* r v4=40 50 60 70 */          \
  "trn1 v6.4h, v21.4h, v23.4h             \n"  /* v4=41 51 61 71 */            \
  "trn2 v10.4h, v20.4h, v22.4h             \n" /* v5=42 52 62 72 */            \
  "trn2 v14.4h, v21.4h, v23.4h             \n" /* v7=43 53 63 73 */            \
                                                                               \
  "trn1 v20.2s, v24.2s, v0.2s             \n"                                  \
  "trn1 v21.2s, v26.2s, v4.2s             \n" /* v4=01 11 21 31 41 51 61 71 */ \
  "trn1 v22.2s, v25.2s, v8.2s             \n" /* v5=02 12 22 32 42 52 62 72 */ \
  "trn1 v23.2s, v27.2s, v12.2s             \n"                                 \
                                                                               \
  "trn2 v3.2s, v24.2s, v0.2s             \n"                                   \
  "trn2 v7.2s, v26.2s, v4.2s             \n"  /* v4=05 11 21 31 41 51 61 71 */ \
  "trn2 v11.2s, v25.2s, v8.2s             \n" /* v5=06 12 22 32 42 52 62 72 */ \
  "trn2 v15.2s, v27.2s, v12.2s             \n" /* v7=07 13 23 33 43 53 63  */  \
                                                                               \
  "trn1 v0.2s, v16.2s, v2.2s             \n"  /* r v4=00 10 20 30 40 50 60 */  \
  "trn1 v4.2s, v18.2s, v6.2s             \n"  /* v4=01 11 21 31 41 51 61 71 */ \
  "trn1 v8.2s, v17.2s, v10.2s             \n" /* v5=02 12 22 32 42 52 62 72 */ \
  "trn1 v12.2s, v19.2s, v14.2s             \n" /* v7=03 13 23 33 43 53 63 */   \
                                                                               \
  "trn2 v24.2s, v16.2s, v2.2s             \n" /* r v4=04 10 20 30 40 50 60 */  \
  "trn2 v25.2s, v18.2s, v6.2s             \n" /* v4=05 11 21 31 41 51 61 71 */ \
  "trn2 v26.2s, v17.2s, v10.2s             \n" /* v5=06 12 22 32 42 52 62 */   \
  "trn2 v27.2s, v19.2s, v14.2s             \n" /* v7=07 13 23 33 43 53 63 */   \
                                                                               \
  "trn1 v16.8b, v1.8b, v5.8b             \n"  /* g v4={00 10 02 12 04 14 06 */ \
  "trn1 v18.8b, v9.8b, v13.8b             \n" /* v4={20 30 22 32 24 34 26 */   \
  "trn2 v17.8b, v1.8b, v5.8b             \n" /* v5={01 11 03 13 05 15 07 17 */ \
  "trn2 v19.8b, v9.8b, v13.8b             \n" /* v7={21 31 23 33 25 35 27 */   \
                                                                               \
  "sub %[inptr0], %[inptr0], %[stride_h_w] \n" /*  4 - 4*w_in + 8 */           \
  "sub %[inptr1], %[inptr1], %[stride_h_w] \n" /*  5 - 4*w_in + 8 */           \
  "sub %[inptr2], %[inptr2], %[stride_h_w] \n" /*  6 - 4*w_in + 8 */           \
  "sub %[inptr3], %[inptr3], %[stride_h_w] \n" /*  7 - 4*w_in + 8 */           \
                                                                               \
  "trn1 v1.4h, v16.4h, v18.4h             \n" /* g v4={00 10 20 30 04 14 24*/  \
  "trn1 v5.4h, v17.4h, v19.4h             \n" /* v4={ 01 11 21 31 05 15 25 */  \
  "trn2 v9.4h, v16.4h, v18.4h             \n" /* v5={02 12 22 32 06 16 26 36*/ \
  "trn2 v13.4h, v17.4h, v19.4h             \n" /* v7={03 13 23 33 07 17 27 */  \
                                                                               \
  "trn1 v2.2s, v28.2s, v1.2s             \n"  /* g v4=00 10 20 30 40 50 60 */  \
  "trn1 v6.2s, v30.2s, v5.2s             \n"  /* v4=01 11 21 31 41 51 61 71 */ \
  "trn1 v10.2s, v29.2s, v9.2s             \n" /* v5=02 12 22 32 42 52 62 72 */ \
  "trn1 v14.2s, v31.2s, v13.2s             \n" /* v7=03 13 23 33 43 53 63 */   \
                                                                               \
  "trn2 v16.2s, v28.2s, v1.2s             \n" /* g v4=04 10 20 30 40 50 60 */  \
  "trn2 v17.2s, v30.2s, v5.2s             \n" /* v4=05 11 21 31 41 51 61 71 */ \
  "trn2 v18.2s, v29.2s, v9.2s             \n" /* v5=06 12 22 32 42 52 62 72 */ \
  "trn2 v19.2s, v31.2s, v13.2s             \n" /* v7=07 13 23 33 43 53 63 */

#define REVERSE_C3                                          \
  "rev64 v28.8b, v20.8b \n" /* b 00 10 20 30 40 50 60 70*/  \
  "rev64 v29.8b, v2.8b \n"  /* g 00 10 20 30 40 50 60 70*/  \
  "rev64 v30.8b, v0.8b \n"  /* r 00 10 20 30 40 50 60 70*/  \
                                                            \
  "rev64 v0.8b, v21.8b \n" /* b 01 11 21 31 41 51 61 71 */  \
  "rev64 v1.8b, v6.8b \n"  /* g 01 11 21 31 41 51 61 71 */  \
  "rev64 v2.8b, v4.8b \n"  /* r 01 11 21 31 41 51 61 71 */  \
                                                            \
  "rev64 v4.8b, v22.8b \n" /* b 02 12 22 32 42 52 62 72 */  \
  "rev64 v5.8b, v10.8b \n" /* g 02 12 22 32 42 52 62 72*/   \
  "rev64 v6.8b, v8.8b \n"  /* r 02 12 22 32 42 52 62 72 */  \
                                                            \
  "rev64 v8.8b, v23.8b \n"  /* b 03 13 23 33 43 53 63 73 */ \
  "rev64 v9.8b, v14.8b \n"  /* g 03 13 23 33 43 53 63 73 */ \
  "rev64 v10.8b, v12.8b \n" /* r 03 13 23 33 43 53 63 73 */ \
                                                            \
  "rev64 v12.8b, v3.8b \n"  /* b 04 14 20 30 40 50 60 70 */ \
  "rev64 v13.8b, v16.8b \n" /* g 04 14 20 30 40 50 60 70 */ \
  "rev64 v14.8b, v24.8b \n" /* r 04 14 20 30 40 50 60 70 */ \
                                                            \
  "rev64 v20.8b, v7.8b \n"  /* b 05 15 20 30 40 50 60 70 */ \
  "rev64 v21.8b, v17.8b \n" /* g 05 15 20 30 40 50 60 70 */ \
  "rev64 v22.8b, v25.8b \n" /* r 05 15 20 30 40 50 60 70 */ \
                                                            \
  "rev64 v23.8b, v11.8b \n" /* b 06 15 20 30 40 50 60 70 */ \
  "rev64 v24.8b, v18.8b \n" /* g 06 15 20 30 40 50 60 70 */ \
  "rev64 v25.8b, v26.8b \n" /* r 06 15 20 30 40 50 60 70 */ \
                                                            \
  "rev64 v16.8b, v15.8b \n" /* b 07 15 20 30 40 50 60 70 */ \
  "rev64 v17.8b, v19.8b \n" /* g 07 15 20 30 40 50 60 70 */ \
  "rev64 v18.8b, v27.8b \n" /* r 07 15 20 30 40 50 60 70 */

#define MOV_C3                                            \
  "mov v28.8b, v20.8b \n" /* b 00 10 20 30 40 50 60 70*/  \
  "mov v29.8b, v2.8b \n"  /* g 00 10 20 30 40 50 60 70*/  \
  "mov v30.8b, v0.8b \n"  /* r 00 10 20 30 40 50 60 70*/  \
                                                          \
  "mov v0.8b, v21.8b \n" /* b 01 11 21 31 41 51 61 71 */  \
  "mov v1.8b, v6.8b \n"  /* g 01 11 21 31 41 51 61 71 */  \
  "mov v2.8b, v4.8b \n"  /* r 01 11 21 31 41 51 61 71 */  \
                                                          \
  "mov v4.8b, v22.8b \n" /* b 02 12 22 32 42 52 62 72 */  \
  "mov v5.8b, v10.8b \n" /* g 02 12 22 32 42 52 62 72*/   \
  "mov v6.8b, v8.8b \n"  /* r 02 12 22 32 42 52 62 72 */  \
                                                          \
  "mov v8.8b, v23.8b \n"  /* b 03 13 23 33 43 53 63 73 */ \
  "mov v9.8b, v14.8b \n"  /* g 03 13 23 33 43 53 63 73 */ \
  "mov v10.8b, v12.8b \n" /* r 03 13 23 33 43 53 63 73 */ \
                                                          \
  "mov v12.8b, v3.8b \n"  /* b 04 14 20 30 40 50 60 70 */ \
  "mov v13.8b, v16.8b \n" /* g 04 14 20 30 40 50 60 70 */ \
  "mov v14.8b, v24.8b \n" /* r 04 14 20 30 40 50 60 70 */ \
                                                          \
  "mov v20.8b, v7.8b \n"  /* b 05 15 20 30 40 50 60 70 */ \
  "mov v21.8b, v17.8b \n" /* g 05 15 20 30 40 50 60 70 */ \
  "mov v22.8b, v25.8b \n" /* r 05 15 20 30 40 50 60 70 */ \
                                                          \
  "mov v23.8b, v11.8b \n" /* b 06 15 20 30 40 50 60 70 */ \
  "mov v24.8b, v18.8b \n" /* g 06 15 20 30 40 50 60 70 */ \
  "mov v25.8b, v26.8b \n" /* r 06 15 20 30 40 50 60 70 */ \
                                                          \
  "mov v16.8b, v15.8b \n" /* b 07 15 20 30 40 50 60 70 */ \
  "mov v17.8b, v19.8b \n" /* g 07 15 20 30 40 50 60 70 */ \
  "mov v18.8b, v27.8b \n" /* r 07 15 20 30 40 50 60 70 */

#define STORE_C3                                              \
  "st3 {v28.8b, v29.8b, v30.8b}, [%[outptr0]]             \n" \
  "st3 {v0.8b, v1.8b, v2.8b}, [%[outptr1]]             \n"    \
  "st3 {v4.8b, v5.8b, v6.8b}, [%[outptr2]]             \n"    \
  "st3 {v8.8b, v9.8b, v10.8b}, [%[outptr3]]             \n"   \
                                                              \
  "st3 {v12.8b, v13.8b, v14.8b}, [%[outptr4]]             \n" \
  "st3 {v20.8b, v21.8b, v22.8b}, [%[outptr5]]             \n" \
  "st3 {v23.8b, v24.8b, v25.8b}, [%[outptr6]]             \n" \
  "st3 {v16.8b, v17.8b, v18.8b}, [%[outptr7]]             \n"

#else

#define INPUT_C1                                                             \
  "vld1.8  {d0}, [%[inptr0]]   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n" \
  "vld1.8  {d4}, [%[inptr1]]   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n" \
  "vld1.8  {d8}, [%[inptr2]]   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n" \
  "vld1.8  {d12}, [%[inptr3]]   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n"

#define INPUT_C3                                                               \
  "vld3.8  {d0, d1, d2}, [%[inptr0]]   @ zip load r0, d0 =00 01 02 03 04 05 "  \
  "06 07\n"                                                                    \
  "vld3.8  {d4, d5, d6}, [%[inptr1]]   @ zip load r1, d2 =10 11 12 13 14 15 "  \
  "16 17\n"                                                                    \
  "vld3.8  {d8, d9, d10}, [%[inptr2]]   @ zip load r1, d4 =20 21 22 23 24 25 " \
  "26 27\n"                                                                    \
  "vld3.8  {d12, d13, d14}, [%[inptr3]]   @ zip load r1, d6 = 30 31 32 33 34 " \
  "35 36 37\n"

#define ADD_INPUT                            \
  "add %[inptr0], %[inptr0], %[stride_h] \n" \
  "add %[inptr1], %[inptr1], %[stride_h] \n" \
  "add %[inptr2], %[inptr2], %[stride_h] \n" \
  "add %[inptr3], %[inptr3], %[stride_h] \n"

#define SUB_INPUT                              \
  "sub %[inptr0], %[inptr0], %[stride_h_w] \n" \
  "sub %[inptr1], %[inptr1], %[stride_h_w] \n" \
  "sub %[inptr2], %[inptr2], %[stride_h_w] \n" \
  "sub %[inptr3], %[inptr3], %[stride_h_w] \n"

#define TRANS_C1                                                               \
  "vtrn.8  d0, d4    @ trans data: \n" /* d0=00 10 02 12 04 14 06 16 */        \
  "vtrn.8  d8, d12   @ trans data: \n" /* d8=20 30 12 32 24 34 26 36 */        \
                                                                               \
  "vld1.8  {d1}, [%[inptr0]]   @ zip load r0, d0 =00 01 02 03 04 05 06 07\n"   \
  "vld1.8  {d5}, [%[inptr1]]   @ zip load r1, d2 =10 11 12 13 14 15 16 17\n"   \
  "vld1.8  {d9}, [%[inptr2]]   @ zip load r1, d4 =20 21 22 23 24 25 26 27\n"   \
  "vld1.8  {d13}, [%[inptr3]]   @ zip load r1, d6 = 30 31 32 33 34 35 36 37\n" \
                                                                               \
  "vtrn.16 d0, d8    @ trans data: \n" /* d0=00 10 20 30 04 14 24 34  */       \
  "vtrn.16  d4, d12  @ trans data:\n"  /* d4=01 11 21 31 05 15 25 35 */        \
                                                                               \
  "vtrn.8  d1, d5    @ trans data: \n" /* d0=40 50 42 52 04 14 06 16 */        \
  "vtrn.8  d9, d13   @ trans data: \n" /* d8=60 70 62 72 24 34 26 36  */       \
                                                                               \
  "sub %[inptr0], %[inptr0], %[stride_h_w] \n"                                 \
  "sub %[inptr1], %[inptr1], %[stride_h_w] \n"                                 \
  "sub %[inptr2], %[inptr2], %[stride_h_w] \n"                                 \
  "sub %[inptr3], %[inptr3], %[stride_h_w] \n"                                 \
                                                                               \
  "vtrn.16 d1, d9    @ trans data: \n" /* d0=40 50 60 70 04 14 24 34 */        \
  "vtrn.16 d5, d13  @ trans data:\n"   /* d4=41 51 61 71 05 15 25 35 */        \
                                                                               \
  "vtrn.32 d0, d1                  @ trans data: \n"                           \
  "vtrn.32  d8, d9                  @ trans data: \n"                          \
  "vtrn.32  d4, d5                  @ trans data: \n"                          \
  "vtrn.32  d12, d13                  @ trans data: \n"

#define REVERSE_C1                              \
  "vrev64.8 d2, d0 @reverse 7 6 5 4 3 2 1 \n"   \
  "vrev64.8 d3, d1 @reverse 7 6 5 4 3 2 1 \n"   \
  "vrev64.8 d10, d8 @reverse 7 6 5 4 3 2 1 \n"  \
  "vrev64.8 d11, d9 @reverse 7 6 5 4 3 2 1 \n"  \
  "vrev64.8 d6, d4 @reverse 7 6 5 4 3 2 1 \n"   \
  "vrev64.8 d7, d5 @reverse 7 6 5 4 3 2 1 \n"   \
  "vrev64.8 d14, d12 @reverse 7 6 5 4 3 2 1 \n" \
  "vrev64.8 d15, d13 @reverse 7 6 5 4 3 2 1 \n"

#define ADD_OUTPUT                               \
  "add %[outptr0], %[outptr0], %[stride_out] \n" \
  "add %[outptr2], %[outptr2], %[stride_out] \n" \
  "add %[outptr1], %[outptr1], %[stride_out] \n" \
  "add %[outptr3], %[outptr3], %[stride_out] \n"

#define SUB_OUTPUT                               \
  "sub %[outptr0], %[outptr0], %[stride_out] \n" \
  "sub %[outptr2], %[outptr2], %[stride_out] \n" \
  "sub %[outptr1], %[outptr1], %[stride_out] \n" \
  "sub %[outptr3], %[outptr3], %[stride_out] \n"

#define STORE_C1_4                                                     \
  "vst1.8  {d0},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d8},    [%[outptr2]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d4},    [%[outptr1]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d12},    [%[outptr3]]   @ write d0(q0,low),r00,r10 20 30\n"

#define STORE_C1_8                                                     \
  "vst1.8  {d1},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d9},    [%[outptr2]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d5},    [%[outptr1]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d13},    [%[outptr3]]   @ write d0(q0,low),r00,r10 20 30\n"

#define STORE_C1_R_4                                                    \
  "vst1.8  {d2},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n"  \
  "vst1.8  {d10},    [%[outptr2]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d6},    [%[outptr1]]   @ write d0(q0,low),r00,r10 20 30\n"  \
  "vst1.8  {d14},    [%[outptr3]]   @ write d0(q0,low),r00,r10 20 30\n"

#define STORE_C1_R_8                                                    \
  "vst1.8  {d3},    [%[outptr0]]   @ write d0(q0,low),r00,r10 20 30\n"  \
  "vst1.8  {d11},    [%[outptr2]]   @ write d0(q0,low),r00,r10 20 30\n" \
  "vst1.8  {d7},    [%[outptr1]]   @ write d0(q0,low),r00,r10 20 30\n"  \
  "vst1.8  {d15},    [%[outptr3]]   @ write d0(q0,low),r00,r10 20 30\n"

#define TRANS_C3                                                              \
  "vtrn.8  d0, d4    @ trans data: \n"                                        \
  "vtrn.8  d8, d12   @ trans data: \n"                                        \
  "vtrn.8  d1, d5    @ trans data: \n"                                        \
  "vtrn.8  d9, d13   @ trans data: \n"                                        \
  "vtrn.8  d2, d6    @ trans data: \n"                                        \
  "vtrn.8  d10, d14   @ trans data: \n"                                       \
                                                                              \
  "vld3.8  {d16, d17, d18}, [%[inptr0]]   @ zip load r0, d0 =40 01 02 03 04 " \
  "05 06 07\n"                                                                \
  "vld3.8  {d20, d21, d22}, [%[inptr1]]   @ zip load r1, d2 =50 11 12 13 14 " \
  "15 16 17\n"                                                                \
  "vld3.8  {d24, d25, d26}, [%[inptr2]]   @ zip load r1, d4 =60 21 22 23 24 " \
  "25 26 27\n"                                                                \
  "vld3.8  {d28, d29, d30}, [%[inptr3]]   @ zip load r1, d6 =70 31 32 33 34 " \
  "35 36 37\n"                                                                \
                                                                              \
  "vtrn.16 d0, d8    @ trans data: \n"                                        \
  "vtrn.16  d4, d12  @ trans data:\n"                                         \
  "vtrn.16 d1, d9    @ trans data: \n"                                        \
  "vtrn.16  d5, d13  @ trans data:\n"                                         \
  "vtrn.16 d2, d10    @ trans data: \n"                                       \
  "vtrn.16  d6, d14  @ trans data:\n"                                         \
                                                                              \
  "vtrn.8  d16, d20    @ trans data: \n"                                      \
  "vtrn.8  d24, d28   @ trans data: \n"                                       \
  "vtrn.8  d17, d21    @ trans data: \n"                                      \
  "vtrn.8  d25, d29   @ trans data: \n"                                       \
  "vtrn.8  d18, d22    @ trans data: \n"                                      \
  "vtrn.8  d26, d30   @ trans data: \n"                                       \
                                                                              \
  "sub %[inptr0], %[inptr0], %[stride_h_w] \n"                                \
  "sub %[inptr1], %[inptr1], %[stride_h_w] \n"                                \
  "sub %[inptr2], %[inptr2], %[stride_h_w] \n"                                \
  "sub %[inptr3], %[inptr3], %[stride_h_w] \n"                                \
                                                                              \
  "vtrn.16 d16, d24    @ trans data: \n"                                      \
  "vtrn.16 d20, d28    @ trans data: \n"                                      \
  "vtrn.16 d17, d25    @ trans data: \n"                                      \
  "vtrn.16 d21, d29    @ trans data: \n"                                      \
  "vtrn.16 d18, d26    @ trans data: \n"                                      \
  "vtrn.16 d22, d30    @ trans data: \n"                                      \
                                                                              \
  "vtrn.32 d0, d16     @ trans data: \n"                                      \
  "vtrn.32 d8, d24     @ trans data: \n"                                      \
  "vtrn.32 d4, d20     @ trans data: \n"                                      \
  "vtrn.32 d12, d28     @ trans data: \n"                                     \
                                                                              \
  "vtrn.32 d1, d17     @ trans data: \n"                                      \
  "vtrn.32 d9, d25     @ trans data: \n"                                      \
  "vtrn.32 d5, d21     @ trans data: \n"                                      \
  "vtrn.32 d13, d29     @ trans data: \n"                                     \
                                                                              \
  "vtrn.32 d2, d18     @ trans data: \n"                                      \
  "vtrn.32 d10, d26     @ trans data: \n"                                     \
  "vtrn.32 d6, d22     @ trans data: \n"                                      \
  "vtrn.32 d14, d30    @ trans data: \n"

#define STORE_C3_4                        \
  "vst3.8 {d0, d1, d2}, [%[outptr0]] \n"  \
  "vst3.8 {d4, d5, d6}, [%[outptr1]] \n"  \
  "vst3.8 {d8, d9, d10}, [%[outptr2]] \n" \
  "vst3.8 {d12, d13, d14}, [%[outptr3]] \n"

#define STORE_C3_8                          \
  "vst3.8 {d16, d17, d18}, [%[outptr0]] \n" \
  "vst3.8 {d20, d21, d22}, [%[outptr1]] \n" \
  "vst3.8 {d24, d25, d26}, [%[outptr2]] \n" \
  "vst3.8 {d28, d29, d30}, [%[outptr3]] \n"

#define REVERSE_C3                 \
  "vrev64.8 d3, d0 \n"  /* b 00*/  \
  "vrev64.8 d7, d4 \n"  /* b 01*/  \
  "vrev64.8 d15, d5 \n" /* g 01*/  \
  "vrev64.8 d11, d8 \n" /* b 02*/  \
  "vrev64.8 d4, d1 \n"  /* g 00*/  \
  "vrev64.8 d5, d2 \n"  /* r 00*/  \
                                   \
  "vrev64.8 d0, d12 \n" /* b 03*/  \
  "vrev64.8 d1, d13 \n" /* g 03*/  \
  "vrev64.8 d2, d14 \n" /* r 03*/  \
                                   \
  "vrev64.8 d12, d9 \n"  /* g 02*/ \
  "vrev64.8 d13, d10 \n" /* r 02*/ \
                                   \
  "vmov d8, d15 \n"    /* g 01*/   \
  "vrev64.8 d9, d6 \n" /* r 01*/   \
                                   \
  "vrev64.8 d14, d16 \n" /* b 04*/ \
  "vrev64.8 d15, d17 \n" /* g 04*/ \
  "vrev64.8 d16, d18 \n" /* r 04*/ \
                                   \
  "vrev64.8 d17, d20 \n" /* b 05*/ \
  "vrev64.8 d18, d21 \n" /* g 05*/ \
  "vrev64.8 d19, d22 \n" /* r 05*/ \
                                   \
  "vrev64.8 d20, d24 \n" /* b 06*/ \
  "vrev64.8 d21, d25 \n" /* g 06*/ \
  "vrev64.8 d22, d26 \n" /* r 06*/ \
                                   \
  "vrev64.8 d24, d28 \n" /* b 07*/ \
  "vrev64.8 d25, d29 \n" /* g 07*/ \
  "vrev64.8 d26, d30 \n" /* r 07*/

#define STORE_C3_R_4                        \
  "vst3.8 {d3, d4, d5}, [%[outptr0]] \n"    \
  "vst3.8 {d0, d1, d2}, [%[outptr3]] \n"    \
  "vst3.8 {d11, d12, d13}, [%[outptr2]] \n" \
  "vst3.8 {d7, d8, d9}, [%[outptr1]] \n"

#define STORE_C3_R_8                        \
  "vst3.8 {d14, d15, d16}, [%[outptr0]] \n" \
  "vst3.8 {d17, d18, d19}, [%[outptr1]] \n" \
  "vst3.8 {d20, d21, d22}, [%[outptr2]] \n" \
  "vst3.8 {d24, d25, d26}, [%[outptr3]] \n"

#endif
/*
1 2 3
4 5 6
7 8 9
rotate:
7 4 1
8 5 2
9 6 3
*/
// transpose
void rotate_hwc1_90(const uint8_t* src,
                    uint8_t* dst,
                    int w_in,
                    int h_in,
                    int w_out,
                    int h_out) {
  uint8_t* zerobuff = new uint8_t[8];
  // block 4*8. -- 8*4
  int i = 0;
  int stride_h = 4 * w_in;
  int stride_h_w = 4 * w_in - 8;
  int stride_out = 4 * w_out;
  int ww = w_out - 8;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in - 7, 0, 8) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;
    int j = 0;
    for (; j < w_in - 7; j += 8) {
      uint8_t* outptr0 = dst + j * w_out + (ww - i);
      uint8_t* outptr1 = outptr0 + w_out;
      uint8_t* outptr2 = outptr1 + w_out;
      uint8_t* outptr3 = outptr2 + w_out;
      uint8_t* outptr4 = outptr3 + w_out;
      uint8_t* outptr5 = outptr4 + w_out;
      uint8_t* outptr6 = outptr5 + w_out;
      uint8_t* outptr7 = outptr6 + w_out;
#ifdef __aarch64__
      asm volatile(INPUT_C1 ADD_INPUT TRANS_C1_8 INPUT_C1 TRANS_C1_16 TRANS_C1_8
                       SUB_INPUT TRANS_C1 REVERSE_C1 STORE_C1_R
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
#else
      asm volatile(INPUT_C1 ADD_INPUT TRANS_C1 REVERSE_C1 STORE_C1_R_4
                       ADD_OUTPUT STORE_C1_R_8
                   : [inptr0] "+r"(inptr0),
                     [inptr1] "+r"(inptr1),
                     [inptr2] "+r"(inptr2),
                     [inptr3] "+r"(inptr3),
                     [outptr0] "+r"(outptr0),
                     [outptr1] "+r"(outptr1),
                     [outptr2] "+r"(outptr2),
                     [outptr3] "+r"(outptr3)
                   : [stride_h] "r"(stride_h),
                     [stride_h_w] "r"(stride_h_w),
                     [stride_out] "r"(stride_out)
                   : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif
    }
    const uint8_t* inptr4 = inptr3 + w_in;
    const uint8_t* inptr5 = inptr4 + w_in;
    const uint8_t* inptr6 = inptr5 + w_in;
    const uint8_t* inptr7 = inptr6 + w_in;
    for (; j < w_in; j++) {
      uint8_t* outptr = dst + j * w_out + ww - i;
      *outptr++ = *inptr7++;
      *outptr++ = *inptr6++;
      *outptr++ = *inptr5++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr0++;
    }
  }
  LITE_PARALLEL_COMMON_END();
  ww = w_out - 1;
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * w_in;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + j * w_out + ww - i;
      *outptr0 = *inptr0++;
    }
  }
  delete[] zerobuff;
}
/*
1 2 3 4
4 5 6 7
7 8 9 10
rotate:
10 9 8 7
7 6 5 4
4 3 2 1
*/
void rotate_hwc1_180(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  uint8_t* zerobuff = new uint8_t[w_in];
  memset(zerobuff, 0, w_in * sizeof(uint8_t));
  int stride_w = 8;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in, 0, 4) {
    const uint8_t* inptr0 = src + i * w_in;
    const uint8_t* inptr1 = inptr0 + w_in;
    const uint8_t* inptr2 = inptr1 + w_in;
    const uint8_t* inptr3 = inptr2 + w_in;

    uint8_t* outptr0 = dst + (h_in - i) * w_out - stride_w;  // last
    uint8_t* outptr1 = outptr0 - w_out;
    uint8_t* outptr2 = outptr1 - w_out;
    uint8_t* outptr3 = outptr2 - w_out;

    if (i + 3 >= h_in) {
      uint8_t* ptr = zerobuff + w_in - stride_w;
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = ptr;
        case 2:
          inptr1 = zerobuff;
          outptr1 = ptr;
        case 1:
          inptr2 = zerobuff;
          outptr2 = ptr;
        case 0:
          inptr3 = zerobuff;
          outptr3 = ptr;
        default:
          break;
      }
    }

    int j = 0;
    for (; j < w_in - 7; j += 8) {
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
          case 2:
            *outptr0-- = *inptr0++;
          case 3:
          default:
            break;
        }
      } else {
        *outptr3-- = *inptr3++;
        *outptr2-- = *inptr2++;
        *outptr1-- = *inptr1++;
        *outptr0-- = *inptr0++;
      }
    }
  }
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuff;
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
void rotate_hwc1_270(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  int stride_h = 4 * w_in;
  int stride_h_w = 4 * w_in - 8;
  int hout = h_out - 1;
  int stride_out = 4 * w_out;

  int i = 0;
  // block 8*8. -- 8*8
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in - 7, 0, 8) {
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

#ifdef __aarch64__
      asm volatile(INPUT_C1 ADD_INPUT TRANS_C1_8 INPUT_C1 TRANS_C1_16 TRANS_C1_8
                       SUB_INPUT TRANS_C1 STORE_C1
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
#else
      asm volatile(INPUT_C1 ADD_INPUT TRANS_C1 STORE_C1_4 ADD_OUTPUT STORE_C1_8
                   : [inptr0] "+r"(inptr0),
                     [inptr1] "+r"(inptr1),
                     [inptr2] "+r"(inptr2),
                     [inptr3] "+r"(inptr3),
                     [outptr0] "+r"(outptr0),
                     [outptr1] "+r"(outptr1),
                     [outptr2] "+r"(outptr2),
                     [outptr3] "+r"(outptr3)
                   : [stride_h] "r"(stride_h),
                     [stride_h_w] "r"(stride_h_w),
                     [stride_out] "r"(stride_out)
                   : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif
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
  LITE_PARALLEL_COMMON_END();
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
  int stride_h = 4 * win;
  int stride_h_w = 4 * win - 24;
  int stride_out = 4 * wout;
  int ww = w_out - 8;
  uint8_t* zerobuff = new uint8_t[8];
  // block 4*8. -- 8*4
  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in - 7, 0, 8) {
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
      asm volatile(INPUT_C3 ADD_INPUT TRANS_C3_8 INPUT_C3 TRANS_C3_16 TRANS_C3
                       REVERSE_C3 STORE_C3
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
                     "v30",
                     "v31");
#else
      asm volatile(INPUT_C3 ADD_INPUT TRANS_C3 REVERSE_C3 STORE_C3_R_4
                       ADD_OUTPUT STORE_C3_R_8
                   : [inptr0] "+r"(inptr0),
                     [inptr1] "+r"(inptr1),
                     [inptr2] "+r"(inptr2),
                     [inptr3] "+r"(inptr3),
                     [outptr0] "+r"(outptr0),
                     [outptr1] "+r"(outptr1),
                     [outptr2] "+r"(outptr2),
                     [outptr3] "+r"(outptr3)
                   : [stride_h] "r"(stride_h),
                     [stride_h_w] "r"(stride_h_w),
                     [stride_out] "r"(stride_out)
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
                     "q12",
                     "q13",
                     "q14");
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
  LITE_PARALLEL_COMMON_END();
  // remain
  ww = w_out - 1;
  for (; i < h_in; i++) {
    const uint8_t* inptr0 = src + i * win;
    for (int j = 0; j < w_in; j++) {
      uint8_t* outptr0 = dst + j * wout + (ww - i) * 3;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
      *outptr0++ = *inptr0++;
    }
  }
  delete[] zerobuff;
}

void rotate_hwc3_180(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  int win = w_in * 3;
  uint8_t* zerobuff = new uint8_t[win];
  memset(zerobuff, 0, win * sizeof(uint8_t));
  int stride_w = 24;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in, 0, 4) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    uint8_t* outptr0 = dst + (h_in - i) * win - stride_w;  // last col
    uint8_t* outptr1 = outptr0 - win;
    uint8_t* outptr2 = outptr1 - win;
    uint8_t* outptr3 = outptr2 - win;

    if (i + 3 >= h_in) {
      uint8_t* ptr = zerobuff + win - stride_w;
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = ptr;
        case 2:
          inptr1 = zerobuff;
          outptr1 = ptr;
        case 1:
          inptr2 = zerobuff;
          outptr2 = ptr;
        case 0:
          inptr3 = zerobuff;
          outptr3 = ptr;
        default:
          break;
      }
    }

    int j = 0;
    for (; j < w_in - 7; j += 8) {
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
          "\n"
          "vld3.8  {d3, d4, d5}, [%[inptr1]]!   @ zip load r1, d2 =10 11 12 13 "
          "\n"
          "vld3.8  {d6, d7, d8}, [%[inptr2]]!   @ zip load r1, d4 =20 21 22 23 "
          "\n"
          "vld3.8  {d9, d10, d11}, [%[inptr3]]!   @ zip load r1, d6 = 30 31 32 "
          "\n"

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

          "vst3.8  {d12, d13, d14},    [%[outptr0]]   @ write \n"
          "vst3.8  {d15, d16, d17},    [%[outptr1]]   @ write \n"
          "vst3.8  {d18, d19, d20},    [%[outptr2]]   @ write \n"
          "vst3.8  {d21, d22, d23},    [%[outptr3]]   @ write \n"

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
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuff;
  delete[] zerobuff;
}

void rotate_hwc3_270(const uint8_t* src,
                     uint8_t* dst,
                     int w_in,
                     int h_in,
                     int w_out,
                     int h_out) {
  int win = w_in * 3;
  int wout = w_out * 3;
  int stride_h = 4 * win;
  int stride_h_w = 4 * win - 24;
  int stride_out = 4 * wout;
  int hout = h_out - 1;
  // block 8*8. -- 8*8
  int i = 0;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in - 7, 0, 8) {
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
      asm volatile(INPUT_C3 ADD_INPUT TRANS_C3_8 INPUT_C3 TRANS_C3_16 TRANS_C3
                       MOV_C3 STORE_C3
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
                     "v30",
                     "v31");
#else
      asm volatile(INPUT_C3 ADD_INPUT TRANS_C3 STORE_C3_4 SUB_OUTPUT STORE_C3_8
                   : [inptr0] "+r"(inptr0),
                     [inptr1] "+r"(inptr1),
                     [inptr2] "+r"(inptr2),
                     [inptr3] "+r"(inptr3),
                     [outptr0] "+r"(outptr0),
                     [outptr1] "+r"(outptr1),
                     [outptr2] "+r"(outptr2),
                     [outptr3] "+r"(outptr3)
                   : [stride_h] "r"(stride_h),
                     [stride_out] "r"(stride_out),
                     [stride_h_w] "r"(stride_h_w)
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
                     "q12",
                     "q13",
                     "q14");
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
  LITE_PARALLEL_COMMON_END();
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
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in - 7, 0, 8) {
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
  LITE_PARALLEL_COMMON_END();
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
  uint8_t* zerobuff = new uint8_t[win];
  memset(zerobuff, 0, win * sizeof(uint8_t));
  int stride_w = 32;
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in, 0, 4) {
    const uint8_t* inptr0 = src + i * win;
    const uint8_t* inptr1 = inptr0 + win;
    const uint8_t* inptr2 = inptr1 + win;
    const uint8_t* inptr3 = inptr2 + win;

    uint8_t* outptr0 = dst + (h_in - i) * win - stride_w;  // last col
    uint8_t* outptr1 = outptr0 - win;
    uint8_t* outptr2 = outptr1 - win;
    uint8_t* outptr3 = outptr2 - win;

    if (i + 3 >= h_in) {
      uint8_t* ptr = zerobuff + win - stride_w;
      switch ((i + 3) - h_in) {
        case 3:
          inptr0 = zerobuff;
          outptr0 = ptr;
        case 2:
          inptr1 = zerobuff;
          outptr1 = ptr;
        case 1:
          inptr2 = zerobuff;
          outptr2 = ptr;
        case 0:
          inptr3 = zerobuff;
          outptr3 = ptr;
        default:
          break;
      }
    }

    int j = 0;
    for (; j < w_in - 7; j += 8) {
#ifdef __aarch64__
      asm volatile(
          "ld4  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[inptr0]], #32    \n"  // v0={00,01,02,
          // 03, 04, 05,
          // 06, 07}"
          "ld4  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[inptr1]], #32     \n"  // v0={10,11,12,
          // 13, 14, 15,
          // 16, 17}"
          "ld4  {v8.8b, v9.8b, v10.8b, v11.8b}, [%[inptr2]], #32    \n"  // v0={20,21,22,
          // 23, 24, 25,
          // 26, 27}"
          "ld4  {v12.8b, v13.8b, v14.8b, v15.8b}, [%[inptr3]], #32    \n"  // v0={30,31,32,
          // 33, 34, 35,
          // 36, 37}"
          "rev64  v16.8b, v0.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 b
          "rev64  v17.8b, v1.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 g
          "rev64  v18.8b, v2.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 r
          "rev64  v19.8b, v3.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00 a

          "rev64  v20.8b, v4.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v21.8b, v5.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v22.8b, v6.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00
          "rev64  v23.8b, v7.8b                \n"  //@ reverse 07 06 05 04 03
                                                    // 02 01 00

          "rev64  v24.8b, v8.8b                \n"   //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v25.8b, v9.8b                \n"   //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v26.8b, v10.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v27.8b, v11.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00

          "rev64  v28.8b, v12.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v29.8b, v13.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v30.8b, v14.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00
          "rev64  v31.8b, v15.8b                \n"  //@ reverse 07 06 05 04 03
                                                     // 02 01 00

          "st4 {v16.8b, v17.8b, v18.8b, v19.8b}, [%[outptr0]]          \n"  // 00 10
          "st4 {v20.8b, v21.8b, v22.8b, v23.8b}, [%[outptr1]]              \n"  // 02 12
          "st4 {v24.8b, v25.8b, v26.8b, v27.8b}, [%[outptr2]]             \n"  // 01 11
          "st4 {v28.8b, v29.8b, v30.8b, v31.8b}, [%[outptr3]]              \n"  // 03 13

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
            "v23",
            "v24",
            "v25",
            "v26",
            "v27",
            "v28",
            "v29",
            "v30",
            "v31");
#else
      asm volatile(
          "vld4.8  {d0, d1, d2, d3}, [%[inptr0]]!   @ zip load r0, d0 =00 01 "
          "02 03 "
          "04 05 06 07\n"
          "vld4.8  {d4, d5, d6, d7}, [%[inptr1]]!   @ zip load r1, d2 =10 11 "
          "12 13 "
          "14 15 16 17\n"
          "vld4.8  {d8, d9, d10, d11}, [%[inptr2]]!   @ zip load r1, d4 =20 21 "
          "22 23 "
          "24 25 26 27\n"
          "vld4.8  {d12, d13, d14, d15}, [%[inptr3]]!   @ zip load r1, d6 = 30 "
          "31 32 "
          "33 34 35 36 37\n"

          "vrev64.8  d16, d0               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d17, d1               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d18, d2               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d19, d3               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d20, d4               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d21, d5               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d22, d6               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d23, d7               @ reverse 07 06 05 04 03 02 01 00 \n"

          "vrev64.8  d24, d8               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d25, d9               @ reverse 07 06 05 04 03 02 01 00 \n"
          "vrev64.8  d26, d10               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"
          "vrev64.8  d27, d11               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"

          "vrev64.8  d28, d12               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"
          "vrev64.8  d29, d13               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"
          "vrev64.8  d30, d14               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"
          "vrev64.8  d31, d15               @ reverse 07 06 05 04 03 02 01 00 "
          "\n"

          "vst4.8  {d16, d17, d18, d19},    [%[outptr0]]   @ write \n"
          "vst4.8  {d20, d21, d22, d23},    [%[outptr1]]   @ write \n"
          "vst4.8  {d24, d25, d26, d27},    [%[outptr2]]   @ write \n"
          "vst4.8  {d28, d29, d30, d31},    [%[outptr3]]   @ write \n"

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
            "q12",
            "q13",
            "q14",
            "q15");
#endif
    }
    outptr3 += stride_w - 4;
    outptr2 += stride_w - 4;
    outptr1 += stride_w - 4;
    outptr0 += stride_w - 4;
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
  LITE_PARALLEL_COMMON_END();
  delete[] zerobuff;
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
  LITE_PARALLEL_COMMON_BEGIN(i, tid, h_in - 7, 0, 8) {
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
  LITE_PARALLEL_COMMON_END();
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
}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
