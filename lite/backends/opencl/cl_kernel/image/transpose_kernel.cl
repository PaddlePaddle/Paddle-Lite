/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

#define GET_THREAD_INDEX               \
  const int out_c = get_global_id(0);  \
  const int out_w = get_global_id(1);  \
  const int out_nh = get_global_id(2); \
  const int out_n = out_nh / out_H;    \
  const int out_h = out_nh % out_H;

#define SET_IDX_NON_CHANN_PERM(out_n, out_c, out_w, out_h) \
  const int in_n = out_n;                                  \
  const int in_c = out_c;                                  \
  const int in_h = out_w;                                  \
  const int in_w = out_h;

#define SET_IDX_CHANN_PERM_H(out_n, out_h, out_c, out_w) \
  const int in_n = out_n;                                \
  const int in_c = out_h / 4;                            \
  const int in_h0 = out_c * 4;                           \
  const int in_h1 = out_c * 4 + 1;                       \
  const int in_h2 = out_c * 4 + 2;                       \
  const int in_h3 = out_c * 4 + 3;                       \
  const int in_w = out_w;

#define SET_IDX_CHANN_PERM_W(out_n, out_h, out_w, out_c) \
  const int in_n = out_n;                                \
  const int in_c = out_h / 4;                            \
  const int in_h = out_w;                                \
  const int in_w0 = out_c * 4;                           \
  const int in_w1 = out_c * 4 + 1;                       \
  const int in_w2 = out_c * 4 + 2;                       \
  const int in_w3 = out_c * 4 + 3;

#define SET_IDX_CHANN_PERM_N(out_c, out_n, out_h, out_w) \
  const int in_n0 = out_c * 4;                           \
  const int in_n1 = out_c * 4 + 1;                       \
  const int in_n2 = out_c * 4 + 2;                       \
  const int in_n3 = out_c * 4 + 3;                       \
  const int in_c = out_n / 4;                            \
  const int in_h = out_h;                                \
  const int in_w = out_w;

#define SET_OUTPUT_POS                  \
  int2 output_pos;                      \
  output_pos.x = out_c * out_W + out_w; \
  output_pos.y = out_nh;

#define SET_INPUT_POS(in_w, in_h)   \
  int2 input_pos;                   \
  input_pos.x = in_c * in_W + in_w; \
  input_pos.y = in_n * in_H + in_h;

#define DECL_VARIABLE \
  int2 input_pos0;    \
  int2 input_pos1;    \
  int2 input_pos2;    \
  int2 input_pos3;    \
  CL_DTYPE4 input0;   \
  CL_DTYPE4 input1;   \
  CL_DTYPE4 input2;   \
  CL_DTYPE4 input3;   \
  CL_DTYPE4 output;

#define SET_FOUR_INPUT_POS_PERM_H     \
  input_pos0.x = in_W * in_c + in_w;  \
  input_pos0.y = in_n * in_H + in_h0; \
  input_pos1.x = in_W * in_c + in_w;  \
  input_pos1.y = in_n * in_H + in_h1; \
  input_pos2.x = in_W * in_c + in_w;  \
  input_pos2.y = in_n * in_H + in_h2; \
  input_pos3.x = in_W * in_c + in_w;  \
  input_pos3.y = in_n * in_H + in_h3;

#define SET_FOUR_INPUT_POS_PERM_W     \
  input_pos0.x = in_W * in_c + in_w0; \
  input_pos0.y = in_n * in_H + in_h;  \
  input_pos1.x = in_W * in_c + in_w1; \
  input_pos1.y = in_n * in_H + in_h;  \
  input_pos2.x = in_W * in_c + in_w2; \
  input_pos2.y = in_n * in_H + in_h;  \
  input_pos3.x = in_W * in_c + in_w3; \
  input_pos3.y = in_n * in_H + in_h;

#define SET_FOUR_INPUT_POS_PERM_N     \
  input_pos0.x = in_W * in_c + in_w;  \
  input_pos0.y = in_n0 * in_H + in_h; \
  input_pos1.x = in_W * in_c + in_w;  \
  input_pos1.y = in_n1 * in_H + in_h; \
  input_pos2.x = in_W * in_c + in_w;  \
  input_pos2.y = in_n2 * in_H + in_h; \
  input_pos3.x = in_W * in_c + in_w;  \
  input_pos3.y = in_n3 * in_H + in_h;

#define COMPUTE_OUTPUT_DATA(out_h)                                           \
  input0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos0);   \
  if (out_h % 4 == 0) {                                                      \
    output.x = input0.x;                                                     \
  } else if (out_h % 4 == 1) {                                               \
    output.x = input0.y;                                                     \
  } else if (out_h % 4 == 2) {                                               \
    output.x = input0.z;                                                     \
  } else {                                                                   \
    output.x = input0.w;                                                     \
  }                                                                          \
  if (out_C - out_c * 4 >= 2) {                                              \
    input1 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos1); \
    if (out_h % 4 == 0) {                                                    \
      output.y = input1.x;                                                   \
    } else if (out_h % 4 == 1) {                                             \
      output.y = input1.y;                                                   \
    } else if (out_h % 4 == 2) {                                             \
      output.y = input1.z;                                                   \
    } else {                                                                 \
      output.y = input1.w;                                                   \
    }                                                                        \
  } else {                                                                   \
    output.y = 0.0f;                                                         \
  }                                                                          \
  if (out_C - out_c * 4 >= 3) {                                              \
    input2 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos2); \
    if (out_h % 4 == 0) {                                                    \
      output.z = input2.x;                                                   \
    } else if (out_h % 4 == 1) {                                             \
      output.z = input2.y;                                                   \
    } else if (out_h % 4 == 2) {                                             \
      output.z = input2.z;                                                   \
    } else {                                                                 \
      output.z = input2.w;                                                   \
    }                                                                        \
  } else {                                                                   \
    output.z = 0.0f;                                                         \
  }                                                                          \
  if (out_C - out_c * 4 >= 4) {                                              \
    input3 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos3); \
    if (out_h % 4 == 0) {                                                    \
      output.w = input3.x;                                                   \
    } else if (out_h % 4 == 1) {                                             \
      output.w = input3.y;                                                   \
    } else if (out_h % 4 == 2) {                                             \
      output.w = input3.z;                                                   \
    } else {                                                                 \
      output.w = input3.w;                                                   \
    }                                                                        \
  } else {                                                                   \
    output.w = 0.0f;                                                         \
  }

#define WRITE_OUTPUT_DATA \
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);

#ifdef NON_CHANNEL_PERMUTATION
__kernel void transpose4d_perm_without_chann(__read_only image2d_t input_image,
                                             __write_only image2d_t
                                                 output_image,
                                             __private const int out_C,
                                             __private const int out_H,
                                             __private const int out_W,
                                             __private const int in_W,
                                             __private const int in_H) {
  GET_THREAD_INDEX
  SET_OUTPUT_POS
#ifdef PERM0132
  SET_IDX_NON_CHANN_PERM(out_n, out_c, out_w, out_h)
#elif defined(PERM2103)
  SET_IDX_NON_CHANN_PERM(out_h, out_c, out_n, out_w)
#elif defined(PERM2130)
  SET_IDX_NON_CHANN_PERM(out_w, out_c, out_n, out_h)
#elif defined(PERM3102)
  SET_IDX_NON_CHANN_PERM(out_h, out_c, out_w, out_n)
#elif defined(PERM3120)
  SET_IDX_NON_CHANN_PERM(out_w, out_c, out_h, out_n)
#endif
  SET_INPUT_POS(in_w, in_h)
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos);
  WRITE_OUTPUT_DATA
}
#endif

__kernel void transpose4d_perm_with_channel(__read_only image2d_t input_image,
                                            __write_only image2d_t output_image,
                                            __private const int out_C,
                                            __private const int out_H,
                                            __private const int out_W,
                                            __private const int in_W,
                                            __private const int in_H) {
  GET_THREAD_INDEX
  SET_OUTPUT_POS
  DECL_VARIABLE
#if defined(N2N) && defined(C2H) && defined(H2C) && defined(W2W)
  SET_IDX_CHANN_PERM_H(out_n, out_h, out_c, out_w)
#elif defined(N2N) && defined(C2W) && defined(H2C) && defined(W2H)
  SET_IDX_CHANN_PERM_H(out_n, out_w, out_c, out_h)
#elif defined(N2N) && defined(C2H) && defined(H2W) && defined(W2C)
  SET_IDX_CHANN_PERM_W(out_n, out_h, out_w, out_c);
#elif defined(N2N) && defined(C2W) && defined(H2H) && defined(W2C)
  SET_IDX_CHANN_PERM_W(out_n, out_w, out_h, out_c);
#elif defined(N2C) && defined(C2N) && defined(H2H) && defined(W2W)
  SET_IDX_CHANN_PERM_N(out_c, out_n, out_h, out_w)
#elif defined(N2C) && defined(C2N) && defined(H2W) && defined(W2H)
  SET_IDX_CHANN_PERM_N(out_c, out_n, out_w, out_h)
#elif defined(N2H) && defined(C2N) && defined(H2C) && defined(W2W)
  SET_IDX_CHANN_PERM_H(out_h, out_n, out_c, out_w)
#elif defined(N2W) && defined(C2N) && defined(H2C) && defined(W2H)
  SET_IDX_CHANN_PERM_H(out_w, out_n, out_c, out_h)
#elif defined(N2H) && defined(C2N) && defined(H2W) && defined(W2C)
  SET_IDX_CHANN_PERM_W(out_h, out_n, out_w, out_c)
#elif defined(N2W) && defined(C2N) && defined(H2H) && defined(W2C)
  SET_IDX_CHANN_PERM_W(out_w, out_n, out_h, out_c)
#elif defined(N2C) && defined(C2H) && defined(H2N) && defined(W2W)
  SET_IDX_CHANN_PERM_N(out_c, out_h, out_n, out_w)
#elif defined(N2C) && defined(C2W) && defined(H2N) && defined(W2H)
  SET_IDX_CHANN_PERM_N(out_c, out_w, out_n, out_h)
#elif defined(N2H) && defined(C2W) && defined(H2N) && defined(W2C)
  SET_IDX_CHANN_PERM_W(out_h, out_w, out_n, out_c)
#elif defined(N2W) && defined(C2H) && defined(H2N) && defined(W2C)
  SET_IDX_CHANN_PERM_W(out_w, out_h, out_n, out_c)
#elif defined(N2C) && defined(C2H) && defined(H2W) && defined(W2N)
  SET_IDX_CHANN_PERM_N(out_c, out_h, out_w, out_n)
#elif defined(N2C) && defined(C2W) && defined(H2H) && defined(W2N)
  SET_IDX_CHANN_PERM_N(out_c, out_w, out_h, out_n)
#elif defined(N2H) && defined(C2W) && defined(H2C) && defined(W2N)
  SET_IDX_CHANN_PERM_H(out_h, out_w, out_c, out_n)
#elif defined(N2W) && defined(C2H) && defined(H2C) && defined(W2N)
  SET_IDX_CHANN_PERM_H(out_w, out_h, out_c, out_n)
#endif

#ifdef H2C
  SET_FOUR_INPUT_POS_PERM_H
#elif defined(W2C)
  SET_FOUR_INPUT_POS_PERM_W
#elif defined(N2C)
  SET_FOUR_INPUT_POS_PERM_N
#endif

#ifdef C2H
  COMPUTE_OUTPUT_DATA(out_h)
#elif defined(C2W)
  COMPUTE_OUTPUT_DATA(out_w)
#elif defined(C2N)
  COMPUTE_OUTPUT_DATA(out_n)
#endif
  WRITE_OUTPUT_DATA
}

__kernel void transpose_2d(__read_only image2d_t input_image,
                           __write_only image2d_t output_image,
                           __private const int out_C,
                           __private const int out_H,
                           __private const int out_W,
                           __private const int in_W,
                           __private const int in_H) {
  GET_THREAD_INDEX
  SET_IDX_NON_CHANN_PERM(out_n, out_c, out_w, out_h)
  int2 input_pos;
  input_pos.x = in_c * in_W + in_w;
  input_pos.y = in_h;
  int2 output_pos;
  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_h;
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos);
  WRITE_OUTPUT_DATA
}
