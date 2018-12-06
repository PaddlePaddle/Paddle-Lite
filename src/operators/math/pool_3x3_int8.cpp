/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef POOL_OP
#ifdef _OPENMP
#include <omp.h>
#endif
#include "framework/tensor.h"
#include "operators/math/pool_3x3.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON
#include <climits>
#include <iostream>
namespace paddle_mobile {
namespace operators {
namespace math {
using framework::Tensor;
using std::max;
using std::min;
using std::vector;
template <typename T>
static void make_paddings(const Tensor *input, Tensor *padded_input,
                          int32_t top, int32_t bottom, int32_t left,
                          int32_t right, T value) {
  const int32_t batch_size = input->dims()[0];
  const int32_t c_in = input->dims()[1];
  const int32_t h_in = input->dims()[2];
  const int32_t w_in = input->dims()[3];
  const int32_t h_padded = h_in + top + bottom;
  const int32_t w_padded = w_in + left + right;
  padded_input->Resize({batch_size, c_in, h_padded, w_padded});
  T *padded_input_data = padded_input->mutable_data<T>();
  const T *input_data = input->data<T>();
  const int32_t input_channel_stride = h_in * w_in;
  const int32_t input_batch_stride = c_in * input_channel_stride;
  const int32_t padded_channel_stride = h_padded * w_padded;
  const int32_t padded_batch_stride = c_in * padded_channel_stride;
  for (int i = 0; i < batch_size; ++i) {
#pragma omp parallel for
    for (int j = 0; j < c_in; ++j) {
      const T *img_in = input_data + j * input_channel_stride;
      T *img_padded = padded_input_data + j * padded_channel_stride;
      int k = 0;
      for (; k < top; ++k) {
        for (int l = 0; l < w_padded; ++l) {
          img_padded[l] = value;
        }
        img_padded += w_padded;
      }
      for (; k < top + h_in; ++k) {
        int l = 0;
        for (; l < left; ++l) {
          img_padded[l] = value;
        }
        memcpy(img_padded + left, img_in, w_in * sizeof(T));
        l += w_in;
        img_in += w_in;
        for (; l < w_padded; ++l) {
          img_padded[l] = value;
        }
        img_padded += w_padded;
      }
      for (; k < h_padded; ++k) {
        for (int l = 0; l < w_padded; ++l) {
          img_padded[l] = value;
        }
        img_padded += w_padded;
      }
    }
    input_data += input_batch_stride;
    padded_input_data += padded_batch_stride;
  }
  //  input_data = input->data<T>();
  //  std::cout << "+++++++++++++++++++Origin begin++++++++++++++++++++"
  //            << std::endl;
  //  for (int i = 0; i < 1; ++i) {
  //    for (int j = 0; j < 1; ++j) {
  //      const T *img_in = input_data + j * input_channel_stride;
  //      for (int k = 0; k < h_in; ++k) {
  //        for (int l = 0; l < w_in; ++l) {
  //          std::cout << (int32_t)*img_in << "\t";
  //          img_in++;
  //        }
  //        std::cout << std::endl;
  //      }
  //    }
  //    input_data += input_batch_stride;
  //  }
  //  std::cout << "+++++++++++++++++++Origin end++++++++++++++++++++" <<
  //  std::endl;
  //
  //  padded_input_data = padded_input->data<T>();
  //  std::cout << "******************Padding begin**********************"
  //            << std::endl;
  //  for (int i = 0; i < 1; ++i) {
  //    for (int j = 0; j < 1; ++j) {
  //      T *img_padded = padded_input_data + j * padded_channel_stride;
  //      for (int k = 0; k < h_padded; ++k) {
  //        for (int l = 0; l < w_padded; ++l) {
  //          std::cout << (int32_t)*img_padded << "\t";
  //          img_padded++;
  //        }
  //        std::cout << std::endl;
  //      }
  //    }
  //    padded_input_data += padded_batch_stride;
  //  }
  //  std::cout << "******************Padding end**********************"
  //            << std::endl;
}
void Pool3x3Maxs1_int8(const Tensor *input, Tensor *output, int32_t pad_h,
                       int32_t pad_w) {
  Tensor padded_input;
  if (pad_h != 0 && pad_w != 0) {
    int8_t value = -SCHAR_MAX;
    make_paddings(input, &padded_input, pad_h, pad_h, pad_w, pad_w, value);
    input = &padded_input;
  }
  const int32_t batch_size = input->dims()[0];
  const int32_t h_in = input->dims()[2];
  const int32_t w_in = input->dims()[3];
  const int8_t *input_data = input->data<int8_t>();
  const int32_t output_channels = output->dims()[1];
  const int32_t h_out = output->dims()[2];
  const int32_t w_out = output->dims()[3];
  int8_t *output_data = output->mutable_data<int8_t>();
  const int32_t outputdata_channel_stride = h_out * w_out;
  const int32_t inputdata_channel_stride = h_in * w_in;
  const int32_t input_batch_stride = output_channels * inputdata_channel_stride;
  const int32_t output_batch_stride =
      output_channels * outputdata_channel_stride;
  //    std::cout << "h_out = " << h_out << ", w_out=" << w_out << std::endl;
  for (int i = 0; i < batch_size; ++i) {
#pragma omp parallel for
    for (int j = 0; j < output_channels; ++j) {
      const int8_t *img_in = input_data + j * inputdata_channel_stride;
      int8_t *img_out = output_data + j * outputdata_channel_stride;
      for (int k = 0; k < h_out; ++k) {
        const int8_t *row0 = img_in + k * w_in;
        const int8_t *row1 = img_in + (k + 1) * w_in;
        const int8_t *row2 = img_in + (k + 2) * w_in;
#if __ARM_NEON
        int32_t nw = w_out >> 4;
        int32_t left_w = w_out & 0xf;
        int32_t nw1 = left_w >> 3;
        int32_t left_w1 = left_w & 0x7;
#if __aarch64__
        // TODO
#else
        if (nw > 0) {
#define LOOP_LABEL "1"
          // result: q15
          asm volatile(
              "vld1.8 {q0}, [%[row0]]! \n\t"  // q0=0-15
              "vld1.8 {q2}, [%[row1]]! \n\t"
              "vld1.8 {q4}, [%[row2]]! \n\t"

              LOOP_LABEL
              ": \n\t"
              "vld1.8 {q1}, [%[row0]]! \n\t"  // q1=16-31
              "vext.8 q6, q0, q1, #1 \n\t"
              "vext.8 q7, q0, q1, #2 \n\t"
              "vld1.8 {q3}, [%[row1]]! \n\t"
              "vmax.s8 q15, q0, q6 \n\t"
              "vmax.s8 q15, q15, q7 \n\t"
              "vext.8 q6, q2, q3, #1 \n\t"
              "vext.8 q7, q2, q3, #2 \n\t"
              "vld1.8 {q5}, [%[row2]]! \n\t"
              "vmax.s8 q14, q2, q6 \n\t"
              "vmax.s8 q14, q14, q7 \n\t"
              "vext.8 q6, q4, q5, #1 \n\t"
              "vext.8 q7, q4, q5, #2 \n\t"
              "vmax.s8 q13, q4, q6 \n\t"
              "vmax.s8 q13, q13, q7 \n\t"
              "vmax.s8 q15, q15, q14 \n\t"
              "vmax.s8 q15, q15, q13 \n\t"
              "vmov.s8 q0, q1 \n\t"
              "vmov.s8 q2, q3 \n\t"
              "vmov.s8 q4, q5 \n\t"
              "vst1.8 {q15}, [%[img_out]]! \n\t"
              "subs %[nw], #1 \n\t"
              "bne " LOOP_LABEL
              "b \n\t"
              "sub %[row0], #16 \n\t"
              "sub %[row1], #16 \n\t"
              "sub %[row2], #16 \n\t"
              : [nw] "+r"(nw), [row0] "+r"(row0), [row1] "+r"(row1),
                [row2] "+r"(row2), [img_out] "+r"(img_out)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q13", "q14", "q15");
#undef LOOP_LABEL
        }
        if (nw1 > 0 || left_w1 > 0) {
#define PADDLE_LABEL_LESS8 "1"
#define PADDLE_LABEL_LESS8_SAVE "2"
#define PADDLE_LABEL_OVER "3"
          // result: d15
          asm volatile(
              "vld1.8 {d0}, [%[row0]]! \n\t"  // d0=0-8
              "vld1.8 {d2}, [%[row1]]! \n\t"
              "vld1.8 {d4}, [%[row2]]! \n\t"
              "mov r0, #1 \n\t"
              "cmp %[nw1], #0 \n\t"
              "beq " PADDLE_LABEL_LESS8
              "f\n\t"
              "vld1.8 {d1}, [%[row0]]! \n\t"  // d1=9-15
              "vext.8 d6, d0, d1, #1 \n\t"
              "vext.8 d7, d0, d1, #2 \n\t"
              "vld1.8 {d3}, [%[row1]]! \n\t"
              "vmax.s8 d15, d0, d6 \n\t"
              "vmax.s8 d15, d15, d7 \n\t"
              "vext.8 d6, d2, d3, #1 \n\t"
              "vext.8 d7, d2, d3, #2 \n\t"
              "vld1.8 {d5}, [%[row2]]! \n\t"
              "vmax.s8 d14, d2, d6 \n\t"
              "vmax.s8 d14, d14, d7 \n\t"
              "vext.8 d6, d4, d5, #1 \n\t"
              "vext.8 d7, d4, d5, #2 \n\t"
              "vmax.s8 d13, d4, d6 \n\t"
              "vmax.s8 d13, d13, d7 \n\t"
              "vmax.s8 d15, d15, d14 \n\t"
              "vmax.s8 d15, d15, d13 \n\t"
              "vmov.s8 d0, d1 \n\t"
              "vmov.s8 d2, d3 \n\t"
              "vmov.s8 d4, d5 \n\t"
              "vst1.8 {d15}, [%[img_out]]! \n\t"

              PADDLE_LABEL_LESS8
              ": \n\t"
              "cmp %[left_w1], #0 \n\t"
              "beq " PADDLE_LABEL_OVER
              "f\n\t"
              "vld1.8 {d1}, [%[row0]] \n\t"  // d1=9-15
              "vext.8 d6, d0, d1, #1 \n\t"
              "vext.8 d7, d0, d1, #2 \n\t"
              "vld1.8 {d3}, [%[row1]] \n\t"
              "vmax.s8 d15, d0, d6 \n\t"
              "vmax.s8 d15, d15, d7 \n\t"
              "vext.8 d6, d2, d3, #1 \n\t"
              "vext.8 d7, d2, d3, #2 \n\t"
              "vld1.8 {d5}, [%[row2]] \n\t"
              "vmax.s8 d14, d2, d6 \n\t"
              "vmax.s8 d14, d14, d7 \n\t"
              "vext.8 d6, d4, d5, #1 \n\t"
              "vext.8 d7, d4, d5, #2 \n\t"
              "vmax.s8 d13, d4, d6 \n\t"
              "vmax.s8 d13, d13, d7 \n\t"
              "vmax.s8 d15, d15, d14 \n\t"
              "vmax.s8 d15, d15, d13 \n\t"

              PADDLE_LABEL_LESS8_SAVE
              ": \n\t"
              "vst1.8 {d15[0]}, [%[img_out]], r0\n\t"
              "add %[row0], %[row0], #1 \n\t"
              "add %[row1], %[row1], #1 \n\t"
              "add %[row2], %[row2], #1 \n\t"
              "vext.8 d15, d15, d15, #1 \n\t"
              "subs %[left_w1], #1 \n\t"
              "bgt " PADDLE_LABEL_LESS8_SAVE "b \n\t"

              PADDLE_LABEL_OVER ": \n\t"
              : [nw1] "+r"(nw1), [left_w1] "+r"(left_w1), [row0] "+r"(row0),
                [row1] "+r"(row1), [row2] "+r"(row2), [img_out] "+r"(img_out)
              :
              : "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6",
                "d7", "d13", "d14", "d15");
#undef PADDLE_LABEL_OVER
#undef PADDLE_LABEL_LESS8_SAVE
#undef PADDLE_LABEL_LESS8
        }
#endif  // __aarch64__
#else
        int32_t left = w_out;
        while (left > 0) {
          const int8_t max0 = std::max(std::max(row0[0], row0[1]), row0[2]);
          const int8_t max1 = std::max(std::max(row1[0], row1[1]), row1[2]);
          const int8_t max2 = std::max(std::max(row2[0], row2[1]), row2[2]);
          *img_out = std::max(std::max(max0, max1), max2);
          row0 += 1;
          row1 += 1;
          row2 += 1;
          img_out++;
          left--;
        }
#endif  // __ARM_NEON
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}
void Pool3x3Maxs2_int8(const Tensor *input, Tensor *output, int32_t pad_h,
                       int32_t pad_w) {
  Tensor padded_input;
  if (pad_h != 0 && pad_w != 0) {
    int8_t value = -SCHAR_MAX;
    make_paddings(input, &padded_input, pad_h, pad_h, pad_w, pad_w, value);
    input = &padded_input;
  }
  const int32_t batch_size = input->dims()[0];
  const int32_t h_in = input->dims()[2];
  const int32_t w_in = input->dims()[3];
  const int32_t output_channels = output->dims()[1];
  const int32_t h_out = output->dims()[2];
  const int32_t w_out = output->dims()[3];
  const int32_t outputdata_channel_stride = h_out * w_out;
  const int32_t inputdata_channel_stride = h_in * w_in;
  const int32_t output_batch_stride =
      output_channels * outputdata_channel_stride;
  const int32_t input_batch_stride = output_channels * inputdata_channel_stride;
  const int8_t *input_data = input->data<int8_t>();
  int8_t *output_data = output->mutable_data<int8_t>();
  for (int i = 0; i < batch_size; ++i) {
#pragma omp parallel for
    for (int j = 0; j < output_channels; ++j) {
      const int8_t *img_in = input_data + j * inputdata_channel_stride;
      int8_t *img_out = output_data + j * outputdata_channel_stride;
      for (int k = 0; k < h_out; ++k) {
        const int8_t *row0 = img_in + 2 * k * w_in;
        const int8_t *row1 = img_in + (2 * k + 1) * w_in;
        const int8_t *row2 = img_in + (2 * k + 2) * w_in;
#if __ARM_NEON
        int32_t nw = w_out >> 4;
        int32_t left_w = w_out & 0xf;
        int32_t nw1 = left_w >> 3;
        int32_t left_w1 = left_w & 0x7;
#if __aarch64__
        // TODO
#else
        if (nw > 0) {
#define LOOP_LABEL "1"
          // result: q15
          asm volatile(
              "vld2.8 {q0, q1}, [%[row0]]! \n\t"  // q0=0-30, q1=1-31
              "vld2.8 {q2, q3}, [%[row1]]! \n\t"
              "vld2.8 {q4, q5}, [%[row2]]! \n\t"

              LOOP_LABEL
              ": \n\t"
              "vmax.s8 q15, q0, q1 \n\t"
              "vld2.8 {q6, q7}, [%[row0]]! \n\t"  // q0=32-62, q1=33-63
              "vmax.s8 q14, q2, q3 \n\t"
              "vmax.s8 q13, q4, q5 \n\t"
              "vld2.8 {q8, q9}, [%[row1]]! \n\t"
              "vext.8 q0, q0, q6, #1 \n\t"
              "vmax.s8 q15, q15, q0 \n\t"
              "vld2.8 {q10, q11}, [%[row2]]! \n\t"
              "vext.8 q2, q2, q8, #1 \n\t"
              "vmax.s8 q14, q14, q2 \n\t"
              "vext.8 q4, q4, q10, #1 \n\t"
              "vmax.s8 q13, q13, q4 \n\t"
              "vmax.s8 q15, q15, q14 \n\t"
              "vmax.s8 q15, q15, q13 \n\t"
              "vmov.s8 q0, q6 \n\t"
              "vmov.s8 q1, q7 \n\t"
              "vmov.s8 q2, q8 \n\t"
              "vmov.s8 q3, q9 \n\t"
              "vmov.s8 q4, q10 \n\t"
              "vmov.s8 q5, q11 \n\t"
              "vst1.8 {q15}, [%[img_out]]! \n\t"
              "subs %[nw], #1 \n\t"
              "bne " LOOP_LABEL
              "b \n\t"
              "sub %[row0], #32 \n\t"
              "sub %[row1], #32 \n\t"
              "sub %[row2], #32 \n\t"
              : [nw] "+r"(nw), [row0] "+r"(row0), [row1] "+r"(row1),
                [row2] "+r"(row2), [img_out] "+r"(img_out)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q13", "q14", "q15");
#undef LOOP_LABEL
        }
        if (nw1 > 0 || left_w1 > 0) {
#define PADDLE_LABEL_LESS8 "1"
#define PADDLE_LABEL_LESS8_SAVE "2"
#define PADDLE_LABEL_OVER "3"
          // result: d15
          asm volatile(
              "vld2.8 {d0, d1}, [%[row0]]! \n\t"  // d0=0-14, d1=1-15
              "vld2.8 {d2, d3}, [%[row1]]! \n\t"
              "vld2.8 {d4, d5}, [%[row2]]! \n\t"
              "mov r0, #1 \n\t"
              "cmp %[nw1], #0 \n\t"
              "beq " PADDLE_LABEL_LESS8
              "f\n\t"
              "vmax.s8 d15, d0, d1 \n\t"
              "vld2.8 {d6, d7}, [%[row0]]! \n\t"  // d0=32-62, d1=33-63
              "vmax.s8 d14, d2, d3 \n\t"
              "vmax.s8 d13, d4, d5 \n\t"
              "vld2.8 {d8, d9}, [%[row1]]! \n\t"
              "vext.8 d0, d0, d6, #1 \n\t"
              "vmax.s8 d15, d15, d0 \n\t"
              "vld2.8 {d10, d11}, [%[row2]]! \n\t"
              "vext.8 d2, d2, d8, #1 \n\t"
              "vmax.s8 d14, d14, d2 \n\t"
              "vext.8 d4, d4, d10, #1 \n\t"
              "vmax.s8 d13, d13, d4 \n\t"
              "vmax.s8 d15, d15, d14 \n\t"
              "vmax.s8 d15, d15, d13 \n\t"
              "vmov.s8 d0, d6 \n\t"
              "vmov.s8 d1, d7 \n\t"
              "vmov.s8 d2, d8 \n\t"
              "vmov.s8 d3, d9 \n\t"
              "vmov.s8 d4, d10 \n\t"
              "vmov.s8 d5, d11 \n\t"
              "vst1.8 {d15}, [%[img_out]]! \n\t"

              PADDLE_LABEL_LESS8
              ": \n\t"
              "cmp %[left_w1], #0 \n\t"
              "beq " PADDLE_LABEL_OVER
              "f\n\t"
              "vmax.s8 d15, d0, d1 \n\t"
              "vld2.8 {d6, d7}, [%[row0]] \n\t"  // d0=32-62, d1=33-63
              "vmax.s8 d14, d2, d3 \n\t"
              "vmax.s8 d13, d4, d5 \n\t"
              "vld2.8 {d8, d9}, [%[row1]] \n\t"
              "vext.8 d0, d0, d6, #1 \n\t"
              "vmax.s8 d15, d15, d0 \n\t"
              "vld2.8 {d10, d11}, [%[row2]] \n\t"
              "vext.8 d2, d2, d8, #1 \n\t"
              "vmax.s8 d14, d14, d2 \n\t"
              "vext.8 d4, d4, d10, #1 \n\t"
              "vmax.s8 d13, d13, d4 \n\t"
              "vmax.s8 d15, d15, d14 \n\t"
              "vmax.s8 d15, d15, d13 \n\t"

              PADDLE_LABEL_LESS8_SAVE
              ": \n\t"
              "vst1.8 {d15[0]}, [%[img_out]], r0\n\t"
              "add %[row0], %[row0], #2 \n\t"
              "add %[row1], %[row1], #2 \n\t"
              "add %[row2], %[row2], #2 \n\t"
              "vext.8 d15, d15, d15, #1 \n\t"
              "subs %[left_w1], #1 \n\t"
              "bgt " PADDLE_LABEL_LESS8_SAVE "b \n\t"

              PADDLE_LABEL_OVER ": \n\t"
              : [nw1] "+r"(nw1), [left_w1] "+r"(left_w1), [row0] "+r"(row0),
                [row1] "+r"(row1), [row2] "+r"(row2), [img_out] "+r"(img_out)
              :
              : "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6",
                "d7", "d8", "d9", "d10", "d11", "d13", "d14", "d15");
#undef PADDLE_LABEL_OVER
#undef PADDLE_LABEL_LESS8_SAVE
#undef PADDLE_LABEL_LESS8
        }
#endif  // __aarch64__
#else
        int32_t left = w_out;
        while (left > 0) {
          const int8_t max0 = std::max(std::max(row0[0], row0[1]), row0[2]);
          const int8_t max1 = std::max(std::max(row1[0], row1[1]), row1[2]);
          const int8_t max2 = std::max(std::max(row2[0], row2[1]), row2[2]);
          *img_out = std::max(std::max(max0, max1), max2);
          row0 += 2;
          row1 += 2;
          row2 += 2;
          img_out++;
          left--;
        }
#endif  // __ARM_NEON
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}
void Pool3x3Max_int8(const vector<int> &strides, const vector<int> &paddings,
                     const Tensor *input, Tensor *output) {
  const int batch_size = input->dims()[0];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const int output_channels = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  //  const int _kernel_size = 3;
  const int stride = strides[0];
  //  const int stride_width = strides[1];
  const int padding = paddings[0];
  //  const int padding_width = paddings[1];
  const int8_t negative_max = -SCHAR_MAX;
  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;
  const int8_t *input_data = input->data<int8_t>();
  int8_t *output_data = output->mutable_data<int8_t>();
  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  for (int i = 0; i < batch_size; ++i) {
#pragma omp parallel for
    for (int c = 0; c < output_channels; ++c) {
      const int8_t *input_seg = input_data + c * input_channel_stride;
      int8_t *output_seg = output_data + c * output_channel_stride;
      for (int ph = 0; ph < output_height; ph++) {
        int hstart = ph * stride - padding;
        int hend = min(hstart + 3, input_height);
        hstart = max(hstart, 0);
        for (int pw = 0; pw < output_width; pw++) {
          int wstart = pw * stride - padding;
          int wend = min(wstart + 3, input_width);
          wstart = max(wstart, 0);
          const int8_t *pos1 = input_seg + hstart * input_width + wstart;
          const int8_t *pos2 = input_seg + (hstart + 1) * input_width + wstart;
          const int8_t *pos3 = input_seg + (hstart + 2) * input_width + wstart;
          int8_t *output_ptr = output_seg + ph * output_width + pw;
          if (hend - hstart != 3 || wend - wstart != 3) {
            int8_t max_value = -SCHAR_MAX;
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                int8_t value = input_seg[h * input_width + w];
                if (value > max_value) {
                  max_value = value;
                }
              }
            }
            output_seg[ph * output_width + pw] = max_value;
          } else {
#if __ARM_NEON
#if __aarch64__
          // TODO
#else
            asm volatile(
                "vld1.8  {d0}, [%[pos1]]        \n\t"
                "vld1.8  {d1}, [%[pos2]]        \n\t"
                "vld1.8  {d2}, [%[pos3]]        \n\t"
                "vmax.s8 d3, d0, d1            \n\t"
                "vmax.s8 d4, d2, d3            \n\t"
                "vmov.s8 d4[3],  %[negative_max] \n\t"
                "vpmax.s8  d5, d4, d4            \n\t"
                "vpmax.s8  d6, d5, d5             \n\t"
                "vst1.8 {d6[0]},[%[output_ptr]]    \n\t"
                :
                : [pos1] "r"(pos1), [pos2] "r"(pos2), [pos3] "r"(pos3),
                  [output_ptr] "r"(output_ptr), [negative_max] "r"(negative_max)
                : "memory", "q0", "q1", "q2", "q3");
#endif
#else
            const int8_t max0 = std::max(std::max(pos1[0], pos1[1]), pos1[2]);
            const int8_t max1 = std::max(std::max(pos2[0], pos2[1]), pos2[2]);
            const int8_t max2 = std::max(std::max(pos3[0], pos3[1]), pos3[2]);
            *output_ptr = std::max(std::max(max0, max1), max2);
#endif  // __ARM_NEON
          }
        }
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
#endif
