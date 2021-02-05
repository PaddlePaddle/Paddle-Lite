/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/conv_utils.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

// From: https://stackoverflow.com/a/25627536
static inline void transpose8_ps(__m256& row0,  // NOLINT
                                 __m256& row1,  // NOLINT
                                 __m256& row2,  // NOLINT
                                 __m256& row3,  // NOLINT
                                 __m256& row4,  // NOLINT
                                 __m256& row5,  // NOLINT
                                 __m256& row6,  // NOLINT
                                 __m256& row7   // NOLINT
                                 ) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(row0, row1);
  __t1 = _mm256_unpackhi_ps(row0, row1);
  __t2 = _mm256_unpacklo_ps(row2, row3);
  __t3 = _mm256_unpackhi_ps(row2, row3);
  __t4 = _mm256_unpacklo_ps(row4, row5);
  __t5 = _mm256_unpackhi_ps(row4, row5);
  __t6 = _mm256_unpacklo_ps(row6, row7);
  __t7 = _mm256_unpackhi_ps(row6, row7);
  __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
  __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
  __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
  __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
  __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
  __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
  __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
  __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
  row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

// input  [bs, ic, ih, iw] => [bs, ic/8, ih, iw, 8]
// filter [oc, 01, ih, iw] => [01, ic/8, ih, iw, 8] for depthwise
void pack8_m256(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter) {
  int batch_size, input_channel, input_height, input_width;
  if (is_filter) {
    batch_size = 1;
    input_channel = input->dims()[0];
    input_height = input->dims()[2];
    input_width = input->dims()[3];
  } else {
    batch_size = input->dims()[0];
    input_channel = input->dims()[1];
    input_height = input->dims()[2];
    input_width = input->dims()[3];
  }
  CHECK_EQ((input_channel & 7), 0);
  const float* input_data = input->data<float>();

  const int kernel_size = input_height * input_width;
  const int pack_step = 8 * kernel_size;
  const int batch_step = channel_num * pack_step;

  output->Resize({batch_size, channel_num, input_height, input_width, 8});
  float* output_data = output->mutable_data<float>();

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* input_ptr = input_data + bs * batch_step + ic * pack_step;

      const float* r0 = (input_ptr);
      const float* r1 = (input_ptr + kernel_size);
      const float* r2 = (input_ptr + kernel_size * 2);
      const float* r3 = (input_ptr + kernel_size * 3);
      const float* r4 = (input_ptr + kernel_size * 4);
      const float* r5 = (input_ptr + kernel_size * 5);
      const float* r6 = (input_ptr + kernel_size * 6);
      const float* r7 = (input_ptr + kernel_size * 7);

#if __AVX__
      int loop_num = kernel_size >> 3;
      int remain = kernel_size & 7;
#else
      int remain = kernel_size;
#endif

#if __AVX__
      for (; loop_num > 0; loop_num--) {
        __m256 _row0 = _mm256_loadu_ps(r0);
        __m256 _row1 = _mm256_loadu_ps(r1);
        __m256 _row2 = _mm256_loadu_ps(r2);
        __m256 _row3 = _mm256_loadu_ps(r3);
        __m256 _row4 = _mm256_loadu_ps(r4);
        __m256 _row5 = _mm256_loadu_ps(r5);
        __m256 _row6 = _mm256_loadu_ps(r6);
        __m256 _row7 = _mm256_loadu_ps(r7);
        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
        _mm256_storeu_ps(output_data, _row0);
        _mm256_storeu_ps(output_data + 8, _row1);
        _mm256_storeu_ps(output_data + 16, _row2);
        _mm256_storeu_ps(output_data + 24, _row3);
        _mm256_storeu_ps(output_data + 32, _row4);
        _mm256_storeu_ps(output_data + 40, _row5);
        _mm256_storeu_ps(output_data + 48, _row6);
        _mm256_storeu_ps(output_data + 56, _row7);
        r0 += 8;
        r1 += 8;
        r2 += 8;
        r3 += 8;
        r4 += 8;
        r5 += 8;
        r6 += 8;
        r7 += 8;
        output_data += 64;
      }
#endif

      for (; remain > 0; remain--) {
        output_data[0] = *r0++;
        output_data[1] = *r1++;
        output_data[2] = *r2++;
        output_data[3] = *r3++;
        output_data[4] = *r4++;
        output_data[5] = *r5++;
        output_data[6] = *r6++;
        output_data[7] = *r7++;
        output_data += 8;
      }  // end of remain
    }    // end of interation_num
  }      // end of batch_size
}

// input  [bs, ic, ih, iw] => [bs, ic/4, ih, iw, 4]
// filter [oc, 01, ih, iw] => [01, ic/4, ih, iw, 4] for depthwise
void pack4_m128(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter) {
  int batch_size, input_channel, input_height, input_width;
  if (is_filter) {
    batch_size = 1;
    input_channel = input->dims()[0];
    input_height = input->dims()[2];
    input_width = input->dims()[3];
  } else {
    batch_size = input->dims()[0];
    input_channel = input->dims()[1];
    input_height = input->dims()[2];
    input_width = input->dims()[3];
  }
  CHECK_EQ((input_channel & 3), 0);
  const float* input_data = input->data<float>();

  const int kernel_size = input_height * input_width;
  const int pack_step = 4 * kernel_size;
  const int batch_step = channel_num * pack_step;

  output->Resize({batch_size, channel_num, input_height, input_width, 4});
  float* output_data = output->mutable_data<float>();

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* input_ptr = input_data + bs * batch_step + ic * pack_step;

      const float* r0 = (input_ptr);
      const float* r1 = (input_ptr + kernel_size);
      const float* r2 = (input_ptr + kernel_size * 2);
      const float* r3 = (input_ptr + kernel_size * 3);

#if __AVX__
      int loop_num = kernel_size >> 2;
      int remain = kernel_size & 3;
#else
      int remain = kernel_size;
#endif

#if __AVX__
      for (; loop_num > 0; loop_num--) {
        __m128 _row0 = _mm_loadu_ps(r0);
        __m128 _row1 = _mm_loadu_ps(r1);
        __m128 _row2 = _mm_loadu_ps(r2);
        __m128 _row3 = _mm_loadu_ps(r3);
        _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);
        _mm_storeu_ps(output_data, _row0);
        _mm_storeu_ps(output_data + 4, _row1);
        _mm_storeu_ps(output_data + 8, _row2);
        _mm_storeu_ps(output_data + 12, _row3);
        r0 += 4;
        r1 += 4;
        r2 += 4;
        r3 += 4;
        output_data += 16;
      }
#endif

      for (; remain > 0; remain--) {
        output_data[0] = *r0++;
        output_data[1] = *r1++;
        output_data[2] = *r2++;
        output_data[3] = *r3++;
        output_data += 4;
      }
    }  // end of for ic
  }    // end of for bs
}

// output_trans [bs, oc/8, oh, ow, 8] => output [bs, oc, oh, ow]
void unpack8_m256(lite::Tensor* input, lite::Tensor* output) {
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  const int kernel_size = input_height * input_width;
  const int pack_step = 8 * kernel_size;
  const int batch_step = channel_num * pack_step;

  output->Resize({batch_size, channel_num * 8, input_height, input_width});
  float* output_data = output->mutable_data<float>();

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* r0 = input_data + bs * batch_step + ic * pack_step;
      float* output_ptr = output_data + bs * batch_step + ic * pack_step;

      float* outptr0 = (output_ptr);
      float* outptr1 = (output_ptr + kernel_size);
      float* outptr2 = (output_ptr + kernel_size * 2);
      float* outptr3 = (output_ptr + kernel_size * 3);
      float* outptr4 = (output_ptr + kernel_size * 4);
      float* outptr5 = (output_ptr + kernel_size * 5);
      float* outptr6 = (output_ptr + kernel_size * 6);
      float* outptr7 = (output_ptr + kernel_size * 7);

#if __AVX__
      int loop_num = kernel_size >> 3;
      int remain = kernel_size & 7;
#else
      int remain = kernel_size;
#endif

#if __AVX__
      for (; loop_num > 0; loop_num--) {
        __m256 _row0 = _mm256_loadu_ps(r0);
        __m256 _row1 = _mm256_loadu_ps(r0 + 8);
        __m256 _row2 = _mm256_loadu_ps(r0 + 16);
        __m256 _row3 = _mm256_loadu_ps(r0 + 24);
        __m256 _row4 = _mm256_loadu_ps(r0 + 32);
        __m256 _row5 = _mm256_loadu_ps(r0 + 40);
        __m256 _row6 = _mm256_loadu_ps(r0 + 48);
        __m256 _row7 = _mm256_loadu_ps(r0 + 56);
        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
        _mm256_storeu_ps(outptr0, _row0);
        _mm256_storeu_ps(outptr1, _row1);
        _mm256_storeu_ps(outptr2, _row2);
        _mm256_storeu_ps(outptr3, _row3);
        _mm256_storeu_ps(outptr4, _row4);
        _mm256_storeu_ps(outptr5, _row5);
        _mm256_storeu_ps(outptr6, _row6);
        _mm256_storeu_ps(outptr7, _row7);
        r0 += 64;
        outptr0 += 8;
        outptr1 += 8;
        outptr2 += 8;
        outptr3 += 8;
        outptr4 += 8;
        outptr5 += 8;
        outptr6 += 8;
        outptr7 += 8;
      }
#endif

      for (; remain > 0; remain--) {
        *outptr0++ = r0[0];
        *outptr1++ = r0[1];
        *outptr2++ = r0[2];
        *outptr3++ = r0[3];
        *outptr4++ = r0[4];
        *outptr5++ = r0[5];
        *outptr6++ = r0[6];
        *outptr7++ = r0[7];
        r0 += 8;
      }  // end of remain
    }    // end of iteration_num
  }      // end of batch_size
}

// output_trans [bs, oc/4, oh, ow, 4] => output [bs, oc, oh, ow]
void unpack4_m128(lite::Tensor* input, lite::Tensor* output) {
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  const int kernel_size = input_height * input_width;
  const int pack_step = 4 * kernel_size;
  const int batch_step = channel_num * pack_step;

  output->Resize({batch_size, channel_num * 4, input_height, input_width});
  float* output_data = output->mutable_data<float>();

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* r0 = input_data + bs * batch_step + ic * pack_step;
      float* output_ptr = output_data + bs * batch_step + ic * pack_step;

      float* outptr0 = (output_ptr);
      float* outptr1 = (output_ptr + kernel_size);
      float* outptr2 = (output_ptr + kernel_size * 2);
      float* outptr3 = (output_ptr + kernel_size * 3);

#if __AVX__
      int loop_num = kernel_size >> 2;
      int remain = kernel_size & 3;
#else
      int remain = kernel_size;
#endif

#if __AVX__
      for (; loop_num > 0; loop_num--) {
        __m128 _row0 = _mm_loadu_ps(r0);
        __m128 _row1 = _mm_loadu_ps(r0 + 4);
        __m128 _row2 = _mm_loadu_ps(r0 + 8);
        __m128 _row3 = _mm_loadu_ps(r0 + 12);
        _MM_TRANSPOSE4_PS(_row0, _row1, _row2, _row3);
        _mm_storeu_ps(outptr0, _row0);
        _mm_storeu_ps(outptr1, _row1);
        _mm_storeu_ps(outptr2, _row2);
        _mm_storeu_ps(outptr3, _row3);
        r0 += 16;
        outptr0 += 4;
        outptr1 += 4;
        outptr2 += 4;
        outptr3 += 4;
      }
#endif

      for (; remain > 0; remain--) {
        *outptr0++ = r0[0];
        *outptr1++ = r0[1];
        *outptr2++ = r0[2];
        *outptr3++ = r0[3];
        r0 += 4;
      }
    }  // end of for ic
  }    // end of for bs
}

void padding8_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings) {
  CHECK_EQ(paddings.size(), 4UL);
  int top = paddings[0];
  int bottom = paddings[1];
  int left = paddings[2];
  int right = paddings[3];

  if (top == 0 && bottom == 0 && left == 0 && right == 0) {
    output->ShareDataWith(*input);
    return;
  }

  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const auto* input_data = input->data<float>();

  int out_height = input_height + top + bottom;
  int out_width = input_width + left + right;

  // output [bs, ic/8, oh, ow, 8]
  output->Resize({batch_size, channel_num, out_height, out_width, 8});
  auto output_data = output->mutable_data<float>();

  int top_size = top * out_width;
  int bottom_size = bottom * out_width;

  __m256 pad_val = _mm256_set1_ps(0.f);

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      // fill top
      for (int y = 0; y < top_size; ++y) {
        _mm256_storeu_ps(output_data, pad_val);
        output_data += 8;
      }
      // fill center
      for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < left; ++x) {
          _mm256_storeu_ps(output_data, pad_val);
          output_data += 8;
        }
        for (int x = 0; x < input_width; ++x) {
          _mm256_storeu_ps(output_data, _mm256_loadu_ps(input_data));
          input_data += 8;
          output_data += 8;
        }
        for (int x = 0; x < right; ++x) {
          _mm256_storeu_ps(output_data, pad_val);
          output_data += 8;
        }
      }
      // fill bottom
      for (int y = 0; y < bottom_size; ++y) {
        _mm256_storeu_ps(output_data, pad_val);
        output_data += 8;
      }
    }
  }
}

void padding4_m128(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings) {
  CHECK_EQ(paddings.size(), 4UL);
  int top = paddings[0];
  int bottom = paddings[1];
  int left = paddings[2];
  int right = paddings[3];

  if (top == 0 && bottom == 0 && left == 0 && right == 0) {
    output->ShareDataWith(*input);
    return;
  }

  // input [bs, ic/4, ih, iw, 4]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const auto* input_data = input->data<float>();

  int out_height = input_height + top + bottom;
  int out_width = input_width + left + right;

  // output [bs, ic/4, oh, ow, 4]
  output->Resize({batch_size, channel_num, out_height, out_width, 4});
  auto output_data = output->mutable_data<float>();

  int top_size = top * out_width;
  int bottom_size = bottom * out_width;

  __m128 pad_val = _mm_set1_ps(0.f);

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      // fill top
      for (int y = 0; y < top_size; ++y) {
        _mm_storeu_ps(output_data, pad_val);
        output_data += 4;
      }
      // fill center
      for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < left; ++x) {
          _mm_storeu_ps(output_data, pad_val);
          output_data += 4;
        }
        for (int x = 0; x < input_width; ++x) {
          _mm_storeu_ps(output_data, _mm_loadu_ps(input_data));
          input_data += 4;
          output_data += 4;
        }
        for (int x = 0; x < right; ++x) {
          _mm_storeu_ps(output_data, pad_val);
          output_data += 4;
        }
      }
      // fill bottom
      for (int y = 0; y < bottom_size; ++y) {
        _mm_storeu_ps(output_data, pad_val);
        output_data += 4;
      }
    }
  }
}

void padding1_float(lite::Tensor* input,
                    lite::Tensor* output,
                    const std::vector<int>& paddings) {
  CHECK_EQ(paddings.size(), 4UL);
  int top = paddings[0];
  int bottom = paddings[1];
  int left = paddings[2];
  int right = paddings[3];

  if (top == 0 && bottom == 0 && left == 0 && right == 0) {
    output->ShareDataWith(*input);
    return;
  }

  // input [bs, ic, ih, iw]
  CHECK_EQ(input->dims().size(), 4UL);
  int batch_size = input->dims()[0];
  int input_channel = input->dims()[1];
  int input_height = input->dims()[2];
  int input_width = input->dims()[3];
  const auto* input_data = input->data<float>();

  int out_height = input_height + top + bottom;
  int out_width = input_width + left + right;

  output->Resize({batch_size, input_channel, out_height, out_width});
  auto output_data = output->mutable_data<float>();

  int top_size = top * out_width;
  int bottom_size = bottom * out_width;

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < input_channel; ++ic) {
      // fill top
      memset(output_data, 0, sizeof(float) * top_size);
      output_data += top_size;
      // fill center
      for (int y = 0; y < input_height; ++y) {
        memset(output_data, 0, sizeof(float) * left);
        output_data += left;
        memcpy(output_data, input_data, sizeof(float) * input_width);
        output_data += input_width;
        input_data += input_width;
        memset(output_data, 0, sizeof(float) * right);
        output_data += right;
      }
      // fill bottom
      memset(output_data, 0, sizeof(float) * bottom_size);
      output_data += bottom_size;
    }
  }
}

void pack_padding8_m256(lite::Tensor* input,
                        lite::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings) {
  CHECK_EQ(input->dims().size(), 4UL);
  int batch_size = input->dims()[0];
  int input_channel = input->dims()[1];
  int input_height = input->dims()[2];
  int input_width = input->dims()[3];

  CHECK_EQ((input_channel & 7), 0);
  const float* input_data = input->data<float>();

  CHECK_EQ(paddings.size(), 4UL);
  int top = paddings[0];
  int bottom = paddings[1];
  int left = paddings[2];
  int right = paddings[3];

  // in
  const int kernel_size = input_height * input_width;
  const int pack_step = 8 * kernel_size;
  const int batch_step = channel_num * pack_step;

  // out
  int out_height = input_height + top + bottom;
  int out_width = input_width + left + right;

  // output [bs, ic/8, oh, ow, 8]
  output->Resize({batch_size, channel_num, out_height, out_width, 8});
  auto output_data = output->mutable_data<float>();

  int top_size = top * out_width;
  int bottom_size = bottom * out_width;

  __m256 pad_val = _mm256_set1_ps(0.f);

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* input_ptr = input_data + bs * batch_step + ic * pack_step;

      const float* r0 = (input_ptr);
      const float* r1 = (input_ptr + kernel_size);
      const float* r2 = (input_ptr + kernel_size * 2);
      const float* r3 = (input_ptr + kernel_size * 3);
      const float* r4 = (input_ptr + kernel_size * 4);
      const float* r5 = (input_ptr + kernel_size * 5);
      const float* r6 = (input_ptr + kernel_size * 6);
      const float* r7 = (input_ptr + kernel_size * 7);

      // fill top
      for (int y = 0; y < top_size; ++y) {
        _mm256_storeu_ps(output_data, pad_val);
        output_data += 8;
      }
      // fill center
      for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < left; ++x) {
          _mm256_storeu_ps(output_data, pad_val);
          output_data += 8;
        }
        // pack and transpose
        int pos = 0;
        for (; pos + 7 < input_width; pos += 8) {
          __m256 _row0 = _mm256_loadu_ps(r0);
          __m256 _row1 = _mm256_loadu_ps(r1);
          __m256 _row2 = _mm256_loadu_ps(r2);
          __m256 _row3 = _mm256_loadu_ps(r3);
          __m256 _row4 = _mm256_loadu_ps(r4);
          __m256 _row5 = _mm256_loadu_ps(r5);
          __m256 _row6 = _mm256_loadu_ps(r6);
          __m256 _row7 = _mm256_loadu_ps(r7);
          transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
          _mm256_storeu_ps(output_data, _row0);
          _mm256_storeu_ps(output_data + 8, _row1);
          _mm256_storeu_ps(output_data + 16, _row2);
          _mm256_storeu_ps(output_data + 24, _row3);
          _mm256_storeu_ps(output_data + 32, _row4);
          _mm256_storeu_ps(output_data + 40, _row5);
          _mm256_storeu_ps(output_data + 48, _row6);
          _mm256_storeu_ps(output_data + 56, _row7);
          r0 += 8;
          r1 += 8;
          r2 += 8;
          r3 += 8;
          r4 += 8;
          r5 += 8;
          r6 += 8;
          r7 += 8;
          output_data += 64;
        }

        for (; pos < input_width; ++pos) {
          output_data[0] = *r0++;
          output_data[1] = *r1++;
          output_data[2] = *r2++;
          output_data[3] = *r3++;
          output_data[4] = *r4++;
          output_data[5] = *r5++;
          output_data[6] = *r6++;
          output_data[7] = *r7++;
          output_data += 8;
        }

        for (int x = 0; x < right; ++x) {
          _mm256_storeu_ps(output_data, pad_val);
          output_data += 8;
        }
      }
      // fill bottom
      for (int y = 0; y < bottom_size; ++y) {
        _mm256_storeu_ps(output_data, pad_val);
        output_data += 8;
      }
    }
  }
}

__m256 activation8_m256(__m256 input, const lite_api::ActivationType act_type) {
  if (act_type == lite_api::ActivationType::kRelu) {
    return _mm256_max_ps(input, _mm256_setzero_ps());
  } else if (act_type == lite_api::ActivationType::kRelu6) {
    __m256 _val = _mm256_max_ps(input, _mm256_setzero_ps());
    return _mm256_max_ps(_val, _mm256_set1_ps(6.f));
  } else {
    LOG(FATAL) << "[X86] activation type not supported";
  }
  return _mm256_setzero_ps();
}

__m128 activation4_m128(__m128 input, const lite_api::ActivationType act_type) {
  if (act_type == lite_api::ActivationType::kRelu) {
    return _mm_max_ps(input, _mm_setzero_ps());
  } else if (act_type == lite_api::ActivationType::kRelu6) {
    __m128 _val = _mm_max_ps(input, _mm_setzero_ps());
    return _mm_max_ps(_val, _mm_set1_ps(6.f));
  } else {
    LOG(FATAL) << "[X86] activation type not supported";
  }
  return _mm_setzero_ps();
}

float activation1_float(float input, const lite_api::ActivationType act_type) {
  if (act_type == lite_api::ActivationType::kRelu) {
    return (std::max)(input, 0.f);
  } else if (act_type == lite_api::ActivationType::kRelu6) {
    return (std::max)((std::max)(input, 0.f), 6.0f);
  } else {
    LOG(FATAL) << "[X86] activation type not supported";
  }
  return 0.f;
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
