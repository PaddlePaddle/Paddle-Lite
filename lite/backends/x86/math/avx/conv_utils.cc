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

#include "lite/backends/x86/math/avx/conv_utils.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

// tranpose [chout, chin, wh, ww] to [chout/block,chin,wh,ww,block]
// dout space should be allocated before calling conv_trans_weights_numc
void conv_trans_weights_numc(const float* din,
                             float* dout,  // dout has been expanded
                             int chout,
                             int chin,
                             int wh,
                             int ww,
                             int block) {
  // dout is [chout_expand / block , chin, wh, ww, block]
  int chout_expand = (chout + block - 1) / block * block;
  memset(dout, 0.f, sizeof(float) * chout_expand * chin * wh * ww);

  const float* from_address = din;
  int wchwb = chin * wh * ww * block;
  int whwb = wh * ww * block;
  int wwb = ww * block;

  for (int wn_i = 0; wn_i < chout; wn_i++) {
    for (int wc_i = 0; wc_i < chin; wc_i++) {
      for (int wh_i = 0; wh_i < wh; wh_i++) {
        for (int ww_i = 0; ww_i < ww; ww_i++) {
          int dst_index = wn_i / block * wchwb + wc_i * whwb + wh_i * wwb +
                          ww_i * block + wn_i % block;
          dout[dst_index] = *from_address;
          from_address++;
        }
      }
    }
  }
}

// tranpose [chout,chin,wh,ww] to [chout/block,wh,ww,chin,block]
// this function is different from conv_trans_weights_numc just
// in that we make chw->hwc
void conv_trans_weights_numc_c3(const float* din,
                                float* dout,
                                int chout,
                                int chin,
                                int wh,
                                int ww,
                                int block) {
  CHECK_EQ(chin, 3);
  int chout_expand = (chout + block - 1) / block * block;
  memset(
      dout, 0, sizeof(float) * chout_expand / block * wh * ww * chin * block);

  const float* from_address = din;
  for (int wn_i = 0; wn_i < chout; wn_i++) {
    for (int wc_i = 0; wc_i < chin; wc_i++) {  // chin=3!
      for (int wh_i = 0; wh_i < wh; wh_i++) {
        for (int ww_i = 0; ww_i < ww; ww_i++) {
          int dst_index = wn_i / block * wh * ww * chin * block +
                          wh_i * ww * chin * block + ww_i * chin * block +
                          wc_i * block + wn_i % block;
          dout[dst_index] = *from_address;
          from_address++;
        }
      }
    }
  }
}

// function: input-4x8, output-8x4
static inline void transpose4x8_ps(__m256& row0,  // NOLINT
                                   __m256& row1,  // NOLINT
                                   __m256& row2,  // NOLINT
                                   __m256& row3   // NOLINT
                                   ) {
  // vtmp0=a0b0a1b1a4b4a5b5
  __m256 vtmp0 = _mm256_unpacklo_ps(row0, row1);
  // vtmp1=a2b2a3b3a6b6a7b7
  __m256 vtmp1 = _mm256_unpackhi_ps(row0, row1);
  // vtmp2=c0d0c1d1c4d4c5d5
  __m256 vtmp2 = _mm256_unpacklo_ps(row2, row3);
  // vtmp3=c2d2c3d3c6d6c7d7
  __m256 vtmp3 = _mm256_unpackhi_ps(row2, row3);
  // vres0=a0b0c0d0a4b4c4d4
  __m256 vres0 = _mm256_shuffle_ps(vtmp0, vtmp2, 0x44);  // 0xaa=[01,00,01,00]
  // vres1=a1b1c1d1a5b5c5d5
  __m256 vres1 = _mm256_shuffle_ps(vtmp0, vtmp2, 0xee);  // 0xaa=[11,10,11,10]
  // vres2=a2b2c2d2a6b6c6d6
  __m256 vres2 = _mm256_shuffle_ps(vtmp1, vtmp3, 0x44);  // 0xaa=[01,00,01,00]
  // vres3=a3b3c3d3a7b7c7d7
  __m256 vres3 = _mm256_shuffle_ps(vtmp1, vtmp3, 0xee);  // 0xaa=[11,10,11,10]
  // row0=a0b0c0d0a1b1c1d1
  row0 = _mm256_permute2f128_ps(vres0, vres1, 0x20);
  // row1=a2b2c2d2a3b3c3d3
  row1 = _mm256_permute2f128_ps(vres2, vres3, 0x20);
  // row2=a4b4c4d4a5b5c5d5
  row2 = _mm256_permute2f128_ps(vres0, vres1, 0x31);
  // row3=a6b6c6d6a7b7c7d7
  row3 = _mm256_permute2f128_ps(vres2, vres3, 0x31);
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

// input  [bs, ic, ih, iw] => [bs, (ic + 7)/8, ih, iw, 8]
// filter [oc, 01, ih, iw] => [01, (ic + 7)/8, ih, iw, 8] for depthwise
void packC8_common(const float* din,
                   float* dout,
                   const std::vector<int>& pad,
                   int h_in,
                   int w_in,
                   int channel) {
  int top = pad[0];
  int bottom = pad[1];
  int left = pad[2];
  int right = pad[3];
  int w_out = (w_in + left + right);
  int h_out = (h_in + top + bottom);
  int block_channel = 8;
  const float* din_init = din;
  float* dout_init = dout;

  for (int c = 0; c < channel; c += block_channel) {
    din = din_init + c * h_in * w_in;
    dout = dout_init + c * w_out * h_out;

    memset(dout, 0, top * w_out * block_channel * sizeof(float));
    auto dout_block = dout + top * w_out * block_channel;

    for (int i = 0; i < h_in; i++) {
      float* douth = dout_block + i * w_out * block_channel;
      const float* dinh = din + i * w_in;
      memset(douth, 0, left * block_channel * sizeof(float));
      douth += left * block_channel;
      int kernel_size = h_in * w_in;
      auto dinr0 = dinh;
      auto dinr1 = dinr0 + kernel_size;
      auto dinr2 = dinr1 + kernel_size;
      auto dinr3 = dinr2 + kernel_size;
      auto dinr4 = dinr3 + kernel_size;
      auto dinr5 = dinr4 + kernel_size;
      auto dinr6 = dinr5 + kernel_size;
      auto dinr7 = dinr6 + kernel_size;

      int j = 0;
      if (c + 7 < channel) {
        for (; j + 7 < w_in; j += 8) {
          __m256 _row0 = _mm256_loadu_ps(dinr0);
          __m256 _row1 = _mm256_loadu_ps(dinr1);
          __m256 _row2 = _mm256_loadu_ps(dinr2);
          __m256 _row3 = _mm256_loadu_ps(dinr3);
          __m256 _row4 = _mm256_loadu_ps(dinr4);
          __m256 _row5 = _mm256_loadu_ps(dinr5);
          __m256 _row6 = _mm256_loadu_ps(dinr6);
          __m256 _row7 = _mm256_loadu_ps(dinr7);
          transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
          _mm256_storeu_ps(douth, _row0);
          _mm256_storeu_ps(douth + 8, _row1);
          _mm256_storeu_ps(douth + 16, _row2);
          _mm256_storeu_ps(douth + 24, _row3);
          _mm256_storeu_ps(douth + 32, _row4);
          _mm256_storeu_ps(douth + 40, _row5);
          _mm256_storeu_ps(douth + 48, _row6);
          _mm256_storeu_ps(douth + 56, _row7);
          dinr0 += 8;
          dinr1 += 8;
          dinr2 += 8;
          dinr3 += 8;
          dinr4 += 8;
          dinr5 += 8;
          dinr6 += 8;
          dinr7 += 8;
          douth += 64;
        }

        for (; j < w_in; j++) {
          douth[0] = *dinr0++;
          douth[1] = *dinr1++;
          douth[2] = *dinr2++;
          douth[3] = *dinr3++;
          douth[4] = *dinr4++;
          douth[5] = *dinr5++;
          douth[6] = *dinr6++;
          douth[7] = *dinr7++;
          douth += 8;
        }
      } else {
        __m256 _row0 = _mm256_setzero_ps();
        __m256 _row1 = _mm256_setzero_ps();
        __m256 _row2 = _mm256_setzero_ps();
        __m256 _row3 = _mm256_setzero_ps();
        __m256 _row4 = _mm256_setzero_ps();
        __m256 _row5 = _mm256_setzero_ps();
        __m256 _row6 = _mm256_setzero_ps();
        __m256 _row7 = _mm256_setzero_ps();
        for (; j + 7 < w_in; j += 8) {
          _row0 = _mm256_loadu_ps(dinr0);
          if (channel - c > 1) _row1 = _mm256_loadu_ps(dinr1);
          if (channel - c > 2) _row2 = _mm256_loadu_ps(dinr2);
          if (channel - c > 3) _row3 = _mm256_loadu_ps(dinr3);
          if (channel - c > 4) _row4 = _mm256_loadu_ps(dinr4);
          if (channel - c > 5) _row5 = _mm256_loadu_ps(dinr5);
          if (channel - c > 6) _row6 = _mm256_loadu_ps(dinr6);
          transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
          _mm256_storeu_ps(douth, _row0);
          _mm256_storeu_ps(douth + 8, _row1);
          _mm256_storeu_ps(douth + 16, _row2);
          _mm256_storeu_ps(douth + 24, _row3);
          _mm256_storeu_ps(douth + 32, _row4);
          _mm256_storeu_ps(douth + 40, _row5);
          _mm256_storeu_ps(douth + 48, _row6);
          _mm256_storeu_ps(douth + 56, _row7);
          dinr0 += 8;
          dinr1 += 8;
          dinr2 += 8;
          dinr3 += 8;
          dinr4 += 8;
          dinr5 += 8;
          dinr6 += 8;
          dinr7 += 8;
          douth += 64;
        }

        for (; j < w_in; j++) {
          douth[0] = *dinr0++;
          douth[1] = channel - c > 1 ? *dinr1++ : 0;
          douth[2] = channel - c > 2 ? *dinr2++ : 0;
          douth[3] = channel - c > 3 ? *dinr3++ : 0;
          douth[4] = channel - c > 4 ? *dinr4++ : 0;
          douth[5] = channel - c > 5 ? *dinr5++ : 0;
          douth[6] = channel - c > 6 ? *dinr6++ : 0;
          douth[7] = 0;
          douth += 8;
        }
      }
      memset(douth, 0, right * block_channel * sizeof(float));
    }
    memset(dout + (h_in + top) * w_out * block_channel,
           0,
           bottom * w_out * block_channel * sizeof(float));
  }
}

// output_trans [bs, (oc + 7)/8, oh, ow, 8] => output [bs, oc, oh, ow]
void unpackC8_common(const float* din,
                     float* dout,
                     int size_out_channel,
                     int channel) {
  int block_channel = 8;
  float* dout_init = dout;

  for (int c = 0; c < channel; c += block_channel) {
    dout = dout_init + c * size_out_channel;
    auto doutr0 = dout;
    auto doutr1 = doutr0 + size_out_channel;
    auto doutr2 = doutr1 + size_out_channel;
    auto doutr3 = doutr2 + size_out_channel;
    auto doutr4 = doutr3 + size_out_channel;
    auto doutr5 = doutr4 + size_out_channel;
    auto doutr6 = doutr5 + size_out_channel;
    auto doutr7 = doutr6 + size_out_channel;
    int j = 0;
    if (c + 7 < channel) {
      for (; j + 7 < size_out_channel; j += 8) {
        __m256 _row0 = _mm256_loadu_ps(din);
        __m256 _row1 = _mm256_loadu_ps(din + 8);
        __m256 _row2 = _mm256_loadu_ps(din + 16);
        __m256 _row3 = _mm256_loadu_ps(din + 24);
        __m256 _row4 = _mm256_loadu_ps(din + 32);
        __m256 _row5 = _mm256_loadu_ps(din + 40);
        __m256 _row6 = _mm256_loadu_ps(din + 48);
        __m256 _row7 = _mm256_loadu_ps(din + 56);
        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
        _mm256_storeu_ps(doutr0, _row0);
        _mm256_storeu_ps(doutr1, _row1);
        _mm256_storeu_ps(doutr2, _row2);
        _mm256_storeu_ps(doutr3, _row3);
        _mm256_storeu_ps(doutr4, _row4);
        _mm256_storeu_ps(doutr5, _row5);
        _mm256_storeu_ps(doutr6, _row6);
        _mm256_storeu_ps(doutr7, _row7);
        doutr0 += 8;
        doutr1 += 8;
        doutr2 += 8;
        doutr3 += 8;
        doutr4 += 8;
        doutr5 += 8;
        doutr6 += 8;
        doutr7 += 8;
        din += 64;
      }

      for (; j < size_out_channel; j++) {
        *doutr0++ = *din++;
        *doutr1++ = *din++;
        *doutr2++ = *din++;
        *doutr3++ = *din++;
        *doutr4++ = *din++;
        *doutr5++ = *din++;
        *doutr6++ = *din++;
        *doutr7++ = *din++;
      }
    } else {
      for (; j + 7 < size_out_channel; j += 8) {
        __m256 _row0 = _mm256_loadu_ps(din);
        __m256 _row1 = _mm256_loadu_ps(din + 8);
        __m256 _row2 = _mm256_loadu_ps(din + 16);
        __m256 _row3 = _mm256_loadu_ps(din + 24);
        __m256 _row4 = _mm256_loadu_ps(din + 32);
        __m256 _row5 = _mm256_loadu_ps(din + 40);
        __m256 _row6 = _mm256_loadu_ps(din + 48);
        __m256 _row7 = _mm256_loadu_ps(din + 56);
        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
        _mm256_storeu_ps(doutr0, _row0);
        if (channel - c > 1) _mm256_storeu_ps(doutr1, _row1);
        if (channel - c > 2) _mm256_storeu_ps(doutr2, _row2);
        if (channel - c > 3) _mm256_storeu_ps(doutr3, _row3);
        if (channel - c > 4) _mm256_storeu_ps(doutr4, _row4);
        if (channel - c > 5) _mm256_storeu_ps(doutr5, _row5);
        if (channel - c > 6) _mm256_storeu_ps(doutr6, _row6);
        doutr0 += 8;
        doutr1 += 8;
        doutr2 += 8;
        doutr3 += 8;
        doutr4 += 8;
        doutr5 += 8;
        doutr6 += 8;
        doutr7 += 8;
        din += 64;
      }

      for (; j < size_out_channel; j++) {
        *doutr0++ = *din;
        if (channel - c > 1) *doutr1++ = *(din + 1);
        if (channel - c > 2) *doutr2++ = *(din + 2);
        if (channel - c > 3) *doutr3++ = *(din + 3);
        if (channel - c > 4) *doutr4++ = *(din + 4);
        if (channel - c > 5) *doutr5++ = *(din + 5);
        if (channel - c > 6) *doutr6++ = *(din + 6);
        din += 8;
      }
    }
  }
}

__m256 activation8_m256(__m256 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param) {
  if (act_type == lite_api::ActivationType::kRelu) {
    return _mm256_max_ps(input, _mm256_setzero_ps());
  } else if (act_type == lite_api::ActivationType::kRelu6) {
    __m256 _val = _mm256_max_ps(input, _mm256_setzero_ps());
    return _mm256_min_ps(_val, _mm256_set1_ps(act_param.Relu_clipped_coef));
  } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
    __m256 _val_scale =
        _mm256_mul_ps(input, _mm256_set1_ps(act_param.Leaky_relu_alpha));
    return _mm256_blendv_ps(
        _val_scale,
        input,
        _mm256_cmp_ps(input, _mm256_setzero_ps(), _CMP_GT_OS));
  } else if (act_type == lite_api::ActivationType::kHardSwish) {
    __m256 _val_offset =
        _mm256_add_ps(input, _mm256_set1_ps(act_param.hard_swish_offset));
    __m256 _val_scale =
        _mm256_mul_ps(input, _mm256_set1_ps(1.0 / act_param.hard_swish_scale));
    __m256 _val =
        _mm256_min_ps(_mm256_set1_ps(act_param.hard_swish_threshold),
                      _mm256_max_ps(_val_offset, _mm256_setzero_ps()));
    return _mm256_mul_ps(_val, _val_scale);
  } else {
    LOG(FATAL) << "[X86] activation type not supported";
  }
  return _mm256_setzero_ps();
}

__m128 activation4_m128(__m128 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param) {
  if (act_type == lite_api::ActivationType::kRelu) {
    return _mm_max_ps(input, _mm_setzero_ps());
  } else if (act_type == lite_api::ActivationType::kRelu6) {
    __m128 _val = _mm_max_ps(input, _mm_setzero_ps());
    return _mm_min_ps(_val, _mm_set1_ps(act_param.Relu_clipped_coef));
  } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
    __m128 _val_scale =
        _mm_mul_ps(input, _mm_set1_ps(act_param.Leaky_relu_alpha));
    return _mm_blendv_ps(
        _val_scale, input, _mm_cmp_ps(input, _mm_setzero_ps(), _CMP_GT_OS));
  } else if (act_type == lite_api::ActivationType::kHardSwish) {
    __m128 _val_offset =
        _mm_add_ps(input, _mm_set1_ps(act_param.hard_swish_offset));
    __m128 _val_scale =
        _mm_mul_ps(input, _mm_set1_ps(1.0 / act_param.hard_swish_scale));
    __m128 _val = _mm_min_ps(_mm_set1_ps(act_param.hard_swish_threshold),
                             _mm_max_ps(_val_offset, _mm_setzero_ps()));
    return _mm_mul_ps(_val, _val_scale);
  } else {
    LOG(FATAL) << "[X86] activation type not supported";
  }
  return _mm_setzero_ps();
}

float activation1_float(float input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param) {
  if (act_type == lite_api::ActivationType::kRelu) {
    return (std::max)(input, 0.f);
  } else if (act_type == lite_api::ActivationType::kRelu6) {
    return (std::min)((std::max)(input, 0.f), act_param.Relu_clipped_coef);
  } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
    return input > 0.f ? input : input * act_param.Leaky_relu_alpha;
  } else if (act_type == lite_api::ActivationType::kHardSwish) {
    return ((std::min)(act_param.hard_swish_threshold,
                       (std::max)(0.f, input + act_param.hard_swish_offset)) *
            input / act_param.hard_swish_scale);
  } else {
    LOG(FATAL) << "[X86] activation type not supported";
  }
  return 0.f;
}

/**
 * \brief inline funcs used in im2col
 * @param a
 * @param b
 * @return
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/**
 * \brief normal im2col function for gemm conv
 * @tparam dtype
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <typename Dtype>
void im2col_common(const Dtype* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   Dtype* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
          stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_top + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_left + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <>
void im2col_s1<float>(const float* data_im,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int pad_top,
                      int pad_bottom,
                      int pad_left,
                      int pad_right,
                      int dilation_h,
                      int dilation_w,
                      float* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) + 1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) + 1;
  const int in_channel_size = height * width;
  const int out_channel_size = output_h * output_w;
  const unsigned int output_plane_size =
      output_h * output_w * kernel_h * kernel_w;
  size_t tmp_size = static_cast<size_t>(output_plane_size);
  size_t mem_size = tmp_size * channels * sizeof(float);
  memset(data_col, 0, mem_size);
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    unsigned int data_im_z = static_cast<unsigned int>(c * in_channel_size);
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * out_channel_size * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * out_channel_size;
        unsigned int data_col_z =
            static_cast<unsigned int>(data_col_z1 + data_col_z2 + data_col_z3);
        int oh_begin = std::max(((pad_top - h_offset)), 0);  // always >= 0
        int oh_end = std::min(((height + pad_bottom - h_offset)), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset)), 0);
        int ow_end = std::min(((width + pad_right - w_offset)), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ++ih) {
          int iw = ow_begin - pad_left + w_offset;
          int ow = ow_begin;
          unsigned int data_im_offset = data_im_z + ih * width;
          unsigned int data_col_offset = data_col_z + oh * output_w;
          const float* data_im_ptr = data_im + data_im_offset;
          float* data_col_ptr = data_col + data_col_offset;
#ifdef __AVX__
          for (; ow + 7 < ow_end; ow += 8, iw += 8) {
            __m256 vtmp = _mm256_loadu_ps(data_im_ptr + iw);
            _mm256_storeu_ps(data_col_ptr + ow, vtmp);
          }
#else
          for (; ow + 3 < ow_end; ow += 4, iw += 4) {
            __m128 vtmp = _mm_loadu_ps(data_im_ptr + iw);
            _mm_storeu_ps(data_col_ptr + ow, vtmp);
          }
#endif
          for (; ow < ow_end; ++ow, ++iw) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
}

template <>
void im2col_s2<float>(const float* data_im,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int pad_top,
                      int pad_bottom,
                      int pad_left,
                      int pad_right,
                      int dilation_h,
                      int dilation_w,
                      float* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) / 2 +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) / 2 +
      1;
  const int in_channel_size = height * width;
  const unsigned int output_plane_size =
      output_h * output_w * kernel_h * kernel_w;
  size_t tmp_size = static_cast<size_t>(output_plane_size);
  size_t mem_size = tmp_size * channels * sizeof(float);
  memset(data_col, 0, mem_size);
#pragma omp parallel for
  for (int c = 0; c < channels; c++) {
    unsigned int data_im_z = static_cast<unsigned int>(c * in_channel_size);
    int data_col_z1 = c * output_plane_size;
    for (int ky = 0, h_offset = 0; ky < kernel_h;
         ky++, h_offset += dilation_h) {
      int data_col_z2 = ky * output_h * output_w * kernel_w;
      for (int kx = 0, w_offset = 0; kx < kernel_w;
           kx++, w_offset += dilation_w) {
        int data_col_z3 = kx * output_h * output_w;
        unsigned int data_col_z =
            static_cast<unsigned int>(data_col_z1 + data_col_z2 + data_col_z3);
        int oh_begin = std::max(((pad_top - h_offset + 1) / 2), 0);
        int oh_end =
            std::min(((height + pad_bottom - h_offset + 1) / 2), output_h);
        oh_end = std::max(oh_begin, oh_end);
        int ow_begin = std::max(((pad_left - w_offset + 1) / 2), 0);
        int ow_end =
            std::min(((width + pad_right - w_offset + 1) / 2), output_w);
        ow_end = std::max(ow_begin, ow_end);
        int ih = oh_begin * 2 - pad_top + h_offset;
        for (int oh = oh_begin; oh < oh_end; ++oh, ih += 2) {
          int iw = ow_begin * 2 - pad_left + w_offset;
          int ow = ow_begin;
          unsigned int data_im_offset = data_im_z + ih * width;
          unsigned int data_col_offset = data_col_z + oh * output_w;
          const float* data_im_ptr = data_im + data_im_offset;
          float* data_col_ptr = data_col + data_col_offset;
          for (; ow + 3 < ow_end; ow += 4, iw += 8) {
            // a0a1a2a3
            __m128 vtmp0 = _mm_loadu_ps(data_im_ptr + iw);
            // a4a5a6a7
            __m128 vtmp1 = _mm_loadu_ps(data_im_ptr + iw + 4);
            // a0a2a4a6
            _mm_storeu_ps(data_col_ptr + ow,
                          _mm_shuffle_ps(vtmp0, vtmp1, 0x88));
          }
          for (; ow < ow_end; ++ow, iw += 2) {
            data_col[data_col_offset + ow] = data_im[data_im_offset + iw];
          }
        }
      }
    }
  }
}

/**
 * \brief normal im2col function for gemm conv
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <>
void im2col<float>(const float* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   float* data_col) {
  bool pads_equal = ((pad_top == pad_bottom) && (pad_left == pad_right));
  bool pads_all_equal = (pads_equal && pad_top == pad_left);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilation_h == 1) && (dilation_w == 1);
  bool kspd = pads_all_equal && ks_equal && no_dilation;
  if (kspd && stride_h == 1) {
    im2col_s1<float>(data_im,
                     channels,
                     height,
                     width,
                     kernel_h,
                     kernel_w,
                     pad_top,
                     pad_bottom,
                     pad_left,
                     pad_right,
                     dilation_h,
                     dilation_w,
                     data_col);
  } else if (kspd && stride_h == 2) {
    im2col_s2<float>(data_im,
                     channels,
                     height,
                     width,
                     kernel_h,
                     kernel_w,
                     pad_top,
                     pad_bottom,
                     pad_left,
                     pad_right,
                     dilation_h,
                     dilation_w,
                     data_col);
  } else {
    im2col_common<float>(data_im,
                         channels,
                         height,
                         width,
                         kernel_h,
                         kernel_w,
                         pad_top,
                         pad_bottom,
                         pad_left,
                         pad_right,
                         stride_h,
                         stride_w,
                         dilation_h,
                         dilation_w,
                         data_col);
  }
}

template <>
void im2col<int8_t>(const int8_t* data_im,
                    int channels,
                    int height,
                    int width,
                    int kernel_h,
                    int kernel_w,
                    int pad_top,
                    int pad_bottom,
                    int pad_left,
                    int pad_right,
                    int stride_h,
                    int stride_w,
                    int dilation_h,
                    int dilation_w,
                    int8_t* data_col) {
  const int output_h =
      (height + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
          stride_h +
      1;
  const int output_w =
      (width + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
          stride_w +
      1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_top + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_left + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
