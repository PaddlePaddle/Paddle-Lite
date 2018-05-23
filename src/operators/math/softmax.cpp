/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "operators/math/softmax.h"
#include "common/types.h"
#if __ARM_NEON
#include <algorithm>
#include "operators/math/math_func_neon.h"
#endif

namespace paddle_mobile {
namespace operators {
namespace math {
using framework::DDim;
using framework::Tensor;
template <typename T>
class SoftmaxFuntor<CPU, T> {
#if __ARM_NEON
  void sum(float *input, float *sumptr, int inner_size, int outter_size) {
    float32x4_t acc = vdupq_n_f32(0);
    float sum_ = 0;
    for (int i = 0; i < outter_size; ++i) {
      float *input_outer_ptr = input + i * inner_size;
      int nn = inner_size >> 2;
      int left = inner_size - (nn << 2);
      for (; nn > 0; nn--) {
        float32x4_t vec_input = vld1q_f32(input_outer_ptr);
        acc = vaddq_f32(acc, vec_input);
        input_outer_ptr += 4;
      }
      float32x2_t vsum_ = vadd_f32(vget_high_f32(acc), vget_low_f32(acc));
      sum_ = vget_lane_f32(vsum_, 0) + vget_lane_f32(vsum_, 1);
      for (; left > 0; left--) {
        sum_ += *input_outer_ptr;
        input_outer_ptr++;
      }
    }
    for (int j = 0; j < inner_size * outter_size; ++j) {
      sumptr[j] = sum_;
    }
  }

  void SoftmaxCacl(const Tensor *X, Tensor *Y) {
    const float *input = X->data<float>();
    const DDim &dDim = X->dims();
    int axis_index = 1;
    if (dDim.size() < 4) {
      axis_index = 0;
    }
    DDim outer_ddim =
        paddle_mobile::framework::slice_ddim(dDim, 0, axis_index + 1);
    DDim inner_ddim =
        paddle_mobile::framework::slice_ddim(dDim, axis_index + 1, dDim.size());
    int out_size = paddle_mobile::framework::product(outer_ddim);
    int inner_size = paddle_mobile::framework::product(inner_ddim);
    auto *max_ptr = new float[inner_size * out_size];
    // max
    for (int j = 0; j < out_size; ++j) {
      const float *input_outer_ptr = input + j * inner_size;
      float *max_outer_ptr = max_ptr + j * inner_size;
      float max_ = 0;
      for (int i = 0; i < inner_size; ++i) {
        const float *input_inner_ptr = input_outer_ptr + i;
        max_ = std::max(max_, input_inner_ptr[0]);
      }
      for (int k = 0; k < inner_size; ++k) {
        max_outer_ptr[k] = max_;
      }
    }
    // exp(value - max)
    float *exp_sub_max = new float[inner_size * out_size];
    float *exp_sub_max_ptr = &exp_sub_max[0];
    for (int l = 0; l < out_size; ++l) {
      const float *input_outer_ptr = input + l * inner_size;
      float *max_outer_ptr = max_ptr + l * inner_size;
      int nn = inner_size >> 2;
      int left = inner_size - (nn << 2);
      for (; nn > 0; nn--) {
        float32x4_t vec_input = vld1q_f32(input_outer_ptr);
        float32x4_t vec_max = vld1q_f32(max_outer_ptr);
        float32x4_t vec_sub = vsubq_f32(vec_input, vec_max);
        float32x4_t vec_exp = exp_ps(vec_sub);
        vst1q_f32(exp_sub_max_ptr, vec_exp);
        input_outer_ptr += 4;
        max_outer_ptr += 4;
        exp_sub_max_ptr += 4;
      }
      for (; left > 0; left--) {
        *exp_sub_max_ptr = expf(*input_outer_ptr - *max_outer_ptr);

        input_outer_ptr++;
        max_outer_ptr++;
        exp_sub_max_ptr++;
      }
    }
    float *sumptr = new float[inner_size * out_size];
    // sum exp
    sum(exp_sub_max, sumptr, inner_size, out_size);
    // div
    auto *out_ptr = static_cast<float *>(Y->mutable_data());
    for (int l = 0; l < out_size; ++l) {
      const float *input_outer_ptr = exp_sub_max + l * inner_size;
      float *output_outer_ptr = out_ptr + l * inner_size;
      float *sum_outer_ptr = sumptr + l * inner_size;
      int nn = inner_size >> 2;
      int left = inner_size - (nn << 2);
      for (; nn > 0; nn--) {
        float32x4_t vec_input = vld1q_f32(input_outer_ptr);
        float32x4_t vec_sum = vld1q_f32(sum_outer_ptr);
        float32x4_t vec_div = div_ps(vec_input, vec_sum);
        vst1q_f32(output_outer_ptr, vec_div);
        input_outer_ptr += 4;
        output_outer_ptr += 4;
        sum_outer_ptr += 4;
      }
      for (; left > 0; left--) {
        *output_outer_ptr = (*input_outer_ptr) / (*sum_outer_ptr);
        input_outer_ptr++;
        output_outer_ptr++;
        sum_outer_ptr++;
      }
    }
  }
#endif  // ARM_NEON

 public:
  void operator()(const framework::Tensor *X, framework::Tensor *Y) {
#if __ARM_NEON
    SoftmaxCacl(X, Y);
#endif
  }
};

template class SoftmaxFuntor<CPU, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
