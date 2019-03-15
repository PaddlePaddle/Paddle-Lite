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

#ifdef SEQUENCE_POOL_OP

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "common/types.h"
#include "framework/context.h"
#include "operators/kernel/sequence_kernels.h"
#include "operators/math/pooling.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif  // __ARM_NEON__

namespace paddle_mobile {
namespace operators {

template <PoolingType P = MAX, typename T = float>
void SequencePoolImpl(const framework::LoDTensor &input,
                      framework::LoDTensor *output) {
  const float *input_ptr = input.data<float>();
  float *output_ptr = output->mutable_data<float>();
  const auto &lod = input.lod()[0];
  int64_t width = input.numel() / input.dims()[0];

  #pragma omp parallel for
  // num_threads(framework::threads())
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float *in_ptr = input_ptr + lod[i] * width;
    float *out_ptr = output_ptr + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (width == 1) {
      float max = -std::numeric_limits<float>::max();
      int remain_h = height;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop = remain_h >> 2;
      remain_h = remain_h & 0x3;
      float32x4_t __max4 = math::vPoolInitq_f32<MAX>();
      for (int h = 0; h < loop; ++h) {
        float32x4_t r0 = vld1q_f32(in_ptr);
        __max4 = vmaxq_f32(__max4, r0);
        in_ptr += 4;
      }
      float32x2_t __max2 =
          vpmax_f32(vget_low_f32(__max4), vget_high_f32(__max4));
      __max2 = vpmax_f32(__max2, __max2);
      max = std::max(max, vget_lane_f32(__max2, 0));
#endif  // __ARM_NEON__
      for (int h = 0; h < remain_h; ++h) {
        max = std::max(max, in_ptr[h]);
      }
      *out_ptr = max;
    } else {
      memcpy(out_ptr, in_ptr, width * sizeof(float));
      in_ptr += width;
      int remain_h = height - 1;
      int remain_w_start = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      remain_w_start = width & 0xfffc;
#endif  // __ARM_NEON__
      for (int h = 0; h < remain_h; ++h) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        for (int w = 0; w < width; w += 4) {
          float32x4_t __in = vld1q_f32(in_ptr + w);
          float32x4_t __out = vld1q_f32(out_ptr + w);
          __out = vmaxq_f32(__out, __in);
          vst1q_f32(out_ptr + w, __out);
        }
#endif  // __ARM_NEON__
        for (int w = remain_w_start; w < width; ++w) {
          out_ptr[w] = std::max(out_ptr[w], in_ptr[w]);
        }
        in_ptr += width;
      }
    }
  }
}

template <>
void SequencePoolImpl<SUM, float>(const framework::LoDTensor &input,
                                  framework::LoDTensor *output) {
  const float *input_ptr = input.data<float>();
  float *output_ptr = output->mutable_data<float>();
  const auto &lod = input.lod()[0];
  int64_t width = input.numel() / input.dims()[0];

  #pragma omp parallel for
  // num_threads(framework::threads())
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float *in_ptr = input_ptr + lod[i] * width;
    float *out_ptr = output_ptr + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (width == 1) {
      float sum = 0.f;
      int remain_h = height;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop = remain_h >> 2;
      remain_h = remain_h & 0x3;
      float32x4_t __sum4 = vdupq_n_f32(0.f);
      for (int h = 0; h < loop; ++h) {
        float32x4_t r0 = vld1q_f32(in_ptr);
        __sum4 = vaddq_f32(__sum4, r0);
        in_ptr += 4;
      }
      float32x2_t __sum2 =
          vpadd_f32(vget_low_f32(__sum4), vget_high_f32(__sum4));
      sum += vget_lane_f32(__sum2, 0) + vget_lane_f32(__sum2, 1);
#endif  // __ARM_NEON__
      for (int h = 0; h < remain_h; ++h) {
        sum += in_ptr[h];
      }
      *out_ptr = sum;
    } else {
      memcpy(out_ptr, in_ptr, width * sizeof(float));
      in_ptr += width;
      int remain_h = height - 1;
      int remain_w_start = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      int loop_w = width >> 2;
      remain_w_start = width & 0xfffc;
#endif  // __ARM_NEON__
      for (int h = 0; h < remain_h; ++h) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        for (int w = 0; w < width - 3; w += 4) {
          float32x4_t __in = vld1q_f32(in_ptr + w);
          float32x4_t __out = vld1q_f32(out_ptr + w);
          __out = vaddq_f32(__out, __in);
          vst1q_f32(out_ptr + w, __out);
        }
#endif  // __ARM_NEON__
        for (int w = remain_w_start; w < width; ++w) {
          out_ptr[w] += in_ptr[w];
        }
        in_ptr += width;
      }
    }
  }
}

template <>
void SequencePoolImpl<FIRST, float>(const framework::LoDTensor &input,
                                    framework::LoDTensor *output) {
  const float *input_ptr = input.data<float>();
  float *output_ptr = output->mutable_data<float>();
  const auto &lod = input.lod()[0];
  int64_t width = input.numel() / input.dims()[0];

  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float *in_ptr = input_ptr + lod[i] * width;
    float *out_ptr = output_ptr + i * width;
    memcpy(out_ptr, in_ptr, width * sizeof(float));
  }
}

template <typename T>
class SequencePoolKernel<CPU, T>
    : public framework::OpKernelBase<CPU, SequencePoolParam<CPU>> {
 public:
  bool Init(SequencePoolParam<CPU> *param) { return true; }

  void Compute(const SequencePoolParam<CPU> &param) {
    const framework::LoDTensor *input = param.input_;
    framework::LoDTensor *output = param.output_;
    output->mutable_data<T>();
    const std::string pooling_type = param.pool_type_;

    if (param.pool_type_ == "MAX") {
      SequencePoolImpl<MAX, T>(*input, output);
    } else if (param.pool_type_ == "FIRST") {
      SequencePoolImpl<FIRST, T>(*input, output);
    } else if (param.pool_type_ == "SUM") {
      SequencePoolImpl<SUM, T>(*input, output);
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION(
          "pooling type `%s` has not been implemented.",
          param.pool_type_.c_str());
    }
  }
};

template class SequencePoolKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif  // SEQUENCE_POOL_OP
