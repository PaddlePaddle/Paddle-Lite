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

#include "lite/kernels/arm/normalize_compute.h"
#include <string>
#include <vector>
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
/*void NormalizeCompute::PrepareForRun() {
  LOG(INFO) << "Normalize prepare to run";
  auto& ctx = this->ctx_->template As<ARMContext>();
}*/

void NormalizeCompute::Run() {
  // 1、读入数据
  LOG(INFO) << "into normalize";
  auto& param = this->Param<param_t>();
  //  auto& ctx = this->ctx_->template As<ARMContext>();
  /////////////// 1给变量malloc空间
  auto x_dims = param.X->dims();
  this->_mean->Resize({1, 1, 1, x_dims[0] * x_dims[1]});
  this->_variance->Resize({1, 1, 1, x_dims[0] * x_dims[1]});
  //  param.Out.push_back(_mean);
  //  param.Out.push_back(_variance);
  /////////////////////////////////////
  const float* input = param.X->data<float>();
  float* out = param.Out->mutable_data<float>();
  const float* mean_data = this->_mean->data<float>();
  const float* variance_data = this->_variance->data<float>();
  auto input_dims = param.X->dims();
  int num = input_dims[0];
  int channel = input_dims[1];
  int height = input_dims[2];
  int width = input_dims[3];
  int spatial_size = width * height;
  int cnt = spatial_size / 8;
  LOG(INFO) << "into compute mean";
  lite::arm::math::compute_mean(
      input, this->_mean, num, channel, height, width);
  LOG(INFO) << "into cmopute variance;";
  lite::arm::math::compute_variance(
      input, this->_mean, this->_variance, num, channel, height, width);
  LOG(INFO) << "into for";
  for (int n = 0; n < num; ++n) {
    const float* input_batch = input + n * spatial_size * channel;
    float* output_batch = out + n * spatial_size * channel;
#pragma omp parallel for
    for (int c = 0; c < channel; ++c) {
      const float* input_channel = input_batch + c * spatial_size;
      float* output_channel = output_batch + c * spatial_size;
      int i = 0;
      float mean_val = mean_data[c];
      float std_val = 1.f / sqrt(variance_data[c] + param.eps);
      float32x4_t vmean_val = vdupq_n_f32(mean_val);
      float32x4_t vstd_val = vdupq_n_f32(std_val);
#ifdef __aarch64__
      for (; i < cnt; ++i) {
        float32x4_t in_data0 = vld1q_f32(input_channel);
        in_data0 = vsubq_f32(in_data0, vmean_val);
        in_data0 = vmulq_f32(in_data0, vstd_val);
        vst1q_f32(output_channel, in_data0);

        float32x4_t in_data1 = vld1q_f32(input_channel + 4);
        in_data1 = vsubq_f32(in_data1, vmean_val);
        in_data1 = vmulq_f32(in_data1, vstd_val);
        vst1q_f32(output_channel + 4, in_data1);

        input_channel += 8;
        output_channel += 8;
      }
#else
      int loop = cnt;
      if (loop > 0) {
        asm volatile(
            "1:                                       \n"
            "vld1.f32   {d0-d1},[%[in_channel]]!      \n"
            "vsub.f32   q1, q0, %q[mean]              \n"
            "vmul.f32   q2, q1, %q[std]               \n"
            "vst1.32    {d4-d5}, [%[out_channel]]!    \n"

            "vld1.f32   {d6-d7}, [%[in_channel]]!     \n"
            "vsub.f32   q4, q3, %q[mean]              \n"
            "vmul.f32   q5, q4, %q[std]               \n"
            "vst1.32    {d10-d11}, [%[out_channel]]!  \n"
            "subs       %[loop], #1                   \n"
            "bne        1b                            \n"
            : [in_channel] "+r"(input_channel),
              [out_channel] "+r"(output_channel),
              [loop] "+r"(loop),
              [mean] "+w"(vmean_val),
              [std] "+w"(vstd_val)
            : "r"(input_channel), "r"(loop)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
      }
#endif  // __aarch64__
      for (i = cnt * 8; i < spatial_size; ++i) {
        float in_data = input_channel[0];
        in_data = (in_data - mean_val) * std_val;
        output_channel[0] = in_data;
        input_channel++;
        output_channel++;
      }
      LOG(INFO) << "out of for";
    }
  }
  LOG(INFO) << "get of normalize:";
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(normalize,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NormalizeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    // .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    // .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
