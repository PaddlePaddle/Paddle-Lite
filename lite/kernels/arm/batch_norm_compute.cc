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

#include "lite/kernels/arm/batch_norm_compute.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T, PrecisionType PType>
void BatchNormCompute<T, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto x_dims = param.x->dims();
  bool global_stats = param.is_test || param.use_global_stats;
  if (global_stats) {
    int64_t channel_size = 0;
    switch (param.data_layout) {
      case DATALAYOUT(kNCHW):
        channel_size = x_dims[1];
        break;
      default:
        LOG(FATAL) << "Unknown storage order: "
                   << DataLayoutToStr(param.data_layout);
        break;
    }
    new_scale.Resize({channel_size});
    new_bias.Resize({channel_size});
    auto* scale_data = param.scale->template data<float>();
    auto* bias_data = param.bias->template data<float>();
    auto* mean_data = param.mean->template data<float>();
    auto* variance_data = param.variance->template data<float>();
    auto* new_scale_data = new_scale.mutable_data<T>();
    auto* new_bias_data = new_bias.mutable_data<T>();
    for (int c = 0; c < channel_size; c++) {
      float inv_scale = 1.f / (std::sqrt(variance_data[c] + param.epsilon));
      new_bias_data[c] =
          bias_data[c] - inv_scale * scale_data[c] * mean_data[c];
      new_scale_data[c] = inv_scale * scale_data[c];
    }
  }
}

template <typename T, PrecisionType PType>
void BatchNormCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto x_dims = param.x->dims();
  auto x_data = param.x->template data<T>();
  auto y_data = param.y->template mutable_data<T>();
  bool global_stats = param.is_test || param.use_global_stats;
  if (global_stats) {
    auto* new_scale_data = new_scale.data<T>();
    auto* new_bias_data = new_bias.data<T>();
    int64_t outer_size = 0;
    int64_t channel_size = 0;
    int64_t inner_size = 0;
    switch (param.data_layout) {
      case DATALAYOUT(kNCHW):
        outer_size = x_dims[0];
        channel_size = x_dims[1];
        inner_size = x_dims.Slice(2, x_dims.size()).production();
        lite::arm::math::scale(x_data,
                               y_data,
                               outer_size,
                               channel_size,
                               inner_size,
                               new_scale_data,
                               new_bias_data);
        break;
      default:
        LOG(FATAL) << "Unknown storage order: "
                   << DataLayoutToStr(param.data_layout);
        break;
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::BatchNormCompute<float16_t,
                                                     PRECISION(kFP16)>
    BnFp16;
REGISTER_LITE_KERNEL(batch_norm, kARM, kFP16, kNCHW, BnFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ReserveSpace", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

typedef paddle::lite::kernels::arm::BatchNormCompute<float, PRECISION(kFloat)>
    BnFp32;
REGISTER_LITE_KERNEL(batch_norm, kARM, kFloat, kNCHW, BnFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ReserveSpace", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(sync_batch_norm, kARM, kFloat, kNCHW, BnFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ReserveSpace", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
