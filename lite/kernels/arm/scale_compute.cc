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

#include "lite/kernels/arm/scale_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T, PrecisionType PType>
void ScaleCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::ScaleParam>();
  int num = param.x->numel();
  const T* x_data = param.x->template data<T>();
  T* output_data = param.output->template mutable_data<T>();
  T scale = static_cast<T>(param.scale);
  T bias = static_cast<T>(param.bias);
  if (!param.bias_after_scale) {
    bias *= scale;
  }
  T alpha = param.alpha;
  if (param.activation_type == "") {  // no act
    lite::arm::math::scale<T>(x_data, output_data, num, scale, bias);
  } else if (param.activation_type == "relu") {  // do relu
    lite::arm::math::scale_relu<T>(x_data, output_data, num, scale, bias);
  } else if (param.activation_type == "relu6") {  // do relu6
    lite::arm::math::scale_relu6<T>(
        x_data, output_data, num, scale, bias, alpha);
  } else if (param.activation_type == "leaky_relu") {  // do leaky_relu
    lite::arm::math::scale_leaky_relu<T>(
        x_data, output_data, num, scale, bias, alpha);
  }
  if (!param.x->lod().empty()) {
    param.output->set_lod(param.x->lod());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using scale_float =
    paddle::lite::kernels::arm::ScaleCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

using scale_int32 =
    paddle::lite::kernels::arm::ScaleCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(scale, kARM, kInt32, kNCHW, scale_int32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
