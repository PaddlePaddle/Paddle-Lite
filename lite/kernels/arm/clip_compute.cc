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

#include "lite/kernels/arm/clip_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ClipCompute::Run() {
  auto& param = Param<operators::ClipParam>();
  lite::Tensor* x = param.x;
  lite::Tensor* min_tensor = param.min_tensor;
  lite::Tensor* max_tensor = param.max_tensor;
  lite::Tensor* out = param.out;
  float min = param.min;
  float max = param.max;

  if (min_tensor != nullptr) {
    min = min_tensor->data<float>()[0];
  }
  if (max_tensor != nullptr) {
    max = max_tensor->data<float>()[0];
  }

  const float* x_ptr = x->data<float>();
  float* out_ptr = out->mutable_data<float>();
  int64_t num = x->numel();
  lite::arm::math::clip_kernel_fp32(x_ptr, num, min, max, out_ptr);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    clip, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ClipCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("clip", 1)
    .Finalize();
