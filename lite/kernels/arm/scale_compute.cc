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

template <typename T>
void scale_with_dtype(const operators::ScaleParam& param) {
  int num = param.x->numel();
  const T* x_data = param.x->data<T>();
  T* output_data = param.output->mutable_data<T>();
  T scale = static_cast<T>(param.scale);
  T bias = static_cast<T>(param.bias);
  if (!param.bias_after_scale) {
    bias *= scale;
  }
  lite::arm::math::scale(x_data, output_data, num, scale, bias);
}

void ScaleCompute::Run() {
  auto& param = Param<operators::ScaleParam>();
  auto x_precision = param.x->precision();
  switch (x_precision) {
    case PRECISION(kFloat):
      scale_with_dtype<float>(param);
      break;
    case PRECISION(kInt32):
      scale_with_dtype<int>(param);
      break;
    default:
      LOG(FATAL) << "unsupported input dtype: " << PrecisionToStr(x_precision);
      break;
  }
  if (!param.x->lod().empty()) {
    param.output->set_lod(param.x->lod());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    scale, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ScaleCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
