// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/cos_compute.h"
#include <algorithm>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

// void cos_func(const lite::Tensor *input,
//                 lite::Tensor *output) {
//     auto input_ddim = input->dims();
//     int num = 1;
//     for (int i = 0; i < input_ddim.size(); i++){
//         num *= input_ddim[i];
//     }

//     const float* inp_ptr = input->data<float>();
//     float* out_ptr = output->mutable_data<float>();
//     for (int n = 0; n < num; n++){
//         out_ptr[n] = cos(inp_ptr[n]);
//     }
// }

void CosCompute::Run() {
  auto& param = Param<operators::CosParam>();
  lite::Tensor* input = param.X;
  lite::Tensor* output = param.Out;
  lite::arm::math::cos_func(input, output);
#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "cos_func";
#endif
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cos, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::CosCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
