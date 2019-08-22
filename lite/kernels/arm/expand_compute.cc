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

#include "lite/kernels/arm/expand_compute.h"
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ExpandCompute::Run() {
  auto& param = Param<operators::ExpandParam>();
  const auto* x = param.X;
  auto* out = param.Out;
  std::vector<int> expand_times = param.expand_times;

  const float* src = x->data<float>();
  float* dst = out->mutable_data<float>();

  int dims = expand_times.size();
  DDim in_shape = x->dims();

  int inner_num = 1;
  int i = dims - 1;
  int outer_num = in_shape.count(0, i);
  inner_num *= in_shape[i];
  for (int j = 0; j < outer_num; ++j) {
    for (int k = 0; k < expand_times[i]; ++k) {
      memcpy(dst + (j * expand_times[i] + k) * inner_num,
             src + j * inner_num,
             sizeof(float) * inner_num);
    }
  }
  inner_num *= expand_times[i];
  for (int i = dims - 2; i >= 0; --i) {
    int outer_num = in_shape.count(0, i);
    inner_num *= in_shape[i];
    for (int j = outer_num - 1; j >= 0; --j) {
      for (int k = expand_times[i] - 1; k >= 0; --k) {
        memcpy(dst + (j * expand_times[i] + k) * inner_num,
               dst + j * inner_num,
               sizeof(float) * inner_num);
      }
    }
    inner_num *= expand_times[i];
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    expand, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ExpandCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
