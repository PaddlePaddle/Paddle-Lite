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
#include "lite/kernels/arm/gather_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void GatherCompute::Run() {
  auto& param = this->Param<operators::GatherParam>();

  auto* p_output = param.Out->mutable_data<float>();
  auto index_size = param.Index->dims()[0];
  auto src_dims = param.X->dims();
  const float* p_src = param.X->data<float>();
  const int* p_index = param.Index->data<int>();

  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  for (int i = 0; i < index_size; ++i) {
    int index_ = p_index[i];
    memcpy(p_output + i * slice_size,
           p_src + index_ * slice_size,
           slice_size * sizeof(float));
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    gather, kARM, kAny, kNCHW, paddle::lite::kernels::arm::GatherCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
