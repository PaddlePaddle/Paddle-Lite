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

#include "lite/kernels/xpu/sum_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SumCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  int N = param.x.size();
  if (N == 1) {
    param.output->ShareDataWith(*param.x[0]);
    return;
  }
  std::vector<const float*> ptrs(N, nullptr);
  for (int i = 0; i < N; i++) {
    ptrs[i] = param.x[i]->data<float>();
  }
  int out_numel = param.output->numel();
  int r = xdnn::sum_batch(ctx.GetRawContext(),
                          ptrs.data(),
                          param.output->mutable_data<float>(TARGET(kXPU)),
                          N,
                          out_numel);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    sum, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SumCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
