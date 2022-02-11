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

#include "lite/kernels/xpu/cvm_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void CvmCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::cvm<float>(ctx.GetRawContext(),
                           param.X->data<float>(),
                           param.Y->mutable_data<float>(TARGET(kXPU)),
                           param.use_cvm,
                           param.X->dims()[0],
                           param.X->dims()[1]);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cvm, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::CvmCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("CVM", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
