// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/lod_reset_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void LodResetCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x = param.X;
  auto output = param.Out;
  output->mutable_data(TARGET(kXPU), x->memory_size());
  int r = xdnn::copy<int8_t>(ctx.GetRawContext(),
                             x->data<int8_t>(),
                             reinterpret_cast<int8_t*>(output->raw_data()),
                             x->memory_size());
  CHECK_EQ(r, 0);
  auto lod = output->mutable_lod();
  if (param.Y) {
    if (param.Y->lod().size()) {
      *lod = param.Y->lod();
    } else {
      const auto* y_data = param.Y->data<int>();
      std::vector<int> y_cpu(param.Y->numel());
      TargetWrapperXPU::MemcpySync(y_cpu.data(),
                                   y_data,
                                   param.Y->numel() * sizeof(int),
                                   IoDirection::DtoH);
      (*lod).resize(1);
      (*lod)[0].resize(param.Y->numel());
      for (int i = 0; i < param.Y->numel(); i++) {
        (*lod)[0][i] = y_cpu[i];
      }
    }
  } else {
    (*lod).resize(1);
    for (auto id : param.target_lod) {
      (*lod)[0].push_back(id);
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lod_reset,
                     kXPU,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::xpu::LodResetCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
