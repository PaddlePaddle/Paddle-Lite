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

#include "lite/kernels/xpu/topk_compute.h"
#include "lite/backends/xpu/target_wrapper.h"  // XPUScratchPadGuard
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void TopkCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  DDim x_dims = param.X->dims();
  int K = param.K;
  int dim_size = x_dims.size();
  int m = x_dims.production() / x_dims[dim_size - 1];
  int n = x_dims[dim_size - 1];

  XPUScratchPadGuard indices_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(m * K * sizeof(int));

  int* indices_int32_device = reinterpret_cast<int*>(indices_xpu_guard_->addr_);
  int64_t* indices_int64_device =
      param.Indices->mutable_data<int64_t>(TARGET(kXPU));

  int r = xdnn::sorted_topk(ctx.GetRawContext(),
                            param.X->data<float>(),
                            param.Out->mutable_data<float>(TARGET(kXPU)),
                            indices_int32_device,
                            m,
                            n,
                            K);
  CHECK_EQ(r, 0);

  r = xdnn::cast_v2<int, int64_t>(
      ctx.GetRawContext(), indices_int32_device, indices_int64_device, m * K);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    top_k, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::TopkCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
