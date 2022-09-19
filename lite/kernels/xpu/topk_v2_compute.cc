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

#include "lite/kernels/xpu/topk_v2_compute.h"
#include "lite/backends/xpu/target_wrapper.h"  // XPUScratchPadGuard
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void TopkV2Compute::Run() {
  auto& param = this->Param<operators::TopkParam>();
  auto& ctx = this->ctx_->As<XPUContext>();

  DDim x_dims = param.X->dims();
  int axis = param.axis;
  CHECK_EQ(axis, -1);
  int dim_size = x_dims.size();
  if (axis < 0) {
    axis += dim_size;
  }

  int k = param.K;
  if (param.k_is_tensor) {
    k = param.KTensor->data<int>()[0];
  }

  int m = x_dims.count(0, axis);
  int n = x_dims[axis];

  XPUScratchPadGuard indices_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(m * k * sizeof(int));

  int* indices_int32_device = reinterpret_cast<int*>(indices_xpu_guard_->addr_);
  int64_t* indices_int64_device =
      param.Indices->mutable_data<int64_t>(TARGET(kXPU));

  int r = xdnn::sorted_topk(ctx.GetRawContext(),
                            param.X->data<float>(),
                            param.Out->mutable_data<float>(TARGET(kXPU)),
                            indices_int32_device,
                            m,
                            n,
                            k);
  CHECK_EQ(r, 0);

  r = xdnn::cast_v2<int, int64_t>(
      ctx.GetRawContext(), indices_int32_device, indices_int64_device, m * k);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(top_k_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::TopkV2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("K", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
