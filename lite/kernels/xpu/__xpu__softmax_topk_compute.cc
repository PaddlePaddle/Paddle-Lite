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

#include "lite/kernels/xpu/__xpu__softmax_topk_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SoftmaxTopkCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  indices_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(param.indices->numel() * sizeof(int));
}

void SoftmaxTopkCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int K = param.K;
  std::vector<int> xdims;
  for (auto i = 0; i < param.x->dims().size(); i++) {
    xdims.push_back(param.x->dims().data()[i]);
  }
  int axis = param.axis < 0 ? param.axis + xdims.size() : param.axis;
  indices_xpu_guard_->Reserve(param.indices->numel() * sizeof(int));
  int* indices_int32_device = reinterpret_cast<int*>(indices_xpu_guard_->addr_);
  int64_t* indices_int64_device =
      param.indices->mutable_data<int64_t>(TARGET(kXPU));
  int r = xdnn::sorted_softmax_topk<float, int>(
      ctx.GetRawContext(),
      param.x->data<float>(),
      param.output->mutable_data<float>(TARGET(kXPU)),
      indices_int32_device,
      xdims,
      axis,
      K);
  CHECK_EQ(r, 0);
  r = xdnn::cast_v2<int, int64_t>(ctx.GetRawContext(),
                                  indices_int32_device,
                                  indices_int64_device,
                                  param.indices->numel());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__softmax_topk,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SoftmaxTopkCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
