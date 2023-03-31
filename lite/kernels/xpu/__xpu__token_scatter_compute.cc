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

#include "lite/kernels/xpu/__xpu__token_scatter_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUTokenScatterCompute::PrepareForRun() {
  scatter_out_guard_ = TargetWrapperXPU::MallocScratchPad(4 * 1024 * 1024);
}

void XPUTokenScatterCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  if (param.SeqLod != nullptr && param.PadSeqLen != nullptr) {
    int64_t batch = param.X->lod()[0].size() - 1;
    int64_t seq_len = param.X->lod()[0].back();
    int64_t dim = param.X->dims()[1];
    scatter_out_guard_->Reserve(param.X->numel() * sizeof(float));
    param.Out->Resize({batch, param.PadSeqLen->data<int>()[0], dim});
    int r = xdnn::scatter<float, int>(
        ctx.GetRawContext(),
        param.X->data<float>(),
        param.Updates->data<float>(),
        reinterpret_cast<float*>(scatter_out_guard_->addr_),
        {param.CLSInds->data<int>(), param.CLSInds->dims()[0], nullptr},
        {seq_len, dim},
        0,
        true);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_pad<float, int>(
        ctx.GetRawContext(),
        reinterpret_cast<float*>(scatter_out_guard_->addr_),
        param.SeqLod->data<int>(),
        param.Out->mutable_data<float>(TARGET(kXPU)),
        batch,
        param.PadSeqLen->data<int>()[0],
        dim,
        0.0f);
    CHECK_EQ(r, 0);
  } else {
    int64_t seq_len = param.X->dims()[0] * param.X->dims()[1];
    int64_t dim = param.X->dims()[2];
    int r = xdnn::scatter<float, int>(
        ctx.GetRawContext(),
        param.X->data<float>(),
        param.Updates->data<float>(),
        param.Out->mutable_data<float>(TARGET(kXPU)),
        {param.CLSInds->data<int>(), param.CLSInds->dims()[0], nullptr},
        {seq_len, dim},
        0,
        true);
    CHECK_EQ(r, 0);
  }
}
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__token_scatter,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUTokenScatterCompute,
                     def)
    .BindInput("CLSInds",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Updates", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
