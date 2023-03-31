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

#include "lite/kernels/xpu/__xpu__clip_lod_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUClipLodCompute::PrepareForRun() {}

void XPUClipLodCompute::Run() {
  auto& param = this->Param<param_t>();
  param.NewPadSeqLen->mutable_data<int>()[0] =
      param.PadSeqLen->data<int>()[0] * param.keep_ratio;
  auto* seq_len = param.SeqLen->data<int64_t>();
  auto* new_seq_lod = param.NewSeqLod->mutable_data<int>();
  auto* new_seq_len = param.NewSeqLen->mutable_data<int64_t>();
  new_seq_lod[0] = 0;
  for (int batch_idx = 0; batch_idx < param.SeqLen->dims()[0]; ++batch_idx) {
    new_seq_len[batch_idx] = seq_len[batch_idx] * param.keep_ratio;
    new_seq_lod[batch_idx + 1] =
        new_seq_lod[batch_idx] + new_seq_len[batch_idx];
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__clip_lod,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUClipLodCompute,
                     def)
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("NewSeqLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("NewSeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("NewPadSeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
