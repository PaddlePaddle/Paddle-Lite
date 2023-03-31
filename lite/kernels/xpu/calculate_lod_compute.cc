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

#include "lite/kernels/xpu/calculate_lod_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void CalculateLodCompute::PrepareForRun() {}

void CalculateLodCompute::Run() {
  auto& param = this->Param<param_t>();

  auto& mask_dims = param.Mask->dims();
  auto batch_size = mask_dims[0];
  auto max_seqlen = mask_dims[1];
  param.PadSeqLen->mutable_data<int>()[0] = max_seqlen;

  auto* mask_ptr = param.Mask->data<float>();
  auto* seq_lod = param.SeqLod->mutable_data<int>();
  auto* seq_len = param.SeqLen->mutable_data<int64_t>();
  seq_lod[0] = 0;
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    seq_len[batch_idx] = 0;
    for (auto seq_idx = max_seqlen - 1; seq_idx >= 0; --seq_idx) {
      if (mask_ptr[batch_idx * max_seqlen + seq_idx] > 1e-7) {
        seq_len[batch_idx] = seq_idx + 1;
        break;
      }
    }
    seq_lod[batch_idx + 1] = seq_lod[batch_idx] + seq_len[batch_idx];
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(calculate_lod,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::CalculateLodCompute,
                     def)
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("SeqLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("PadSeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
