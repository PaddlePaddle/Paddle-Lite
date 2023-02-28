// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__mask_adaptive_compute.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUMaskAdaptiveCompute::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.Mask && param.Mask->data<float>()) << "mask null";
  auto& mask_dims = param.Mask->dims();
  auto batch_size = mask_dims[0];
  auto pad_seq_len = mask_dims[1];
  param.PadSeqLen->mutable_data<int>()[0] = pad_seq_len;
  auto* seq_lod = param.SeqLod;
  seq_lod->Resize({batch_size + 1});
  std::vector<int> cpu_seq_lod{0};
  auto* seq_len = param.Length;
  seq_len->Resize({batch_size});
  std::vector<int64_t> cpu_seq_lens;

  const float* mask_ptr = param.Mask->data<float>();

  for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int cur_batch_seq_len = 0;
    for (auto seq_idx = 0; seq_idx < pad_seq_len; seq_idx++) {
      if (mask_ptr[batch_idx * pad_seq_len + seq_idx] > 1e-7) {
        cur_batch_seq_len += 1;
      } else {
        break;
      }
    }
    CHECK_GT(cur_batch_seq_len, 0);
    cpu_seq_lod.push_back(cpu_seq_lod.back() + cur_batch_seq_len);
    cpu_seq_lens.push_back(cur_batch_seq_len);
  }
  auto* seq_lod_ptr = seq_lod->mutable_data<int>();
  memcpy(seq_lod_ptr, cpu_seq_lod.data(), cpu_seq_lod.size() * sizeof(int));
  auto* seq_lens_ptr = seq_len->mutable_data<int64_t>();
  memcpy(
      seq_lens_ptr, cpu_seq_lens.data(), cpu_seq_lens.size() * sizeof(int64_t));
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__mask_adaptive,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMaskAdaptiveCompute,
                     def)
    .BindInput("Mask",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("SeqLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("PadSeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
