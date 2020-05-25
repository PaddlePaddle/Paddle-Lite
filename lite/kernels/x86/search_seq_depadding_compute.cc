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

#include "lite/kernels/x86/search_seq_depadding_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
void SearchSeqDepaddingCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto* pad = param.pad;
  auto* src = param.src;
  auto* out = param.out;

  const int pad_batch = pad->lod()[0].size() - 1;
  const int src_batch = src->lod()[0].size() - 1;
  if (pad_batch % src_batch != 0) {
    LOG(FATAL) << "Mismatch batch size.";
  }

  const auto& pad_offset = pad->lod()[0];
  const int pad_cap_e = pad->dims()[1];
  const auto& src_offset = src->lod()[0];
  const int src_cap_l = src->dims()[0];

  LoD out_lod;
  out_lod.push_back(src_offset);
  out->set_lod(out_lod);
  out->Resize({src_cap_l, pad_cap_e});

  const auto* pad_data = pad->template data<T>();
  auto* out_data = out->template mutable_data<T>();
  for (int i = 0; i < src_batch; ++i) {
    const int src_i_l = src_offset[i + 1] - src_offset[i];
    const int pad_i_l = pad_offset[i + 1] - pad_offset[i];
    if (pad_i_l < src_i_l) {
      LOG(FATAL)
          << "the length of padding seq input is less than source seq input.";
    }
    memcpy(out_data + src_offset[i] * pad_cap_e,
           pad_data + pad_offset[i] * pad_cap_e,
           src_i_l * pad_cap_e * sizeof(T));
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    search_seq_depadding,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::SearchSeqDepaddingCompute<float>,
    def)
    .BindInput("Pad", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Src", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
