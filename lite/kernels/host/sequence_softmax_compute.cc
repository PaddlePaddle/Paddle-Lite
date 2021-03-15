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

#include "lite/kernels/host/sequence_softmax_compute.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void sequence_softmax(const float* input,
                      const std::vector<uint64_t>& seq_offset,
                      float* out) {
  int seq_num = seq_offset.size() - 1;
  for (int i = 0; i < seq_num; i++) {
    float seq_max = input[seq_offset[i]];
    float exp_sum = 0.f;
    for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
      seq_max = std::max(seq_max, input[j]);
    }
    for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
      exp_sum += exp(input[j] - seq_max);
    }
    for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
      out[j] = exp(input[j] - seq_max) / exp_sum;
    }
  }
}

void SequenceSoftmaxCompute::Run() {
  auto& param = this->Param<param_t>();
  const auto* x_data = param.X->data<float>();
  auto* o_data = param.Out->mutable_data<float>();
  auto input_dims = param.X->dims();
  int in_h = input_dims[0];
  int in_w = param.X->numel() / in_h;
  CHECK_EQ(in_w, 1) << "input dims is not valid";
  auto seq_offset = param.X->lod()[0];
  CHECK_EQ(in_h, seq_offset.back()) << "input dims is not valid";

  sequence_softmax(x_data, seq_offset, o_data);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_softmax,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequenceSoftmaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
