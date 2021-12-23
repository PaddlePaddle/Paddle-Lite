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

#include "lite/kernels/host/crf_decoding_compute.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void CrfDecodingCompute::Run() {
  auto& param = Param<operators::CrfDecodingParam>();
  auto* emission_weights = param.emission;
  auto* transition_weights = param.transition;
  auto* label = param.label;
  auto* decoded_path = param.viterbi_path;

  int64_t* path = decoded_path->mutable_data<int64_t>();
  std::fill(path, path + decoded_path->numel(), 0);

  if (param.length != nullptr) {
    auto* length = param.length;
    int64_t seq_num = length->numel();
    const int64_t* length_data = length->data<int64_t>();
    auto in_dims = emission_weights->dims();

    Tensor emission_weights_tmp = *emission_weights;
    emission_weights_tmp.Resize({in_dims[0] * in_dims[1], in_dims[2]});
    decoded_path->Resize({in_dims[0] * in_dims[1], 1});
    for (int64_t i = 0; i < seq_num; ++i) {
      if (length_data[i] == 0) continue;
      int64_t start_pos = i * in_dims[1];
      int64_t end_pos = start_pos + length_data[i];
      Tensor decoded_path_one_seq =
          decoded_path->Slice<int64_t>(start_pos, end_pos);
      Decode<float>(emission_weights_tmp.Slice<float>(start_pos, end_pos),
                    *transition_weights,
                    &decoded_path_one_seq);
    }
    decoded_path->Resize({in_dims[0], in_dims[1]});
    if (label != nullptr) {
      const int64_t* label_value = label->data<int64_t>();
      for (int64_t i = 0; i < seq_num; ++i) {
        for (int64_t j = 0; j < in_dims[1]; ++j) {
          int64_t start_pos = i * in_dims[1];
          if (j < length_data[i]) {
            path[start_pos + j] =
                label_value[start_pos + j] == path[start_pos + j] ? 1 : 0;
          } else {
            path[start_pos + j] = 0;
          }
        }
      }
    }
  } else {
    auto lod = emission_weights->lod();
    CHECK_EQ(lod.size(), 1UL);
    CHECK_GT(lod.size(), 0);
    const size_t level = 0;
    const size_t seq_num = lod[level].size() - 1;

    for (size_t i = 0; i < seq_num; ++i) {
      if (lod[level][i] == lod[level][i + 1]) continue;
      int64_t start_pos = static_cast<int64_t>(lod[level][i]);
      int64_t end_pos = static_cast<int64_t>(lod[level][i + 1]);
      Tensor decoded_path_one_seq =
          decoded_path->Slice<int64_t>(start_pos, end_pos);
      Decode<float>(emission_weights->Slice<float>(start_pos, end_pos),
                    *transition_weights,
                    &decoded_path_one_seq);
    }
    if (label != nullptr) {
      auto label_lod = label->lod();
      CHECK_EQ(label_lod.size(), 1);
      const int64_t* label_value = label->data<int64_t>();
      int64_t numel = label->numel();
      for (int64_t i = 0; i < numel; ++i) {
        path[i] = label_value[i] == path[i] ? 1 : 0;
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(crf_decoding,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::CrfDecodingCompute,
                     def)
    .BindInput("Emission", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Transition", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Label",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Length",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("ViterbiPath",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
