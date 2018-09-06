/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef CRF_OP
#pragma once

#include <limits>
#include <vector>
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
template <typename P>
void Decode(const Tensor& emission_weights, const Tensor& transition_weights,
            Tensor* decoded_path) {
  auto emission_dims = emission_weights.dims();
  const size_t seq_len = emission_dims[0];
  const size_t tag_num = emission_dims[1];

  const size_t state_trans_base_idx = 2;

  const P* x = emission_weights.data<P>();
  const P* w = transition_weights.data<P>();
  int64_t* path = decoded_path->data<int64_t>();

  // alpha is a memo table. An element alpha(k, v) records the score of the
  // best sequence of tags from position 1 to position k with v being the end
  // tag.
  Tensor alpha;
  P* alpha_value = alpha.mutable_data<P>(emission_dims);
  Tensor track;
  int* track_value = track.mutable_data<int>(emission_dims);
  for (size_t i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

  for (size_t k = 1; k < seq_len; ++k) {
    for (size_t i = 0; i < tag_num; ++i) {
      P max_score = -std::numeric_limits<P>::max();
      int max_j = 0;
      for (size_t j = 0; j < tag_num; ++j) {
        P score = alpha_value[(k - 1) * tag_num + j] +
                  w[(j + state_trans_base_idx) * tag_num + i];
        if (score > max_score) {
          max_score = score;
          max_j = j;
        }
      }

      alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
      track_value[k * tag_num + i] = max_j;
    }
  }
  P max_score = -std::numeric_limits<P>::max();
  int max_i = 0;
  for (size_t i = 0; i < tag_num; ++i) {
    P score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];
    if (score > max_score) {
      max_score = score;
      max_i = i;
    }
  }
  path[seq_len - 1] = max_i;
  for (int k = seq_len - 1; k >= 1; --k) {
    path[k - 1] = max_i = track_value[k * tag_num + max_i];
  }
}
template <typename P>
void CrfCompute(const CrfParam<CPU>& param) {
  auto* emission = param.InputEmission();
  auto* transition = param.InputTransition();
  auto* label = param.InputLabel();
  auto* decoded_path = param.outputVBP();
  //  DLOG<<*emission;
  //  DLOG<<*transition;
  //  DLOG<<*label;

  PADDLE_MOBILE_ENFORCE(emission->NumLevels() == 1U,
                        "The Input(Emission) should be a sequence.");
  auto lod = emission->lod();
  PADDLE_MOBILE_ENFORCE(lod.size(),
                        "The Input(Emission) should be a sequence.");
  const size_t level = 0;
  const size_t seq_num = lod[level].size() - 1;
  int64_t* path = decoded_path->mutable_data<int64_t>();
  int numel = decoded_path->numel();
  memset(static_cast<void*>(path), 0, sizeof(int64_t) * numel);
  for (size_t i = 0; i < seq_num; ++i) {
    int start_pos = static_cast<int>(lod[level][i]);
    int end_pos = static_cast<int>(lod[level][i + 1]);
    Tensor decoded_path_one_seq = decoded_path->Slice(start_pos, end_pos);
    Decode<P>(emission->Slice(start_pos, end_pos), *transition,
              &decoded_path_one_seq);
  }
  if (label) {
    PADDLE_MOBILE_ENFORCE(label->NumLevels() == 1U,
                          "The Input(Label) should be a sequence.");
    const int64_t* label_value = label->data<int64_t>();
    size_t batch_size = emission->dims()[0];
    for (size_t i = 0; i < batch_size; ++i) {
      path[i] = label_value[i] == path[i] ? 1 : 0;
    }
  }
}
}  // namespace operators

}  // namespace paddle_mobile

#endif
