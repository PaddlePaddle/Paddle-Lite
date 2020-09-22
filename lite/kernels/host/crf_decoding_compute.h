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

#pragma once
#include <algorithm>
#include <limits>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void Decode(const Tensor& emission_weights,
            const Tensor& transition_weights,
            Tensor* decoded_path) {
  auto emission_dims = emission_weights.dims();
  const int64_t seq_len = emission_dims[0];
  const int64_t tag_num = emission_dims[1];
  const T* x = emission_weights.data<T>();
  const T* w = transition_weights.data<T>();
  int64_t* path = decoded_path->mutable_data<int64_t>();

  // alpha is a memo table. An element alpha(k, v) records the score of the
  // best sequence of tags from position 1 to position k with v being the end
  // tag.
  Tensor alpha;
  alpha.Resize(emission_dims);
  T* alpha_value = alpha.mutable_data<T>();
  Tensor track;
  track.Resize(emission_dims);
  int* track_value = track.mutable_data<int>();

  const int state_trans_base_idx = 2;
  for (int i = 0; i < tag_num; ++i) {
    alpha_value[i] = w[i] + x[i];
  }

  for (int k = 1; k < seq_len; ++k) {
    for (int i = 0; i < tag_num; ++i) {
      T max_score = -(std::numeric_limits<T>::max)();
      int max_j = 0;
      for (size_t j = 0; j < tag_num; ++j) {
        T score = alpha_value[(k - 1) * tag_num + j] +
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

  T max_score = -(std::numeric_limits<T>::max)();
  int max_i = 0;
  for (size_t i = 0; i < tag_num; ++i) {
    T score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];
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

class CrfDecodingCompute : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  void Run() override;

  virtual ~CrfDecodingCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
