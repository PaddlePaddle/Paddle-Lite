// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T, typename IndexT = int>
void Gather(const Tensor &src, const Tensor &index, Tensor *output) {
  auto *p_src = src.data<T>();
  auto *p_index = index.data<IndexT>();
  auto *p_output = output->mutable_data<T>();

  auto src_dims = src.dims();
  int64_t slice_size = 1;
  for (size_t i = 1; i < src_dims.size(); i++) slice_size *= src_dims[i];
  size_t slice_bytes = slice_size * sizeof(T);

  int64_t index_size = index.numel();
  for (int64_t i = 0; i < index_size; i++) {
    IndexT index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
