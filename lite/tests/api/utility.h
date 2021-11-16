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

// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

template <class T = float>
T CalOutAccuracy(const std::vector<std::vector<T>>& out,
                 const std::vector<std::vector<T>>& ref_out,
                 const float abs_error = 1e-5) {
  size_t right_count = 0;
  size_t all_count = 0;
  for (size_t i = 0; i < out.size(); i++) {
    if (out[i].size() != ref_out[i].size()) {
      LOG(FATAL) << "size error: out_size = " << out[i].size()
                 << ", ref_out_size = " << ref_out[i].size() << ", i = " << i;
    }
    for (size_t j = 0; j < out[i].size(); j++) {
      if (std::abs(out[i][j] - ref_out[i][j]) < abs_error) {
        right_count++;
      }
      all_count++;
    }
  }
  return static_cast<float>(right_count) / static_cast<float>(all_count);
}

template <typename T>
void fill_tensor(std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
                 const int idx,
                 const T* data,
                 const std::vector<int64_t>& shape) {
  auto tensor = predictor->GetInput(idx);
  tensor->Resize(shape);
  auto tensor_data = tensor->mutable_data<T>();
  int64_t size = 1;
  for (auto i : shape) size *= i;
  memcpy(tensor_data, data, sizeof(T) * size);
}

}  // namespace lite
}  // namespace paddle
