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

#include <vector>
#include "lite/model_parser/naive_buffer/naive_buffer.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

/// BuilderType must have data method
template <typename T, typename BuilderType>
std::vector<T> RepeatedToVector(const ListBuilder<BuilderType>& builder) {
  std::vector<T> res;
  for (size_t i = 0; i < builder.size(); ++i) {
    res.push_back(builder.Get(i).data());
  }
  return res;
}

/// BuilderType must have set method
template <typename T, typename BuilderType>
void VectorToRepeated(const std::vector<T>& data,
                      ListBuilder<BuilderType>* builder) {
  CHECK(builder);
  builder->Clear();
  for (auto& val : data) {
    builder->New()->set(val);
  }
}

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
