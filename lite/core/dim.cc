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

#include "lite/core/dim.h"
#include <string>

namespace paddle {
namespace lite {
using value_type = int64_t;

value_type DDimLite::production() const {
  value_type res = 1;
  for (size_t i = 0; i < data_.size(); i++) {
    res *= data_[i];
  }
  return res;
}

value_type DDimLite::count(int start, int end) const {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int>(data_.size()));
  if (end < start) {
    return 0;
  }
  value_type sum = 1;
  for (auto i = start; i < end; ++i) {
    sum *= data_[i];
  }
  return sum;
}

DDimLite DDimLite::Slice(int start, int end) const {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int>(data_.size()));
  std::vector<value_type> new_dim(end - start);
  for (int i = start; i < end; i++) {
    new_dim[i - start] = data_[i];
  }
  return DDim(new_dim);
}

std::string DDimLite::repr() const {
  STL::stringstream ss;
  if (empty()) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < this->size() - 1; i++) {
    ss << (*this)[i] << ",";
  }
  if (!this->empty()) ss << (*this)[size() - 1];
  ss << "}";
  return ss.str();
}
}  // namespace lite
}  // namespace paddle
