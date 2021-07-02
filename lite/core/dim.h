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

#ifndef LITE_CORE_DIM_H_
#define LITE_CORE_DIM_H_

#include <algorithm>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
// class DDimLite;

class DDimLite {
 public:
  using value_type = int64_t;

  DDimLite() = default;

  explicit DDimLite(const std::vector<value_type> &x) { ConstructFrom(x); }
  // DDimLite(std::initializer_list<value_type> init_list) :
  // DDimLite(std::vector<value_type>(init_list)) {}

  void ConstructFrom(const std::vector<value_type> &x) { data_ = x; }

  value_type operator[](int offset) const { return data_[offset]; }
  value_type &operator[](int offset) { return data_[offset]; }
  std::vector<int64_t> Vectorize() const { return data_; }

  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  value_type production() const;

  const std::vector<value_type> &data() const { return data_; }
  value_type count(int start, int end) const;

  DDimLite Slice(int start, int end) const;

  DDimLite Flatten2D(int col) const {
    return DDimLite(std::vector<value_type>(
        {Slice(0, col).production(), Slice(col, size()).production()}));
  }

  std::string repr() const;

  friend STL::ostream &operator<<(STL::ostream &os, const DDimLite &dims) {
    os << dims.repr();
    return os;
  }

  friend bool operator==(const DDimLite &a, const DDimLite &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const DDimLite &a, const DDimLite &b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return true;
    }
    return false;
  }

 private:
  std::vector<value_type> data_;
};

using DDim = paddle::lite::DDimLite;
}  // namespace lite
}  // namespace paddle

#endif  // LITE_CORE_DIM_H_
