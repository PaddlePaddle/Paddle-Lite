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
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {

template <typename Elem>
class OrderedMap {
  std::vector<Elem> list_;
  std::map<std::string, int> order_;

 public:
  void Set(const std::string& key, Elem&& e) {
    list_.emplace_back(std::move(e));
    CHECK(!order_.count(key)) << "duplicate key '" << key << "' found";
    order_[key] = list_.size() - 1;
  }

  const Elem& Get(const std::string& key) const {
    CHECK(order_.count(key)) << "No key " << key << " found";
    return list_[order_.at(key)];
  }

  Elem& GetMutable(const std::string& key) {
    CHECK(order_.count(key)) << "No key " << key << " found";
    return list_[order_[key]];
  }

  std::vector<Elem>& elements() { return list_; }
  const std::vector<Elem>& elements() const { return list_; }
};

}  // namespace lite
}  // namespace paddle
