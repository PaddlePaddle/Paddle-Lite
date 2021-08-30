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

#include "lite/core/model/general/op_desc.h"
#include <set>
#include <utility>

namespace paddle {
namespace lite {
namespace general {

std::vector<std::string> OpDesc::OutputArgumentNames() const {
  std::vector<std::string> res;
  for (const auto& x : outputs_) res.push_back(x.first);
  return res;
}

std::vector<std::string> OpDesc::input_vars() const {
  std::vector<std::string> res;
  for (const auto& arg : InputArgumentNames()) {
    for (auto& vars : Input(arg)) {
      res.emplace_back(vars.begin(), vars.end());
    }
  }
  return res;
}

std::vector<std::string> OpDesc::output_vars() const {
  std::vector<std::string> res;
  for (const auto& arg : OutputArgumentNames()) {
    for (auto& vars : Output(arg)) {
      res.emplace_back(vars.begin(), vars.end());
    }
  }
  return res;
}

std::vector<std::string> OpDesc::InputArgumentNames() const {
  std::vector<std::string> res;
  for (const auto& x : inputs_) res.push_back(x.first);
  return res;
}

std::vector<std::string> OpDesc::Input(const std::string& param) const {
  auto it = inputs_.find(param);
  CHECK(it != inputs_.end());
  return it->second;
}

std::vector<std::string> OpDesc::Output(const std::string& param) const {
  auto it = outputs_.find(param);
  CHECK(it != outputs_.end());
  return it->second;
}

bool OpDesc::HasOutput(const std::string& param) const {
  auto it = outputs_.find(param);
  return it != outputs_.end();
}

}  // namespace general
}  // namespace lite
}  // namespace paddle
