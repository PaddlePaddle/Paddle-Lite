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

#include <set>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/core/model/base/apis.h"

namespace paddle {
namespace lite {

template <typename T>
class CompatibleChecker {
 public:
  explicit CompatibleChecker(const T& program,
                             const int64_t mini_version = 1005000)
      : program_(program), mini_version_(mini_version) {}

  bool operator()(const lite_api::Place& place) {
    bool status = true;
    const std::set<std::string>& ops_type = OpsType(&program_);
    if (ops_type.empty()) {
      VLOG(3) << "You are checking the compatibility of an empty program.";
    }
    for (const auto& type : ops_type) {
      bool ret = CheckKernelVersion(type, place);
      VLOG(3) << "Kernel version is supported: " << type << ", " << ret;
      status = status && ret;
    }
    return status;
  }

 private:
  std::set<std::string> OpsType(T* program);
  bool CheckKernelVersion(const std::string& type,
                          const lite_api::Place& place);
  T program_;
  int64_t mini_version_;
};

}  // namespace lite
}  // namespace paddle
