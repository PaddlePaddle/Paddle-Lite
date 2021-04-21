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

#include "../../nnadapter_driver.h"  // NOLINT

namespace nnadapter {
namespace driver {
namespace huawei_kirin_npu {

class Context {
 public:
  explicit Context(void* raw_ctx) : raw_ctx_(raw_ctx) {}
  ~Context() {}

 private:
  void* raw_ctx_{nullptr};
};

class Model {
 public:
  Model() {}
  ~Model();

  int CreateFromGraph(driver::Graph* graph);
  int CreateFromCache(void* buffer, size_t length);
};

}  // namespace huawei_kirin_npu
}  // namespace driver
}  // namespace nnadapter
