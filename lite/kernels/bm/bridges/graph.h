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

#include <pthread.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

// Graph to collect all of converted BM IR nodes
class Graph {
 public:
  void AddNode(const std::string& name);
  bool HasNode(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }
  void CreateCompilerHandle();
  void* GetCompilerHandle() { return compiler_handle_; }
  void UnlockCompilerMutex();

 private:
  std::unordered_map<std::string, std::string> nodes_;
  void* compiler_handle_;
  static pthread_mutex_t mutex_compiler_;
};

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
