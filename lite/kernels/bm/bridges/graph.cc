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

#include "lite/kernels/bm/bridges/graph.h"
#include <bmcompiler_if.h>

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

void Graph::AddNode(const std::string& name) {
  nodes_.insert(std::make_pair(name, name));
}

void Graph::CreateCompilerHandle() {
  compiler_handle_ = create_bmcompiler("BM1684");
  CHECK(compiler_handle_ != nullptr);
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
