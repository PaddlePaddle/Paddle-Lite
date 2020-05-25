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

pthread_mutex_t Graph::mutex_compiler_ = PTHREAD_MUTEX_INITIALIZER;

void Graph::AddNode(const std::string& name) {
  nodes_.insert(std::make_pair(name, name));
}

void Graph::CreateCompilerHandle() {
  pthread_mutex_lock(&mutex_compiler_);
#ifdef BM1682
  compiler_handle_ = create_bmcompiler("BM1682");
#else
  compiler_handle_ = create_bmcompiler("BM1684");
#endif
  CHECK(compiler_handle_ != nullptr);
}

void Graph::UnlockCompilerMutex() { pthread_mutex_unlock(&mutex_compiler_); }

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
