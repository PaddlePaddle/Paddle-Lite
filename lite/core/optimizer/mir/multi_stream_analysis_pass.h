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

#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/kernel.h"
#include "lite/core/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * MultiStreamAnalysisPass will find the correct launch sequence for all ops.
 * Ideally, the order should be multiple asynchronous ops and a small number of
 * synchronous ops.
 */
class MultiStreamAnalysisPass : public StmtPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  // Init resource list. Set all ops except feed to inaccessible state and set
  // stream id according to the numer of inputs.
  void Init(SSAGraph* graph);

  // Clean state information of all member variables.
  void CleanUp();

  // After launching, unlock the output resources of op.
  void Launch(Node* stmt_node);

  // If all inputs of an op are accessible, the op is considered to be in the
  // prepared state
  bool IsPrepared(Node* stmt_node);

  // Determine if all inputs of op are accessible.
  bool CheckAccess(const std::vector<std::string>& args);

  // The logic of selecting a stream:
  // 1. Make the number of ops on each stream as close as possible.
  // 2. The selected stream must be one of the streams contained in the input
  // arg
  int SelectStreamId(const std::vector<int>& lanes);

  // Check if the model's ops are all supported. If you encounter unsupported
  // ops, exit
  bool CheckOpSupport();

 private:
  std::list<Node*> wait_que_;
  std::list<Node*> wait_que_cpu_;
  std::queue<Node*> exec_que_;
  std::vector<Node*> exec_ops_;
  std::vector<std::vector<Node*>> ops_in_streams_;
  std::map<std::string, bool> resources_;
  std::map<std::string, int> map_arg_to_lane_;
  int max_stream_;
  int io_copy_once_num_;
  std::set<std::string> op_types_set_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
