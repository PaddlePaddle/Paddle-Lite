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

#include "lite/core/mir/fusion/inplace_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void InplaceFuser::BuildPattern() { OpNode("inplace", type_); }

void InplaceFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto out_var_nodes = matched.at("inplace")->outlinks;
  bool inplace = true;
  for (auto& out_var_node : out_var_nodes) {
    if (out_var_node->outlinks.size() > 1) {
      inplace = false;
    }
  }
  auto* stmt = matched.at("inplace")->stmt();
  auto op = stmt->op();
  cpp::OpDesc* op_desc = op->mutable_op_info();
  op_desc->SetAttr<bool>("inplace", inplace);
  stmt->op()->Attach(*op_desc, op->scope());
  stmt->op()->AttachKernel(&(stmt->picked_kernel()));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
