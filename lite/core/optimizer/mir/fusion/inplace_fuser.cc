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

#include "lite/core/optimizer/mir/fusion/inplace_fuser.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void InplaceFuser::BuildPattern() {
  auto* input = VarNode("input")
                    ->assert_is_op_input(type_, "X")
                    ->assert_only_one_output()
                    ->assert_var_not_persistable()
                    ->AsInput();

  auto* op_node = OpNode("inplace", type_)->assert_is_op(type_);

  auto* output = VarNode("output")
                     ->assert_is_op_output(type_, "Out")
                     ->assert_only_one_output()
                     ->AsOutput();

  *input >> *op_node >> *output;
}

void InplaceFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  bool inplace = true;
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
