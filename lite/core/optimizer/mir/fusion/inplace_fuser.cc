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

void InplaceFuser::BuildPattern() {
  auto* x = VarNode("x");
  auto* inplace = OpNode("inplace", type_);
  auto* inplace_out = VarNode("Out");
  auto* out1 = OpNode("out1");

  *x >> *inplace >> *inplace_out >> *out1;
}

void InplaceFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = const_cast<OpInfo*>(matched.at("inplace")->stmt()->op_info());
  op_desc->SetAttr<bool>("inplace", true);
}

void Inplace2OutFuser::BuildPattern() {
  auto* x = VarNode("x");
  auto* inplace =
      OpNode("inplace", type_)->assert_op_attr<bool>("inplace", true);
  auto* inplace_out = VarNode("Out");
  auto* out1 = OpNode("out1");
  auto* out2 = OpNode("out2");

  *x >> *inplace >> *inplace_out >> *out1;
  *inplace_out >> *out2;
}

void Inplace2OutFuser::InsertNewNode(SSAGraph* graph,
                                     const key2nodes_t& matched) {
  auto op_desc = const_cast<OpInfo*>(matched.at("inplace")->stmt()->op_info());
  op_desc->SetAttr<bool>("inplace", false);
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
