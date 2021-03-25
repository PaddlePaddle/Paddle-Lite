// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/mir/elimination/reshape_eliminate_pass.h"
#include <memory>
#include <set>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
void ReshapeEliminator::ComputeReshape(const lite::Tensor* in,
                                       lite::Tensor* out) {
  // In CopyDataFrom, the target tensor's dims will be set to the source
  // tensor's dims.
  auto out_dims = out->dims();
  out->CopyDataFrom(*in);
  out->Resize(out_dims);
}

void ReshapeEliminator::BuildPattern() {
  // reshape2 #0 node
  auto* reshape_input = VarNode("reshape_input")
                            ->assert_is_op_input("reshape2", "X")
                            ->assert_is_persistable_var();
  auto* reshape2 = OpNode("reshape2", "reshape2");
  auto* reshape2_output_out =
      VarNode("reshape2_output_out")->assert_is_op_output("reshape2", "Out");

  // reshape2 topology
  *reshape_input >> *reshape2 >> *reshape2_output_out;
}

void ReshapeEliminator::DeleteInterNodes(SSAGraph* graph) {
  GraphSafeRemoveNodes(graph, nodes2rm_);
}

void ReshapeEliminator::InsertNewNode(SSAGraph* graph,
                                      const key2nodes_t& matched) {
  auto reshape_instruct = matched.at("reshape2")->stmt();
  auto op_desc = reshape_instruct->mutable_op_info();
  auto* scope = reshape_instruct->op()->scope();
  // get reshape's input tensor
  auto input_var = scope->FindVar(op_desc->Input("X").front());
  auto input_t = &(input_var->Get<lite::Tensor>());
  // get reshape's output tensor
  auto output_var = scope->FindVar(op_desc->Output("Out").front());
  auto output_t = output_var->GetMutable<lite::Tensor>();
  // get reshape's other attr

  // calcu reshape
  ComputeReshape(input_t, output_t);
  // set the output as persistable-tensor
  output_t->set_persistable(true);
  auto reshape_output_node = matched.at("reshape2_output_out");
  reshape_output_node->arg()->is_weight = true;

  nodes2rm_.insert(matched.at("reshape2"));
  nodes2rm_.insert(matched.at("reshape_input"));
}

void ReshapeEliminatePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  ReshapeEliminator eliminator;
  eliminator(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_reshape_eliminate_pass,
                  paddle::lite::mir::ReshapeEliminatePass)
    .BindTargets({TARGET(kNPU), TARGET(kRKNPU)});
