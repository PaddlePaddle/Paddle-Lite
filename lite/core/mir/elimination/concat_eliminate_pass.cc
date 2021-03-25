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

#include "lite/core/mir/elimination/concat_eliminate_pass.h"
#include <memory>
#include <set>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
std::vector<size_t> ConcatEliminator::StrideNumel(const DDim& ddim) {
  std::vector<size_t> strides(ddim.size());
  strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return strides;
}
void ConcatEliminator::ComputeConcat(const std::vector<lite::Tensor*> inputs,
                                     lite::Tensor* output) {
  size_t output_offset = 0;
  for (auto* in : inputs) {
    auto in_stride = StrideNumel(in->dims());
    auto out_stride = StrideNumel(output->dims());
    void* dst = output->mutable_data<float>() + output_offset;
    const void* src = in->data<float>();
    // src and dst tensor should have the same dims size.
    CHECK(in_stride.size() == out_stride.size());
    std::memcpy(dst, src, sizeof(float) * in_stride[0]);
    output_offset += in_stride[0];
  }
}
void ConcatEliminator::BuildPattern() {
  auto* reshape2_output_0 = VarNode("reshape2_output_0")
                                ->assert_is_op_nth_input("concat", "X", 0)
                                ->assert_is_persistable_var();
  auto* reshape2_output_1 = VarNode("reshape2_output_1")
                                ->assert_is_op_nth_input("concat", "X", 1)
                                ->assert_is_persistable_var();
  auto* reshape2_output_2 = VarNode("reshape2_output_2")
                                ->assert_is_op_nth_input("concat", "X", 2)
                                ->assert_is_persistable_var();
  auto* reshape2_output_3 = VarNode("reshape2_output_3")
                                ->assert_is_op_nth_input("concat", "X", 3)
                                ->assert_is_persistable_var();
  auto* reshape2_output_4 = VarNode("reshape2_output_4")
                                ->assert_is_op_nth_input("concat", "X", 4)
                                ->assert_is_persistable_var();
  auto* reshape2_output_5 = VarNode("reshape2_output_5")
                                ->assert_is_op_nth_input("concat", "X", 5)
                                ->assert_is_persistable_var();
  // concat #0 node
  auto* concat = OpNode("concat", "concat");
  auto* concat_output_0 =
      VarNode("concat_output_0")->assert_is_op_output("concat", "Out");

  // concat topology
  std::vector<PMNode*> concat_inputs_0{reshape2_output_0,
                                       reshape2_output_1,
                                       reshape2_output_2,
                                       reshape2_output_3,
                                       reshape2_output_4,
                                       reshape2_output_5};
  concat_inputs_0 >> *concat >> *concat_output_0;
}

void ConcatEliminator::DeleteInterNodes(SSAGraph* graph) {
  GraphSafeRemoveNodes(graph, nodes2rm_);
}

void ConcatEliminator::InsertNewNode(SSAGraph* graph,
                                     const key2nodes_t& matched) {
  auto concat_instruct = matched.at("concat")->stmt();
  auto op_desc = concat_instruct->mutable_op_info();
  auto* scope = concat_instruct->op()->scope();

  // get concat's input tensor
  std::vector<lite::Tensor*> inputs_tensors;
  inputs_tensors.push_back(
      scope->FindVar(matched.at("reshape2_output_0")->arg()->name)
          ->GetMutable<lite::Tensor>());
  inputs_tensors.push_back(
      scope->FindVar(matched.at("reshape2_output_1")->arg()->name)
          ->GetMutable<lite::Tensor>());
  inputs_tensors.push_back(
      scope->FindVar(matched.at("reshape2_output_2")->arg()->name)
          ->GetMutable<lite::Tensor>());
  inputs_tensors.push_back(
      scope->FindVar(matched.at("reshape2_output_3")->arg()->name)
          ->GetMutable<lite::Tensor>());
  inputs_tensors.push_back(
      scope->FindVar(matched.at("reshape2_output_4")->arg()->name)
          ->GetMutable<lite::Tensor>());
  inputs_tensors.push_back(
      scope->FindVar(matched.at("reshape2_output_5")->arg()->name)
          ->GetMutable<lite::Tensor>());

  // get concat's output tensor
  auto output_var = scope->FindVar(op_desc->Output("Out").front());
  auto output_t = output_var->GetMutable<lite::Tensor>();

  // get concat's other attr
  auto axis = op_desc->GetAttr<int>("axis");
  if (axis != 0) {
    LOG(WARNING) << "the ssd priorbox concat's axis must be 0 ";
  }

  // calcu concat
  ComputeConcat(inputs_tensors, output_t);

  // set the output as persistable-tensor
  output_t->set_persistable(true);
  auto concat_output_node = matched.at("concat_output_0");
  concat_output_node->arg()->is_weight = true;

  nodes2rm_.insert(matched.at("concat"));
  nodes2rm_.insert(matched.at("reshape2_output_0"));
  nodes2rm_.insert(matched.at("reshape2_output_1"));
  nodes2rm_.insert(matched.at("reshape2_output_2"));
  nodes2rm_.insert(matched.at("reshape2_output_3"));
  nodes2rm_.insert(matched.at("reshape2_output_4"));
  nodes2rm_.insert(matched.at("reshape2_output_5"));
}

void ConcatEliminatePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  ConcatEliminator eliminator;
  eliminator(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_concat_eliminate_pass,
                  paddle::lite::mir::ConcatEliminatePass)
    .BindTargets({TARGET(kNPU), TARGET(kRKNPU)});
