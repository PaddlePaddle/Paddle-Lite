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

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/context.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/type_precision_cast_pass.h"  // For UpdateInputs()
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/* fuse multi slice with softmax as __xpu__multi_softmax */
/* graph                                                 */
/*                                                       */
/*                         in_Input                      */
/*                    /   /  |  \    \                   */
/*                /      |   |      \     \              */
/*              /        |   |       |    |              */
/*             |         |   |       |    |              */
/*             |         |   |       |    |              */
/*           slice   slice  slice   ... slice            */
/*             |       |     |            |              */
/*         softmax softmax softmax  ... softmax          */
/*----------------------------------------------------   */

class XPUSingleSliceSoftmaxFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_output("__xpu__fc", "Output")
                      ->assert_is_op_input("slice", "Input")
                      ->AsInput();
    auto* slice =
        OpNode("slice", "slice")
            ->assert_op_attr_satisfied<std::vector<int>>(
                "axes",
                [](const std::vector<int>& attr) {
                  return attr.size() == 1 && attr[0] == 1;
                })
            ->assert_op_attr_satisfied<std::vector<int>>(
                "starts",
                [](const std::vector<int>& attr) { return attr.size() == 1; })
            ->assert_op_attr_satisfied<std::vector<int>>(
                "ends",
                [](const std::vector<int>& attr) { return attr.size() == 1; })
            ->AsIntermediate();
    auto* slice_out = VarNode("slice_out")
                          ->assert_is_op_output("slice", "Out")
                          ->assert_is_op_input("softmax", "X")
                          ->assert_only_one_output()
                          ->AsIntermediate();
    auto* softmax = OpNode("softmax", "softmax")
                        ->assert_op_attr<int>("axis", -1)
                        ->AsIntermediate();
    auto* softmax_out = VarNode("softmax_out")
                            ->assert_is_op_output("softmax", "Out")
                            ->AsOutput();

    *input >> *slice >> *slice_out >> *softmax >> *softmax_out;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* slice_instruct = matched.at("slice")->stmt();
    auto slice_op_desc = *slice_instruct->op_info();
    auto slice_op = matched.at("slice")->stmt()->op();
    auto* scope = slice_op->scope();

    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multi_softmax");
    auto input_name = matched.at("input")->arg()->name;
    op_desc.SetInput("Input", {input_name});
    op_desc.SetOutput("Output", {matched.at("softmax_out")->arg()->name});
    std::vector<int> lod{slice_op_desc.GetAttr<std::vector<int>>("starts")[0],
                         slice_op_desc.GetAttr<std::vector<int>>("ends")[0]};
    op_desc.SetAttr<std::vector<int>>("lod", lod);

    auto multi_softmax_op =
        LiteOpRegistry::Global().Create("__xpu__multi_softmax");
    auto& valid_places = slice_op->valid_places();
    multi_softmax_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(multi_softmax_op, valid_places);

    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(new_op_node, matched.at("softmax_out"));
  }
};

class XPUMultiSliceSoftmaxFuser {
 public:
  bool IsSamePredecessorOf(Node* op1, Node* op2) {
    for (auto* in1 : op1->inlinks) {
      for (auto* in2 : op2->inlinks) {
        if (in1 != in2) return false;
      }
    }
    return true;
  }
  void operator()(SSAGraph* graph) {
    std::vector<Node*> all_softmax;
    std::set<const Node*> to_remove;
    for (auto* node : graph->StmtTopologicalOrder()) {
      CHECK(node->IsStmt());
      if (node->stmt()->op_info()->Type() == "__xpu__multi_softmax") {
        if (all_softmax.empty() ||
            IsSamePredecessorOf(all_softmax.back(), node)) {
          all_softmax.push_back(node);
        } else {
          break;
        }
      }
    }
    VLOG(3) << "Found single_slice_softmax num: " << all_softmax.size();
    if (all_softmax.size() == 0) {
      return;
    }

    auto first_softmax = all_softmax[0]->stmt()->op();
    auto* scope = first_softmax->scope();
    auto& valid_places = first_softmax->valid_places();

    std::vector<bool> used(all_softmax.size(), false);
    std::vector<std::string> out_names;
    Node* input_node = all_softmax[0]->inlinks.front();
    std::vector<Node*> output_node;
    std::string in_name =
        all_softmax[0]->stmt()->op_info()->Input("Input").front();
    std::vector<int> lod{0};
    bool all_used = false;
    int last_remain = used.size();
    int cur_remain = last_remain;
    while (all_used == false) {
      all_used = true;
      last_remain = cur_remain;
      for (int i = 0; i < used.size(); i++) {
        if (used[i] == false) {
          auto cur_lod =
              all_softmax[i]->stmt()->op_info()->GetAttr<std::vector<int>>(
                  "lod");
          if (cur_lod[0] == lod.back()) {
            lod.push_back(cur_lod[1]);
            out_names.push_back(
                all_softmax[i]->stmt()->op_info()->Output("Output").front());
            to_remove.insert(all_softmax[i]);
            output_node.push_back(all_softmax[i]->outlinks.front());
            used[i] = true;
            cur_remain = cur_remain - 1;
          } else if (cur_lod[0] < lod.back()) {
            LOG(FATAL) << "Invalid Lod :" << cur_lod[0];
          }
        }
        all_used = all_used & used[i];
      }
      if (cur_remain == last_remain) {
        LOG(FATAL) << "Invalid Lod in multi softmax";
      }
    }
    GraphSafeRemoveNodes(graph, to_remove);

    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__multi_softmax");
    op_desc.SetInput("Input", {in_name});
    op_desc.SetOutput("Output", out_names);
    op_desc.SetAttr<std::vector<int>>("lod", lod);
    std::string concat_output_name =
        "__xpu__multi_softmax_concat_output_" + in_name;
    CHECK(graph->RetrieveArgument(concat_output_name) == nullptr);
    auto* concat_output_node = graph->NewArgumentNode(concat_output_name);
    concat_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->NewTensor(concat_output_name);
    op_desc.SetOutput("ConcatOut", {concat_output_name});

    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    DirectedLink(new_op_node, concat_output_node);
    DirectedLink(input_node, new_op_node);
    for (Node* node : output_node) {
      DirectedLink(new_op_node, node);
    }
  }
};

}  // namespace fusion

class XPUMultiSliceSoftmaxFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUSingleSliceSoftmaxFuser single;
    single(graph.get());
    fusion::XPUMultiSliceSoftmaxFuser multi_fuser;
    multi_fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_softmax_fuse_pass,
                  paddle::lite::mir::XPUMultiSliceSoftmaxFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_softmax");
