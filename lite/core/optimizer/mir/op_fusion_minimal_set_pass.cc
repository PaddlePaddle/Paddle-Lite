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

#include "lite/core/optimizer/mir/op_fusion_minimal_set_pass.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

#define SKIP_DELETE_INTERMEDIATE_NODES \
  for (auto& item : key2nodes_) {      \
    if (&item == &matched) {           \
      item.clear();                    \
    }                                  \
  }

class IdentityScaleEliminator : public FuseBase {
 public:
  void BuildPattern() override {
    // Create the pattern nodes.
    auto prev_node = OpNode("prev")
                         ->assert_is_not_op_type("conditional_block")
                         ->assert_is_not_op_type("while")
                         ->assert_is_not_op_type("scale");
    auto scale_x_node =
        VarNode("scale_x")->assert_is_op_input("scale", "X")->AsIntermediate();
    auto scale_node =
        OpNode("scale", "scale")
            ->assert_node_satisfied([](const Node* node) -> bool {
              auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
              auto scale = op_desc.GetAttr<float>("scale");
              auto bias = op_desc.GetAttr<float>("bias");
              bool with_act = (op_desc.HasAttr("with_act") &&
                               op_desc.GetAttr<bool>("with_act")) ||
                              op_desc.HasAttr("fuse_relu");
              return std::fabs(scale - 1.0f) <= 1e-5f &&
                     std::fabs(bias) <= 1e-5f && !with_act;
            });
    auto scale_out_node =
        VarNode("scale_out")->assert_is_op_output("scale", "Out");
    // Create the topological connections for the above pattern nodes.
    *prev_node >> *scale_x_node >> *scale_node >> *scale_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto prev_node = matched.at("prev");
    auto prev_op = prev_node->stmt()->op();
    auto& valid_places = prev_op->valid_places();
    auto scale_node = matched.at("scale");
    auto scale_x_node = matched.at("scale_x");
    auto scale_x_name = scale_x_node->arg()->name;
    auto scale_out_node = matched.at("scale_out");
    auto scale_out_name = scale_out_node->arg()->name;
    // Remove the scale op and link the previous op to the output
    auto prev_desc = *prev_node->stmt()->op_info();
    prev_desc.UpdateAllOutputs(scale_x_name, scale_out_name);
    prev_node->stmt()->ResetOp(prev_desc, valid_places);
    GraphSafeRemoveNodes(graph, {scale_node});
    IR_NODE_LINK_TO(prev_node, scale_out_node);
  }
};

class IdentityDropoutEliminator : public FuseBase {
 public:
  static bool DropoutIsTest(const Node* x) {
    if (x && x->IsStmt()) {
      auto* op_info = x->stmt()->op_info();
      if (op_info->HasAttr("is_test")) {
        auto attr_type = op_info->GetAttrType("is_test");
        if (attr_type == paddle::lite::OpDescAPI::AttrType::INT &&
            op_info->GetAttr<int>("is_test") == 1) {
          return true;
        } else if (attr_type == paddle::lite::OpDescAPI::AttrType::BOOLEAN &&
                   op_info->GetAttr<bool>("is_test")) {
          return true;
        }
      }
    }
    return false;
  }

  void BuildPattern() override {
    // the previous op's output need updat
    auto* pre_op = OpNode("preop")->assert_is_not_op_type("conditional_block");
    // TODO(Superjomn) check has only one output
    auto* x = VarNode("x")->assert_is_op_input("dropout", "X");
    auto* dropout_op =
        OpNode("dropout", "dropout")
            ->assert_node_satisfied(IdentityDropoutEliminator::DropoutIsTest)
            ->assert_op_attr<std::string>("dropout_implementation",
                                          "upscale_in_train");
    auto* out = VarNode("out")->assert_is_op_output("dropout", "Out");
    auto* mask = VarNode("mask")->assert_is_op_output("dropout", "Mask");

    *pre_op >> *x >> *dropout_op >> *out;
    *dropout_op >> *mask;

    // The pre_op will be eliminated, and a new output-updated op will insert.
    x->AsIntermediate();  // x is pre_op's output, need to update
    dropout_op->AsIntermediate();
    mask->AsIntermediate();
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& pre_op = matched.at("preop")->AsStmt();
    auto op_info = *pre_op.op_info();

    op_info.UpdateAllOutputs(matched.at("x")->AsArg().name,
                             matched.at("out")->AsArg().name);
    pre_op.ResetOp(op_info, graph->valid_places());

    IR_NODE_LINK_TO(matched.at("preop"), matched.at("out"));
  }
};

void ApplyIdentityScaleEliminator(SSAGraph* graph) {
  IdentityScaleEliminator fuser;
  fuser(graph);
}

void ApplyIdentityDropoutEliminator(SSAGraph* graph) {
  IdentityDropoutEliminator fuser;
  fuser(graph);
}

void OpFusionMinimalSetPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  ApplyIdentityScaleEliminator(graph.get());
  ApplyIdentityDropoutEliminator(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(op_fusion_minimal_set_pass,
                  paddle::lite::mir::OpFusionMinimalSetPass)
    .BindTargets({TARGET(kNNAdapter)});
