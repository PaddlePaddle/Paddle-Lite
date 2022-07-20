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

void ApplyIdentityScaleEliminator(SSAGraph* graph) {
  IdentityScaleEliminator fuser;
  fuser(graph);
}

void OpFusionMinimalSetPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  ApplyIdentityScaleEliminator(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(op_fusion_minimal_set_pass,
                  paddle::lite::mir::OpFusionMinimalSetPass)
    .BindTargets({TARGET(kNNAdapter)});
