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
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUSoftmaxTopkFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("softmax", "X")->AsInput();
    auto* softmax = OpNode("softmax", "softmax")
                        ->assert_op_attr<int>("axis", -1)
                        ->AsIntermediate();
    auto* softmax_out = VarNode("softmax_out")
                            ->assert_is_op_output("softmax", "Out")
                            ->assert_is_op_input("top_k", "X")
                            ->AsIntermediate();
    auto* top_k = OpNode("top_k", "top_k")->AsIntermediate();
    auto* indices =
        VarNode("indices")->assert_is_op_output("top_k", "Indices")->AsOutput();
    auto* out = VarNode("out")->assert_is_op_output("top_k", "Out")->AsOutput();

    *input >> *softmax >> *softmax_out >> *top_k;
    *top_k >> *indices;
    *top_k >> *out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__softmax_topk");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("out")->arg()->name});
    op_desc.SetOutput("Indices", {matched.at("indices")->arg()->name});
    int axis = matched.at("softmax")->stmt()->op_info()->GetAttr<int>("axis");
    int k = matched.at("top_k")->stmt()->op_info()->GetAttr<int>("k");
    op_desc.SetAttr("axis", axis);
    op_desc.SetAttr("k", k);

    auto softmax = matched.at("softmax")->stmt()->op();
    auto* scope = softmax->scope();
    auto& valid_places = softmax->valid_places();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("out"));
    IR_NODE_LINK_TO(new_op_node, matched.at("indices"));
  }
};

}  // namespace fusion

class XPUSoftmaxTopkFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUSoftmaxTopkFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__softmax_topk_fuse_pass,
                  paddle::lite::mir::XPUSoftmaxTopkFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__softmax_topk");
