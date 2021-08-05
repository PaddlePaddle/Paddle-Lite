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

class XPUGenerateSequenceFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input =
        VarNode("input")->assert_is_op_input("fill_any_like", "X")->AsInput();
    auto* fill = OpNode("fill_like", "fill_any_like")->AsIntermediate();
    auto* out_fill = VarNode("fill_like_out")
                         ->assert_is_op_output("fill_any_like", "Out")
                         ->assert_is_op_input("cumsum", "X")
                         ->assert_is_op_input("elementwise_sub", "Y")
                         ->AsIntermediate();
    auto* cumsum = OpNode("cumsum", "cumsum")
                       ->assert_op_attr<bool>("exclusive", false)
                       ->assert_op_attr<bool>("reverse", false)
                       ->AsIntermediate();
    auto* out_cumsum = VarNode("cumsum_out")
                           ->assert_is_op_output("cumsum", "Out")
                           ->assert_is_op_input("elementwise_sub", "X")
                           ->AsIntermediate();
    auto* ew_sub = OpNode("ew_sub", "elementwise_sub")
                       ->assert_op_attr<int>("axis", -1)
                       ->AsIntermediate();
    auto* out = VarNode("output")
                    ->assert_is_op_output("elementwise_sub", "Out")
                    ->AsOutput();

    *input >> *fill >> *out_fill;
    *out_fill >> *cumsum >> *out_cumsum >> *ew_sub;
    *out_fill >> *ew_sub;
    *ew_sub >> *out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__generate_sequence");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
    int dtype =
        matched.at("fill_like")->stmt()->op_info()->GetAttr<int>("dtype");
    float value =
        matched.at("fill_like")->stmt()->op_info()->GetAttr<float>("value");
    bool flatten =
        matched.at("cumsum")->stmt()->op_info()->GetAttr<bool>("flatten");
    int axis = matched.at("cumsum")->stmt()->op_info()->GetAttr<int>("axis");
    op_desc.SetAttr("axis", axis);
    op_desc.SetAttr("flatten", flatten);
    op_desc.SetAttr("value", value);
    op_desc.SetAttr("dtype", dtype);

    auto fill_like = matched.at("fill_like")->stmt()->op();
    auto* scope = fill_like->scope();
    auto& valid_places = fill_like->valid_places();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("output"));
  }
};

}  // namespace fusion

class XPUGenerateSequenceFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUGenerateSequenceFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__generate_sequence_fuse_pass,
                  paddle::lite::mir::XPUGenerateSequenceFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__generate_sequence");
