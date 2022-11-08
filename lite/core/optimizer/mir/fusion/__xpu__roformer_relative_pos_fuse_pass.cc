// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

/* support xpu roformer relative pos                    */
/*                in_Input ---------------              */
/*                    |    \             |              */
/*                    |     \            |              */
/*                  split    shape       |              */
/*                 /  |        \         |              */
/*                /   |         \        |              */
/*               |  scale      slice     |              */
/*                \   |         /  \     |              */
/*                 \  |        /    \    |              */
/*                  concat  slice  slice |              */
/*                    |      /        \  |              */
/*                    |     /          \ |              */
/*             elementwise_mul     elementwise_mul      */
/*                    |           /                     */
/*                    |          /                      */
/*                elementwise_add                       */
/*                    |                                 */
/*                    |                                 */
/*                out_Output                            */
/*-------------------------------------------*/
/* After the pass apply:                     */
/*                in_Input                   */
/*          cos_emb   |   sin_emb            */
/*                 \  |  /                   */
/*          xpu_roformer_relative            */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/

class XPURoformerRelativePosFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("split", "X")
                      ->assert_is_op_input("elementwise_mul", "X")
                      ->assert_is_op_input("shape", "Input")
                      ->AsInput();
    auto* split =
        OpNode("split", "split")
            ->assert_op_attr<int32_t>("axis", 3)
            ->assert_op_attr<int32_t>("num", 2)  // do we really need it
            ->AsIntermediate();
    auto* split_out0 = VarNode("split_out0")
                           ->assert_is_op_nth_input("concat", "X", 1)
                           ->assert_is_op_nth_output("split", "Out", 0)
                           ->AsIntermediate();
    auto* split_out1 = VarNode("split_out1")
                           ->assert_is_op_input("scale", "X")
                           ->assert_is_op_nth_output("split", "Out", 1)
                           ->AsIntermediate();
    auto* scale =
        OpNode("scale", "scale")
            ->assert_op_attr_satisfied<float>(
                "scale",
                [](float attr) { return (std::fabs(attr + 1.0) < 1e-5); })
            ->AsIntermediate();
    auto* scale_out = VarNode("scale_out")
                          ->assert_is_op_input("concat", "X")
                          ->assert_is_op_output("scale", "Out")
                          ->AsIntermediate();
    auto* concat = OpNode("concat", "concat")->AsIntermediate();
    auto* concat_out = VarNode("concat_out")
                           ->assert_is_op_input("elementwise_mul", "X")
                           ->assert_is_op_output("concat", "Out")
                           ->AsIntermediate();
    auto* shape = OpNode("shape", "shape")->AsIntermediate();
    auto* shape_out = VarNode("shape_out")
                          ->assert_is_op_input("slice", "Input")
                          ->assert_is_op_output("shape", "Out")
                          ->AsIntermediate();
    auto* slice1 = OpNode("slice1", "slice")->AsIntermediate();
    auto* slice1_out = VarNode("slice1_out")
                           ->assert_is_op_input("slice", "EndsTensorList")
                           ->assert_is_op_output("slice", "Out")
                           ->AsIntermediate();
    auto* sin_emb =
        VarNode("sin_emb")->assert_is_op_input("slice", "Input")->AsInput();
    auto* cos_emb =
        VarNode("cos_emb")->assert_is_op_input("slice", "Input")->AsInput();
    auto* slice_sin = OpNode("slice_sin", "slice")->AsIntermediate();
    auto* slice_sin_out = VarNode("slice_sin_out")
                              ->assert_is_op_input("elementwise_mul", "Y")
                              ->assert_is_op_output("slice", "Out")
                              ->AsIntermediate();
    auto* ew_mul_sin =
        OpNode("ew_mul_sin", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_sin_out = VarNode("ew_mul_sin_out")
                               ->assert_is_op_input("elementwise_add", "Y")
                               ->assert_is_op_output("elementwise_mul", "Out")
                               ->AsIntermediate();
    auto* ew_add = OpNode("ew_add", "elementwise_add")->AsIntermediate();
    auto* ew_add_out = VarNode("ew_add_out")
                           ->assert_is_op_output("elementwise_add", "Out")
                           ->AsOutput();
    auto* slice_cos = OpNode("slice_cos", "slice")->AsIntermediate();
    auto* slice_cos_out = VarNode("slice_cos_out")
                              ->assert_is_op_input("elementwise_mul", "Y")
                              ->assert_is_op_output("slice", "Out")
                              ->AsIntermediate();
    auto* ew_mul_cos =
        OpNode("ew_mul_cos", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_cos_out = VarNode("ew_mul_cos_out")
                               ->assert_is_op_input("elementwise_add", "X")
                               ->assert_is_op_output("elementwise_mul", "Out")
                               ->AsIntermediate();
    *input >> *split >> *split_out1 >> *scale >> *scale_out >> *concat >>
        *concat_out >> *ew_mul_sin >> *ew_mul_sin_out >> *ew_add >> *ew_add_out;
    *input >> *ew_mul_cos >> *ew_mul_cos_out >> *ew_add;
    *input >> *shape >> *shape_out >> *slice1 >> *slice1_out >> *slice_sin >>
        *slice_sin_out >> *ew_mul_sin;
    *slice1_out >> *slice_cos >> *slice_cos_out >> *ew_mul_cos;
    *sin_emb >> *slice_sin;
    *cos_emb >> *slice_cos;
    *split >> *split_out0 >> *concat;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__roformer_relative_embedding");
    // use "X", be consistent with target_op_type_ in multiencoder pass
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetInput("CosEmbbeding", {matched.at("cos_emb")->arg()->name});
    op_desc.SetInput("SinEmbbeding", {matched.at("sin_emb")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("ew_add_out")->arg()->name});
    auto* scope = matched.at("split")->stmt()->op()->scope();

    auto cos_emb_name = matched.at("cos_emb")->arg()->name;
    auto cos_emb_shape = scope->FindMutableTensor(cos_emb_name)->dims();
    auto sin_emb_name = matched.at("sin_emb")->arg()->name;
    auto sin_emb_shape = scope->FindMutableTensor(sin_emb_name)->dims();
    CHECK_EQ(cos_emb_shape.size(), 4) << cos_emb_shape.size();
    CHECK_GT(cos_emb_shape[2], 0) << cos_emb_shape[2];
    CHECK_EQ(sin_emb_shape.size(), 4) << sin_emb_shape.size();
    for (int i = 0; i < sin_emb_shape.size(); ++i) {
      CHECK_EQ(sin_emb_shape[i], cos_emb_shape[i])
          << i << " th dim: " << sin_emb_shape[i] << ", " << cos_emb_shape[i];
    }
    op_desc.SetAttr<int>("max_pos_len", cos_emb_shape[2]);

    auto& valid_places = matched.at("split")->stmt()->op()->valid_places();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(matched.at("cos_emb"), new_op_node);
    DirectedLink(matched.at("sin_emb"), new_op_node);
    DirectedLink(new_op_node, matched.at("ew_add_out"));
  }
};

}  // namespace fusion

class XPURoformerRelativePosFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPURoformerRelativePosFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__roformer_relative_pos_fuse_pass,
                  paddle::lite::mir::XPURoformerRelativePosFusePass)
    .BindTargets({TARGET(kXPU)});
