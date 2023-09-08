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

/*-----------------------------------------------------*/
/* support xpu greater_than_cast_mul_fuse_pass         */
/*                in_Input --------------              */
/*                    |         |       |              */
/*                    |         |       |              */
/*              fill_any_like   |       |              */
/*                    |         |       |              */
/*                    |         |       |              */
/*                  scale      /       /               */
/*                    |       /       /                */
/*                    |      /       /                 */
/*              greater_than        /                  */
/*                    |            /                   */
/*                    |           /                    */
/*                  cast         /                     */
/*                    |         /                      */
/*                    |        /                       */
/*              elementwise_mul                        */
/*                    |                                */
/*                    |                                */
/*                out_Output                           */
/*-----------------------------------------------------*/

/*-----------------------------------------------------*/
/* After the pass apply:                               */
/*                in_Input                             */
/*                    |                                */
/*                    |                                */
/*          xpu_greater_than_filter                    */
/*                    |                                */
/*                    |                                */
/*                out_Output                           */
/*-----------------------------------------------------*/

class XPUGreaterThanCastMulFuser : public FuseBase {
 public:
  void BuildPattern() override {
    PMNode* input = nullptr;
    PMNode* fill_any_like = nullptr;
    PMNode* greater_than = nullptr;
    PMNode* elementwise_mul = nullptr;
    PMNode* fill_any_like_out = nullptr;
    PMNode* scale = nullptr;
    PMNode* scale_out = nullptr;
    PMNode* greater_than_out = nullptr;
    PMNode* cast = nullptr;
    PMNode* cast_out = nullptr;
    PMNode* elementwise_mul_out = nullptr;
    input = VarNode("input")
                ->assert_is_op_input("fill_any_like", "X")
                ->assert_is_op_input("greater_than", "X")
                ->assert_is_op_input("elementwise_mul", "X")
                ->AsInput();
    fill_any_like = OpNode("fill_any_like", "fill_any_like")->AsIntermediate();
    fill_any_like_out = VarNode("fill_any_like_out")
                            ->assert_is_op_output("fill_any_like", "Out")
                            ->assert_is_op_input("scale", "X")
                            ->AsIntermediate();
    scale = OpNode("scale", "scale")->AsIntermediate();
    scale_out = VarNode("scale_out")
                    ->assert_is_op_output("scale", "Out")
                    ->assert_is_op_input("greater_than", "Y")
                    ->AsIntermediate();
    greater_than = OpNode("greater_than", "greater_than")->AsIntermediate();
    greater_than_out = VarNode("greater_than_out")
                           ->assert_is_op_output("greater_than", "Out")
                           ->assert_is_op_input("cast", "X")
                           ->AsIntermediate();
    cast = OpNode("cast", "cast")->AsIntermediate();
    cast_out = VarNode("cast_out")
                   ->assert_is_op_output("cast", "Out")
                   ->assert_is_op_input("elementwise_mul", "Y")
                   ->AsIntermediate();
    elementwise_mul =
        OpNode("elementwise_mul", "elementwise_mul")->AsIntermediate();
    elementwise_mul_out = VarNode("elementwise_mul_out")
                              ->assert_is_op_output("elementwise_mul", "Out")
                              ->AsOutput();
    *input >> *fill_any_like >> *fill_any_like_out >> *scale >> *scale_out >>
        *greater_than >> *greater_than_out >> *cast >> *cast_out >>
        *elementwise_mul >> *elementwise_mul_out;
    *input >> *greater_than;
    *input >> *elementwise_mul;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__greater_than_filter");
    op_desc.SetInput("X", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("elementwise_mul_out")->arg()->name});

    auto* scope = matched.at("fill_any_like")->stmt()->op()->scope();
    auto input_name = matched.at("input")->arg()->name;

    float value_filled = 1.0f;
    value_filled =
        matched.at("fill_any_like")->stmt()->op_info()->GetAttr<float>("value");
    float scale_val = 0.f;
    scale_val = matched.at("scale")->stmt()->op_info()->GetAttr<float>("scale");
    scale_val *= value_filled;
    op_desc.SetAttr<float>("scale", scale_val);
    auto& valid_places =
        matched.at("fill_any_like")->stmt()->op()->valid_places();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    DirectedLink(matched.at("input"), new_op_node);
    DirectedLink(new_op_node, matched.at("elementwise_mul_out"));
  }
};

}  // namespace fusion

class XPUGreaterThanCastMulFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUGreaterThanCastMulFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__greater_than_cast_mul_fuse_pass,
                  paddle::lite::mir::XPUGreaterThanCastMulFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__greater_than_filter");
