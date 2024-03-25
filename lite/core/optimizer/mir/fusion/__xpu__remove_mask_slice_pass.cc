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

/* support token-slcie feature of adaptive seq len for aurora long-text models
 */
/*-------------------------------------------*/
/* Before the pass apply:                    */
/*                                           */
/*                 in_Mask                   */
/*                    |                      */
/*                    |                      */
/*                  matmul                   */
/*                    |                      */
/*                    |                      */
/*                  scale                    */
/*                    |                      */
/*                    |                      */
/*                  stack                    */
/*                 /  |   \                  */
/*                /   |    \                 */
/*             slice ...  sclie              */
/*                \   |      \               */
/*                 \  |       \ (悬空)        */
/*              xpu_encoder                  */
/*-------------------------------------------*/
/* After the pass apply:                     */
/*                                           */
/*                 in_Mask                   */
/*                    |                      */
/*                    |                      */
/*                  matmul                   */
/*                    |                      */
/*                    |                      */
/*                  scale                    */
/*                    |                      */
/*                    |                      */
/*                  stack                    */
/*                    |                      */
/*                    |                      */
/*                xpu_encoder                */
/*-------------------------------------------*/

class XPURemoveMaskSlice : public FuseBase {
 public:
  explicit XPURemoveMaskSlice(bool with_dangling_slice = false)
      : with_dangling_slice_(with_dangling_slice) {}
  void BuildPattern() override {
    PMNode* mask = nullptr;
    PMNode* matmul = nullptr;
    PMNode* matmul_out = nullptr;
    PMNode* scale = nullptr;
    PMNode* scale_out = nullptr;
    PMNode* stack = nullptr;
    PMNode* stack_out = nullptr;
    PMNode* slice = nullptr;
    PMNode* slice_out = nullptr;
    PMNode* xpu_encoder = nullptr;

    mask = VarNode("mask")
               ->assert_is_op_input("matmul", "X")
               ->assert_is_op_input("matmul", "Y");
    matmul = OpNode("matmul", "matmul");
    matmul_out = VarNode("matmul_out")
                     ->assert_is_op_input("scale", "X")
                     ->assert_is_op_output("matmul", "Out");
    scale = OpNode("scale", "scale");
    scale_out = VarNode("scale_out")
                    ->assert_is_op_input("stack", "X")
                    ->assert_is_op_output("scale", "Out");
    stack = OpNode("stack", "stack");
    stack_out = VarNode("stack_out")
                    ->assert_is_op_input("slice", "Input")
                    ->assert_is_op_output("stack", "Y");
    slice = OpNode("slice", "slice")->AsIntermediate();
    slice_out = VarNode("slice_out")
                    ->assert_is_op_output("slice", "Out")
                    ->AsIntermediate();
    if (!with_dangling_slice_) {
      xpu_encoder = OpNode("xpu_encoder", "__xpu__multi_encoder")
                        ->assert_op_attr<bool>("adaptive_seqlen", true);
    }
    matmul->LinksFrom({mask});
    matmul->LinksTo({matmul_out});
    scale->LinksFrom({matmul_out});
    scale->LinksTo({scale_out});
    stack->LinksFrom({scale_out});
    stack->LinksTo({stack_out});
    slice->LinksFrom({stack_out});
    slice->LinksTo({slice_out});
    if (!with_dangling_slice_) {
      xpu_encoder->LinksFrom({slice_out});
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    if (!with_dangling_slice_) {
      auto* encoder_instruct = matched.at("xpu_encoder")->stmt();
      auto encoder_op_desc = *encoder_instruct->mutable_op_info();
      auto encoder_op = encoder_instruct->op();

      encoder_op_desc.SetInput("Mask", {matched.at("stack_out")->arg()->name});
      encoder_instruct->ResetOp(encoder_op_desc, encoder_op->valid_places());
      DirectedLink(matched.at("stack_out"), matched.at("xpu_encoder"));
    }
  }

 private:
  bool with_dangling_slice_;
};

}  // namespace fusion

class XPURemoveMaskSlicePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto with_dangling_slice : {true, false}) {
      fusion::XPURemoveMaskSlice fuser(with_dangling_slice);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__remove_mask_slice_pass,
                  paddle::lite::mir::XPURemoveMaskSlicePass)
    .BindTargets({TARGET(kXPU)});
