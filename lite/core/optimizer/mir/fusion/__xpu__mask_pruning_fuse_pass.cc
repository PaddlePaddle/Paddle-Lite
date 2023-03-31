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

class XPUMaskPruningFuser : public FuseBase {
 public:
  void BuildPattern() override {
    PMNode* old_mask = VarNode("old_mask")->AsInput();
    PMNode* new_mask = VarNode("new_mask")->AsOutput();

    PMNode* shape = OpNode("shape", "shape")->AsIntermediate();
    PMNode* shape_out = VarNode("shape_out")->AsIntermediate();
    PMNode* slice1 = OpNode("slice1", "slice")->AsIntermediate();
    PMNode* slice1_out = VarNode("slice1_out")->AsIntermediate();
    PMNode* tile = OpNode("tile", "tile")->AsIntermediate();
    *old_mask >> *shape >> *shape_out >> *slice1 >> *slice1_out >> *tile >>
        *new_mask;

    PMNode* slice2 = OpNode("slice2", "slice")->AsIntermediate();
    PMNode* slice2_out = VarNode("slice2_out")->AsIntermediate();
    PMNode* cast1 = OpNode("cast1", "cast")->AsIntermediate();
    PMNode* cast1_out = VarNode("cast1_out")->AsIntermediate();
    PMNode* greater_equal =
        OpNode("greater_equal", "greater_equal")->AsIntermediate();
    *old_mask >> *slice2 >> *slice2_out >> *cast1 >> *cast1_out >>
        *greater_equal;

    PMNode* constant1 = OpNode("constant1", "fill_constant")->AsIntermediate();
    PMNode* constant1_out = VarNode("constant1_out")->AsIntermediate();

    PMNode* greater_equal_out = VarNode("greater_equal_out")->AsIntermediate();
    PMNode* cast2 = OpNode("cast2", "cast")->AsIntermediate();
    PMNode* cast2_out = VarNode("cast2_out")->AsIntermediate();
    PMNode* reduce_sum = OpNode("reduce_sum", "reduce_sum")->AsIntermediate();
    PMNode* reduce_sum_out = VarNode("reduce_sum_out")->AsIntermediate();
    PMNode* cast3 = OpNode("cast3", "cast")->AsIntermediate();
    PMNode* cast3_out = VarNode("cast3_out")->AsIntermediate();
    PMNode* scale1 = OpNode("scale1", "scale")->AsIntermediate();
    PMNode* scale1_out = VarNode("scale1_out")->AsIntermediate();
    PMNode* clip = OpNode("clip", "clip")->AsIntermediate();
    PMNode* clip_out = VarNode("clip_out")->AsIntermediate();

    PMNode* cast4 = OpNode("cast4", "cast")->AsIntermediate();
    PMNode* cast4_out = VarNode("cast4_out")->AsIntermediate();
    PMNode* sequence_mask =
        OpNode("sequence_mask", "sequence_mask")->AsIntermediate();
    PMNode* sequence_mask_out = VarNode("sequence_mask_out")->AsIntermediate();
    PMNode* unsqueeze1 = OpNode("unsqueeze1", "unsqueeze2")->AsIntermediate();
    PMNode* unsqueeze1_out = VarNode("unsqueeze1_out")->AsIntermediate();
    PMNode* unsqueeze1_xshape = VarNode("unsqueeze1_xshape")->AsIntermediate();
    *unsqueeze1 >> *unsqueeze1_xshape;

    PMNode* matmul = OpNode("matmul", "matmul_v2")->AsIntermediate();
    PMNode* matmul_out = VarNode("matmul_out")->AsIntermediate();
    PMNode* scale2 = OpNode("scale2", "scale")->AsIntermediate();
    PMNode* scale2_out = VarNode("scale2_out")->AsIntermediate();
    PMNode* unsqueeze2 = OpNode("unsqueeze2", "unsqueeze2")->AsIntermediate();
    PMNode* unsqueeze2_out = VarNode("unsqueeze2_out")->AsIntermediate();
    PMNode* unsqueeze2_xshape = VarNode("unsqueeze2_xshape")->AsIntermediate();
    *unsqueeze2 >> *unsqueeze2_xshape;
    *constant1 >> *constant1_out >> *greater_equal >> *greater_equal_out >>
        *cast2 >> *cast2_out >> *reduce_sum >> *reduce_sum_out >> *cast3 >>
        *cast3_out >> *scale1 >> *scale1_out >> *clip >> *clip_out >> *cast4 >>
        *cast4_out >> *sequence_mask >> *sequence_mask_out >> *unsqueeze1 >>
        *unsqueeze1_out >> *matmul >> *matmul_out >> *scale2 >> *scale2_out >>
        *unsqueeze2 >> *unsqueeze2_out >> *tile;

    PMNode* constant2 = OpNode("constant2", "fill_constant")->AsIntermediate();
    PMNode* constant2_out = VarNode("constant2_out")->AsIntermediate();
    *constant2 >> *constant2_out >> *tile;
    PMNode* constant3 = OpNode("constant3", "fill_constant")->AsIntermediate();
    PMNode* constant3_out = VarNode("constant3_out")->AsIntermediate();
    *constant3 >> *constant3_out >> *tile;
    PMNode* constant4 = OpNode("constant4", "fill_constant")->AsIntermediate();
    PMNode* constant4_out = VarNode("constant4_out")->AsIntermediate();
    *constant4 >> *constant4_out >> *tile;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_mask_pruning_fuse_pass______";
    auto* scope = matched.at("shape")->stmt()->op()->scope();
    auto valid_places = matched.at("shape")->stmt()->op()->valid_places();
    cpp::OpDesc mask_pruning_op_desc;
    mask_pruning_op_desc.mutable_inputs()->clear();
    mask_pruning_op_desc.mutable_outputs()->clear();
    mask_pruning_op_desc.SetType("__xpu__clip_mask");
    mask_pruning_op_desc.SetInput("old_mask",
                                  {matched.at("old_mask")->arg()->name});
    mask_pruning_op_desc.SetOutput("new_mask",
                                   {matched.at("new_mask")->arg()->name});

    float keep_ratio =
        matched.at("scale1")->stmt()->op_info()->GetAttr<float>("scale");
    mask_pruning_op_desc.SetAttr<float>("keep_ratio", keep_ratio);

    auto mask_pruning_op = LiteOpRegistry::Global().Create("__xpu__clip_mask");
    mask_pruning_op->Attach(mask_pruning_op_desc, scope);
    mask_pruning_op->SetValidPlaces(valid_places);
    auto* mask_pruning_op_node =
        graph->GraphCreateInstructNode(mask_pruning_op, valid_places);
    DirectedLink(matched.at("old_mask"), mask_pruning_op_node);
    DirectedLink(mask_pruning_op_node, matched.at("new_mask"));
  }
};

}  // namespace fusion

class XPUMaskPruningFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUMaskPruningFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__mask_pruning_fuse_pass,
                  paddle::lite::mir::XPUMaskPruningFusePass)
    .BindTargets({TARGET(kXPU)});
