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

class XPUTokenScatterFuser : public FuseBase {
 public:
  explicit XPUTokenScatterFuser(bool adaptive_seqlen = true)
      : adaptive_seqlen_(adaptive_seqlen) {}
  void BuildPattern() override {
    PMNode* slimmed_shape = OpNode("slimmed_shape", "shape")->AsIntermediate();
    PMNode* slimmed_shape_out = VarNode("slimmed_shape_out")
                                    ->assert_is_op_input("slice", "Input")
                                    ->assert_is_op_output("shape", "Out")
                                    ->AsIntermediate();
    PMNode* slimmed_slice0 =
        OpNode("slimmed_slice0", "slice")->AsIntermediate();
    PMNode* slimmed_slice0_out = VarNode("slimmed_slice0_out")
                                     ->assert_is_op_input("scale", "X")
                                     ->assert_is_op_output("slice", "Out")
                                     ->AsIntermediate();
    PMNode* slimmed_slice1 =
        OpNode("slimmed_slice1", "slice")->AsIntermediate();
    PMNode* slimmed_slice1_out = VarNode("slimmed_slice1_out")
                                     ->assert_is_op_output("slice", "Out")
                                     ->AsIntermediate();

    PMNode* x_shape = OpNode("x_shape", "shape")->AsIntermediate();
    PMNode* x_shape_out = VarNode("x_shape_out")
                              ->assert_is_op_input("slice", "Input")
                              ->assert_is_op_output("shape", "Out")
                              ->AsIntermediate();
    PMNode* x_slice = OpNode("x_slice", "slice")->AsIntermediate();
    PMNode* x_slice_out = VarNode("x_slice_out")
                              ->assert_is_op_output("slice", "Out")
                              ->AsIntermediate();

    PMNode* scale0 = OpNode("scale0", "scale")->AsIntermediate();
    PMNode* scale0_out = VarNode("scale0_out")->AsIntermediate();
    PMNode* ew_mul = OpNode("ew_mul", "elementwise_mul")->AsIntermediate();
    PMNode* ew_mul_out = VarNode("ew_mul_out")->AsIntermediate();
    PMNode* scale1 = OpNode("scale1", "scale")->AsIntermediate();
    PMNode* scale1_out = VarNode("scale1_out")->AsIntermediate();
    PMNode* cast0 = OpNode("cast0", "cast")->AsIntermediate();
    PMNode* cast0_out = VarNode("cast0_out")
                            ->assert_is_op_output("cast", "Out")
                            ->AsIntermediate();
    PMNode* cast1 = OpNode("cast1", "cast")->AsIntermediate();
    PMNode* cast1_out = VarNode("cast1_out")
                            ->assert_is_op_output("cast", "Out")
                            ->AsIntermediate();

    PMNode* constant0 = OpNode("constant0", "fill_constant")->AsIntermediate();
    PMNode* constant0_out = VarNode("constant0_out")->AsIntermediate();
    PMNode* constant1 = OpNode("constant1", "fill_constant")->AsIntermediate();
    PMNode* constant1_out = VarNode("constant1_out")->AsIntermediate();

    PMNode* range = OpNode("range", "range")->AsIntermediate();
    PMNode* range_out = VarNode("range_out")->AsIntermediate();
    PMNode* unsqueeze = OpNode("unsqueeze", "unsqueeze2")->AsIntermediate();
    PMNode* unsqueeze_out = VarNode("unsqueeze_out")->AsIntermediate();
    PMNode* unsqueeze_xshape = VarNode("unsqueeze_xshape")->AsIntermediate();
    PMNode* tile = OpNode("tile", "tile")->AsIntermediate();
    PMNode* tile_out = VarNode("tile_out")->AsIntermediate();

    PMNode* ew_add = OpNode("ew_add", "elementwise_add")->AsIntermediate();
    PMNode* ew_add_out = VarNode("ew_add_out")->AsIntermediate();

    PMNode* reshape0 = OpNode("reshape0", "reshape2")->AsIntermediate();
    PMNode* reshape0_out = VarNode("reshape0_out")->AsIntermediate();
    PMNode* reshape0_xshape = VarNode("reshape0_xshape")->AsIntermediate();
    PMNode* reshape1 = OpNode("reshape1", "reshape2")->AsIntermediate();
    PMNode* reshape1_out = VarNode("reshape1_out")->AsIntermediate();
    PMNode* reshape1_xshape = VarNode("reshape1_xshape")->AsIntermediate();
    PMNode* reshape2 = OpNode("reshape2", "reshape2")->AsIntermediate();
    PMNode* reshape2_out = VarNode("reshape2_out")->AsOutput();
    PMNode* reshape2_xshape = VarNode("reshape2_xshape")->AsIntermediate();
    PMNode* scatter = OpNode("scatter", "scatter")->AsIntermediate();
    PMNode* scatter_out = VarNode("scatter_out")->AsIntermediate();

    PMNode* encoder1 = OpNode("encoder1", "__xpu__encoder");

    PMNode* CLSInds = VarNode("CLSInds");
    PMNode* X = VarNode("X");

    PMNode* encoder2 = OpNode("encoder2", "__xpu__encoder");
    PMNode* encoder2_out = VarNode("encoder2_out");

    PMNode* seq_lod = nullptr;
    PMNode* pad_seq_len = nullptr;
    if (adaptive_seqlen_) {
      seq_lod = VarNode("seq_lod")
                    ->assert_is_op_input("__xpu__encoder", "SeqLod")
                    ->AsInput();
      pad_seq_len = VarNode("pad_seq_len")
                        ->assert_is_op_input("__xpu__encoder", "PadSeqLen")
                        ->AsInput();
      encoder1->assert_op_attr<bool>("adaptive_seqlen", true);
      encoder2->assert_op_attr<bool>("adaptive_seqlen", true);
    } else {
      encoder2->assert_op_attr<bool>("adaptive_seqlen", false);
    }

    PMNode* encoder_shape = OpNode("encoder_shape", "shape")->AsIntermediate();
    PMNode* encoder_shape_out = VarNode("encoder_shape_out")->AsIntermediate();
    PMNode* encoder_slice0 =
        OpNode("encoder_slice0", "slice")->AsIntermediate();
    PMNode* encoder_slice0_out =
        VarNode("encoder_slice0_out")->AsIntermediate();
    PMNode* encoder_slice1 =
        OpNode("encoder_slice1", "slice")->AsIntermediate();
    PMNode* encoder_slice1_out =
        VarNode("encoder_slice1_out")->AsIntermediate();

    PMNode* encoder_constant0 =
        OpNode("encoder_constant0", "fill_constant")->AsIntermediate();
    PMNode* encoder_constant0_out =
        VarNode("encoder_constant0_out")->AsIntermediate();
    PMNode* encoder_constant1 =
        OpNode("encoder_constant1", "fill_constant")->AsIntermediate();
    PMNode* encoder_constant1_out =
        VarNode("encoder_constant1_out")->AsIntermediate();

    PMNode* encoder_reshape0 =
        OpNode("encoder_reshape0", "reshape2")->AsIntermediate();
    PMNode* encoder_reshape0_out =
        VarNode("encoder_reshape0_out")->AsIntermediate();
    PMNode* encoder_reshape0_xshape =
        VarNode("encoder_reshape0_xshape")->AsIntermediate();
    PMNode* encoder_reshape1 =
        OpNode("encoder_reshape1", "reshape2")->AsIntermediate();
    PMNode* encoder_reshape1_out =
        VarNode("encoder_reshape1_out")->AsIntermediate();
    PMNode* encoder_reshape1_xshape =
        VarNode("encoder_reshape1_xshape")->AsIntermediate();

    encoder1->LinksTo({CLSInds, X});
    slimmed_shape->LinksFrom({CLSInds}).LinksTo({slimmed_shape_out});
    slimmed_slice0->LinksFrom({slimmed_shape_out})
        .LinksTo({slimmed_slice0_out});
    slimmed_slice1->LinksFrom({slimmed_shape_out})
        .LinksTo({slimmed_slice1_out});
    x_shape->LinksFrom({X}).LinksTo({x_shape_out});
    x_slice->LinksFrom({x_shape_out}).LinksTo({x_slice_out});
    scale0->LinksFrom({slimmed_slice0_out}).LinksTo({scale0_out});
    ew_mul->LinksFrom({x_slice_out, scale0_out}).LinksTo({ew_mul_out});
    scale1->LinksFrom({ew_mul_out}).LinksTo({scale1_out});
    cast0->LinksFrom({scale1_out}).LinksTo({cast0_out});
    cast1->LinksFrom({x_slice_out}).LinksTo({cast1_out});
    constant0->LinksTo({constant0_out});
    constant1->LinksTo({constant1_out});
    range->LinksFrom({constant0_out, cast0_out, cast1_out})
        .LinksTo({range_out});
    unsqueeze->LinksFrom({range_out})
        .LinksTo({unsqueeze_out, unsqueeze_xshape});
    tile->LinksFrom({slimmed_slice1_out, unsqueeze_out, constant1_out})
        .LinksTo({tile_out});
    ew_add->LinksFrom({tile_out, CLSInds}).LinksTo({ew_add_out});
    reshape0->LinksFrom({ew_add_out}).LinksTo({reshape0_out, reshape0_xshape});
    reshape1->LinksFrom({X}).LinksTo({reshape1_out, reshape1_xshape});
    reshape2
        ->LinksFrom({scatter_out,
                     slimmed_slice0_out,
                     x_slice_out,
                     encoder_constant1_out})
        .LinksTo({reshape2_out, reshape2_xshape});
    scatter->LinksFrom({reshape0_out, reshape1_out, encoder_reshape1_out})
        .LinksTo({scatter_out});
    encoder_shape->LinksFrom({encoder2_out}).LinksTo({encoder_shape_out});
    encoder_slice0->LinksFrom({encoder_shape_out})
        .LinksTo({encoder_slice0_out});
    encoder_slice1->LinksFrom({encoder_shape_out})
        .LinksTo({encoder_slice1_out});
    encoder_constant0->LinksTo({encoder_constant0_out});
    encoder_constant1->LinksTo({encoder_constant1_out});
    encoder_reshape0
        ->LinksFrom({encoder2_out,
                     encoder_constant0_out,
                     encoder_slice0_out,
                     encoder_slice1_out})
        .LinksTo({encoder_reshape0_out, encoder_reshape0_xshape});
    encoder_reshape1->LinksFrom({encoder_reshape0_out})
        .LinksTo({encoder_reshape1_out, encoder_reshape1_xshape});
    encoder2->LinksTo({encoder2_out});
    if (adaptive_seqlen_) {
      encoder1->LinksFrom({seq_lod, pad_seq_len});
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_token_scatter_fuse_pass______";
    /*CLSInds
    X (OrgTokens)
    encoder_out

    reshape2_out*/
    ////// unpad vsl~~~~~~~~~~~~~~~~~~~

    auto* scope = matched.at("slimmed_shape")->stmt()->op()->scope();
    auto valid_places =
        matched.at("slimmed_shape")->stmt()->op()->valid_places();
    cpp::OpDesc token_scatter_op_desc;
    cpp::OpDesc encoder2_op_desc = *matched.at("encoder2")->stmt()->op_info();
    auto encoder2_instruct = matched.at("encoder2")->stmt();
    token_scatter_op_desc.mutable_inputs()->clear();
    token_scatter_op_desc.mutable_outputs()->clear();
    token_scatter_op_desc.SetType("__xpu__token_scatter");
    token_scatter_op_desc.SetInput("CLSInds",
                                   {matched.at("CLSInds")->arg()->name});
    token_scatter_op_desc.SetInput("X", {matched.at("X")->arg()->name});
    token_scatter_op_desc.SetInput("Updates",
                                   {matched.at("encoder2_out")->arg()->name});
    if (adaptive_seqlen_) {
      token_scatter_op_desc.SetInput("SeqLod",
                                     {matched.at("seq_lod")->arg()->name});
      token_scatter_op_desc.SetInput("PadSeqLen",
                                     {matched.at("pad_seq_len")->arg()->name});
    }
    token_scatter_op_desc.SetOutput("Out",
                                    {matched.at("reshape2_out")->arg()->name});
    encoder2_op_desc.SetAttr<bool>("do_padding", false);

    auto token_scatter_op =
        LiteOpRegistry::Global().Create("__xpu__token_scatter");
    token_scatter_op->Attach(token_scatter_op_desc, scope);
    token_scatter_op->SetValidPlaces(valid_places);
    auto* token_scatter_op_node =
        graph->GraphCreateInstructNode(token_scatter_op, valid_places);

    encoder2_instruct->ResetOp(encoder2_op_desc, valid_places);
    if (adaptive_seqlen_) {
      DirectedLink(matched.at("seq_lod"), token_scatter_op_node);
      DirectedLink(matched.at("pad_seq_len"), token_scatter_op_node);
    }
    DirectedLink(matched.at("CLSInds"), token_scatter_op_node);
    DirectedLink(matched.at("X"), token_scatter_op_node);
    DirectedLink(matched.at("encoder2_out"), token_scatter_op_node);
    DirectedLink(token_scatter_op_node, matched.at("reshape2_out"));
  }

 private:
  bool adaptive_seqlen_;
};

}  // namespace fusion

class XPUTokenScatterFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<bool> adaptive_seqlens{true, false};
    for (auto adaptive_seqlen : adaptive_seqlens) {
      fusion::XPUTokenScatterFuser fuser(adaptive_seqlen);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__token_scatter_fuse_pass,
                  paddle::lite::mir::XPUTokenScatterFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__token_scatter");
