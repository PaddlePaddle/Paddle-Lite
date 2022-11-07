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

class XPUMultiEncoderSliceLinkFuser : public FuseBase {
 public:
  explicit XPUMultiEncoderSliceLinkFuser(bool pre_ln = false)
      : pre_ln_(pre_ln) {}
  void BuildPattern() override {
    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__multi_encoder");
    auto* encoder_out =
        VarNode("encoder_out")
            ->assert_is_op_output("__xpu__multi_encoder", "Output")
            ->assert_only_one_output();
    PMNode* layer_norm = nullptr;
    PMNode* layer_norm_out = nullptr;

    auto* slice = OpNode("slice", "slice")
                      ->assert_op_attr_satisfied<std::vector<int>>(
                          "axes",
                          [](const std::vector<int>& attr) {
                            return attr.size() == 1 && attr[0] == 1;
                          })
                      ->assert_op_attr_satisfied<std::vector<int>>(
                          "starts",
                          [](const std::vector<int>& attr) {
                            return attr.size() == 1 && attr[0] == 0;
                          })
                      ->assert_op_attr_satisfied<std::vector<int>>(
                          "ends", [](const std::vector<int>& attr) {
                            return attr.size() == 1 && attr[0] == 1;
                          });
    if (pre_ln_) {
      xpu_encoder->assert_op_attr<bool>("norm_before", true);
      encoder_out->assert_is_op_input("layer_norm", "X");
      layer_norm = OpNode("layer_norm", "layer_norm");
      layer_norm_out = VarNode("layer_norm_out")
                           ->assert_is_op_output("layer_norm", "Y")
                           ->assert_is_op_input("slice", "Input");
    } else {
      xpu_encoder->assert_op_attr<bool>("norm_before", false);
      encoder_out->assert_is_op_input("slice", "Input")->AsIntermediate();
      slice->AsIntermediate();
    }
    auto* slice_out = VarNode("slice_out")->assert_is_op_output("slice", "Out");
    if (pre_ln_) {
      *xpu_encoder >> *encoder_out >> *layer_norm >> *layer_norm_out >>
          *slice >> *slice_out;
    } else {
      *xpu_encoder >> *encoder_out >> *slice >> *slice_out;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* encoder_instruct = matched.at("xpu_encoder")->stmt();
    auto encoder_op_desc = *encoder_instruct->mutable_op_info();
    auto encoder_op = encoder_instruct->op();
    auto* slice_instruct = matched.at("slice")->stmt();
    auto slice_op_desc = *slice_instruct->op_info();
    std::string slice_out_name = matched.at("slice_out")->arg()->name;

    if (!pre_ln_) {
      encoder_op_desc.SetOutput("Output", {slice_out_name});
    }
    auto slice_axes = slice_op_desc.GetAttr<std::vector<int>>("axes");
    encoder_op_desc.SetAttr("slice_axes", slice_axes);
    if (slice_op_desc.HasAttr("starts")) {
      auto slice_starts = slice_op_desc.GetAttr<std::vector<int>>("starts");
      encoder_op_desc.SetAttr("slice_starts", slice_starts);
    }
    if (slice_op_desc.HasAttr("ends")) {
      auto slice_ends = slice_op_desc.GetAttr<std::vector<int>>("ends");
      encoder_op_desc.SetAttr("slice_ends", slice_ends);
    }
    if (slice_op_desc.HasAttr("decrease_axis")) {
      auto slice_decrease_axis =
          slice_op_desc.GetAttr<std::vector<int>>("decrease_axis");
      encoder_op_desc.SetAttr("slice_decrease_axis", slice_decrease_axis);
    }
    encoder_instruct->ResetOp(encoder_op_desc, encoder_op->valid_places());
    if (!pre_ln_) {
      DirectedLink(matched.at("xpu_encoder"), matched.at("slice_out"));
    }
  }

 private:
  bool pre_ln_;
};

}  // namespace fusion

class XPUMultiEncoderSliceLinkFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<bool> pre_lns{true, false};
    for (auto pre_ln : pre_lns) {
      fusion::XPUMultiEncoderSliceLinkFuser fuser(pre_ln);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_encoder_slice_link_fuse_pass,
                  paddle::lite::mir::XPUMultiEncoderSliceLinkFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_encoder");
