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
    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__encoder");
    auto* encoder_out = VarNode("encoder_out")
                            ->assert_is_op_output("__xpu__encoder", "Output")
                            ->assert_only_one_output();
    PMNode* layer_norm = nullptr;
    PMNode* layer_norm_out = nullptr;

    auto* slice =
        OpNode("slice", "slice")
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
                "ends",
                [](const std::vector<int>& attr) {
                  return attr.size() == 1 && attr[0] == 1;
                })
            ->assert_node_satisfied([](const Node* node) -> bool {
              if (!const_cast<Node*>(node)->stmt()->op_info()->HasAttr(
                      "decrease_axis")) {
                return true;
              }
              std::vector<int> decrease_axis =
                  const_cast<Node*>(node)
                      ->stmt()
                      ->op_info()
                      ->GetAttr<std::vector<int>>("decrease_axis");
              if (decrease_axis.size() == 0) {
                return true;
              } else if (decrease_axis.size() == 1) {
                return decrease_axis[0] == 1;
              }
              return false;
            });
    if (pre_ln_) {
      xpu_encoder->assert_op_attr<bool>("norm_before", true);
      encoder_out->assert_is_op_input("layer_norm", "X");
      layer_norm = OpNode("layer_norm", "layer_norm");
      layer_norm_out = VarNode("layer_norm_out")
                           ->assert_is_op_output("layer_norm", "Y")
                           ->assert_is_op_input("slice", "Input")
                           ->assert_only_one_output();
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

    if (pre_ln_) {
      encoder_op_desc.SetAttr<bool>("has_slice_decrease_axis", false);
    } else {
      DirectedLink(matched.at("xpu_encoder"), matched.at("slice_out"));
      encoder_op_desc.SetOutput("Output",
                                {matched.at("slice_out")->arg()->name});
      if (matched.at("slice")->stmt()->op_info()->HasAttr("decrease_axis") &&
          matched.at("slice")
                  ->stmt()
                  ->op_info()
                  ->GetAttr<std::vector<int>>("decrease_axis")
                  .size() > 0) {
        encoder_op_desc.SetAttr<bool>("has_slice_decrease_axis", true);
      } else {
        encoder_op_desc.SetAttr<bool>("has_slice_decrease_axis", false);
      }
    }
    encoder_op_desc.SetAttr<bool>("do_slice", true);
    encoder_op_desc.SetAttr<bool>("do_padding", false);
    encoder_instruct->ResetOp(encoder_op_desc, encoder_op->valid_places());
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
    .BindKernel("__xpu__encoder");
