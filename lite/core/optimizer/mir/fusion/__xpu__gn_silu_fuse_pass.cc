// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class GnSilufuser : public FuseBase {
 public:
  void BuildPattern() override {
    // group norm
    auto* input =
        VarNode("input")->assert_is_op_input("group_norm", "X")->AsInput();
    auto* gn_scale = VarNode("gn_scale")
                         ->assert_is_op_input("group_norm", "Scale")
                         ->AsInput();
    auto* gn_bias = VarNode("gn_bias")
                        ->AsIntermediate()
                        ->assert_is_op_input("group_norm", "Bias")
                        ->AsInput();
    auto* gn = OpNode("gn", "group_norm")->AsIntermediate();
    auto* gn_out = VarNode("gn_out")
                       ->assert_is_op_output("group_norm", "Y")
                       ->assert_is_op_input("silu", "X")
                       ->AsIntermediate();
    auto* gn_mean = VarNode("gn_mean")
                        ->assert_is_op_output("group_norm", "Mean")
                        ->AsIntermediate();
    auto* gn_var = VarNode("gn_var")
                       ->assert_is_op_output("group_norm", "Variance")
                       ->AsIntermediate();
    auto* silu = OpNode("silu", "silu");
    auto* silu_out =
        VarNode("silu_out")->assert_is_op_output("silu", "Out")->AsOutput();

    // group norm
    std::vector<PMNode*> gn_input{input, gn_scale, gn_bias};
    std::vector<PMNode*> gn_output{gn_out, gn_mean, gn_var};
    gn_input >> *gn >> gn_output;
    *gn_out >> *silu >> *silu_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__gn_silu");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("GNScale",
                     {
                         matched.at("gn_scale")->arg()->name,
                     });
    op_desc.SetInput("GNBias",
                     {
                         matched.at("gn_bias")->arg()->name,
                     });
    op_desc.SetOutput("Output", {matched.at("silu_out")->arg()->name});
    auto* scope = matched.at("silu")->stmt()->op()->scope();
    auto gn_op_desc = *matched.at("gn")->stmt()->op_info();
    op_desc.SetAttr<int>("groups", gn_op_desc.GetAttr<int>("groups"));
    op_desc.SetAttr<float>("epsilon", gn_op_desc.GetAttr<float>("epsilon"));

    auto gnsilu_op = LiteOpRegistry::Global().Create(op_desc.Type());
    gnsilu_op->Attach(op_desc, scope);
    gnsilu_op->SetValidPlaces(matched.at("silu")->stmt()->op()->valid_places());
    auto kernels = gnsilu_op->CreateKernels(gnsilu_op->valid_places());
    matched.at("silu")->stmt()->SetOp(gnsilu_op);
    matched.at("silu")->stmt()->SetKernels(std::move(kernels));

    std::vector<std::string> froms = {
        "gn_scale", "gn_bias", "input",
    };

    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), matched.at("silu"));
    }
    IR_OP_VAR_LINK(matched.at("silu"), matched.at("silu_out"));
  }
};

}  // namespace fusion

class XPUGnSilufusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    fusion::GnSilufuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__gn_silu_fuse_pass,
                  paddle::lite::mir::XPUGnSilufusePass)
    .BindTargets({TARGET(kXPU)});
