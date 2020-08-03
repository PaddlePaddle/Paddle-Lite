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
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUConv2dLinkFuser : public FuseBase {
 public:
  explicit XPUConv2dLinkFuser() {}

  void BuildPattern() override {
    auto* input = VarNode("input")->assert_is_op_input("__xpu__conv2d", "Input")
                                ->AsInput();
    auto* xpu_conv = OpNode("xpu_conv", "__xpu__conv2d");
    auto* xpu_conv_out_max = VarNode("xpu_conv_out_max")
                                ->assert_is_op_output("__xpu__conv2d", "OutputMax");

    *input >> *xpu_conv >> *xpu_conv_out_max;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc = *matched.at("xpu_conv")->stmt()->op_info();

    // try to find input_max
    std::string max_input_name = matched.at("input")->arg()->name + "_max";
    auto* max_input_node = graph->RetrieveArgument(max_input_name);
    if (max_input_node != nullptr) {
      LOG(INFO) << "!!!!!!!!found max_input_node: " << max_input_name << std::endl;
      op_desc.SetAttr("has_input_max", true);
      op_desc.SetInput("InputMax", {max_input_name});
      IR_NODE_LINK_TO(max_input_node, matched.at("xpu_conv"))
    } else {
      LOG(INFO) << "!!!!!!!!not found max_input_node: " << max_input_name << std::endl;
    }
  }
};

}  // namespace fusion

class XPUConv2dLinkPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    fusion::XPUConv2dLinkFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_link_max_pass, paddle::lite::mir::XPUConv2dLinkPass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__conv2d");
