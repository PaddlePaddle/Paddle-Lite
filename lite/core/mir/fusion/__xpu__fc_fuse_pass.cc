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

class XPUFcFuser : public FuseBase {
 public:
  explicit XPUFcFuser(bool with_relu) : with_relu_(with_relu) {}

  void BuildPattern() override {
    // create nodes.
    auto* x = VarNode("x")->assert_is_op_input("mul", "X");
    auto* W = VarNode("W")->assert_is_op_input("mul", "Y");
    auto* b = VarNode("b")->assert_is_persistable_var();
    auto* mul = OpNode("mul", "mul");
    auto* mul_out = VarNode("mul_out");
    auto* add = OpNode("add", "elementwise_add");
    auto* Out = VarNode("Out");

    // create topology.
    std::vector<PMNode*> mul_inputs{W, x};
    std::vector<PMNode*> add_inputs{mul_out, b};
    mul_inputs >> *mul >> *mul_out;

    // Some op specialities.
    mul_out->AsIntermediate();
    mul->AsIntermediate();
    add->AsIntermediate();

    if (with_relu_) {
      auto* add_out = VarNode("add_out");
      auto* relu = OpNode("relu", "relu");
      std::vector<PMNode*> relu_inputs{add_out};
      add_inputs >> *add >> *add_out;
      relu_inputs >> *relu >> *Out;
      add_out->AsIntermediate();
      relu->AsIntermediate();
    } else {
      add_inputs >> *add >> *Out;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto mul = matched.at("mul")->stmt()->op();
    auto* scope = mul->scope();

    // convert W from float to int16, and transpose W
    auto weight_name = matched.at("W")->arg()->name;
    auto* weight_t = scope->FindMutableTensor(weight_name);
    auto weight_dims = weight_t->dims();
    int weight_len = weight_t->numel();
    float* weight_on_host = weight_t->mutable_data<float>();
    float max_f =
        paddle::lite::xpu::math::FindMaxAbs(weight_on_host, weight_len);

    std::unique_ptr<int16_t[]> weight_int16(new int16_t[weight_len]);
    std::unique_ptr<int16_t[]> weight_trans_int16(new int16_t[weight_len]);
    paddle::lite::xpu::math::ConvertFP32ToInt16(
        weight_on_host, weight_int16.get(), max_f, weight_len);
    paddle::lite::xpu::math::Transpose(weight_int16.get(),
                                       weight_trans_int16.get(),
                                       weight_dims[0],
                                       weight_dims[1]);
    memcpy(
        weight_on_host, weight_trans_int16.get(), weight_len * sizeof(int16_t));

    auto op_desc = GenOpDesc(matched, max_f, true);
    auto fc_op = LiteOpRegistry::Global().Create("__xpu__fc");
    auto& valid_places = mul->valid_places();
    fc_op->Attach(op_desc, scope);

    auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

    IR_NODE_LINK_TO(matched.at("W"), new_op_node);
    IR_NODE_LINK_TO(matched.at("x"), new_op_node);
    IR_NODE_LINK_TO(matched.at("b"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
  }

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched,
                        float w_max,
                        bool transpose_w) {
    cpp::OpDesc op_desc = *matched.at("mul")->stmt()->op_info();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__fc");
    op_desc.SetInput("Input", {matched.at("x")->arg()->name});
    op_desc.SetInput("W", {matched.at("W")->arg()->name});
    op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
    op_desc.SetAttr(
        "in_num_col_dims",
        matched.at("mul")->stmt()->op_info()->GetAttr<int>("x_num_col_dims"));
    op_desc.SetAttr("w_max", w_max);
    op_desc.SetAttr("transpose_w", transpose_w);
    if (with_relu_) {
      op_desc.SetAttr("activation_type", std::string{"relu"});
    }
    return op_desc;
  }

  bool with_relu_;
};

}  // namespace fusion

class XPUFcFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;

    fusion::XPUFcFuser fuser(true /* with_relu */);
    fuser(graph.get());

    fusion::XPUFcFuser fuser2(false /* with_relu */);
    fuser2(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__fc_fuse_pass, paddle::lite::mir::XPUFcFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__fc");
