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
  explicit XPUFcFuser(bool with_bias, const std::string& act_type) {
    with_bias_ = with_bias;
    act_type_ = act_type;
  }

  void BuildPattern() override {
    auto* x = VarNode("x")->assert_is_op_input("mul", "X")->AsInput();
    auto* W = VarNode("W")->assert_is_op_input("mul", "Y")->AsInput();
    auto* mul = OpNode("mul", "mul")->AsIntermediate();
    auto* mul_out = VarNode("mul_out")->assert_is_op_output("mul", "Out");
    PMNode* bias = nullptr;
    PMNode* add = nullptr;
    PMNode* add_out = nullptr;
    PMNode* act = nullptr;
    PMNode* act_out = nullptr;
    if (with_bias_) {
      mul_out->assert_is_op_input("elementwise_add", "X");
      bias = VarNode("bias")
                 ->assert_is_op_input("elementwise_add", "Y")
                 ->assert_is_persistable_var()
                 ->AsInput();
      add = OpNode("add", "elementwise_add")->AsIntermediate();
      add_out =
          VarNode("add_out")->assert_is_op_output("elementwise_add", "Out");
    }
    if (act_type_ != "linear") {
      act = OpNode("act", act_type_)->AsIntermediate();
      act_out = VarNode("act_out")->assert_is_op_output(act_type_, "Out");
    }
    *x >> *mul >> *mul_out;
    if (with_bias_) {
      mul_out->AsIntermediate();
      *mul_out >> *add >> *add_out;
      *bias >> *add;
    } else {
      add_out = mul_out;
    }
    if (act_type_ != "linear") {
      add_out->assert_is_op_input(act_type_, "X")->AsIntermediate();
      *add_out >> *act >> *act_out;
    } else {
      act_out = add_out;
    }
    *W >> *mul;
    act_out->AsOutput();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc = *matched.at("mul")->stmt()->op_info();
    auto mul = matched.at("mul")->stmt()->op();
    auto* scope = mul->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__fc");
    op_desc.SetInput("Input", {matched.at("x")->arg()->name});
    op_desc.SetInput("Filter", {matched.at("W")->arg()->name});

    std::string precision = "int16";
#ifdef LITE_WITH_XPU
    std::string multi_encoder_precision = ContextScheduler::Global()
                                              .NewContext(TARGET(kXPU))
                                              ->As<XPUContext>()
                                              .MultiEncoderPrecision();
    if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int31" ||
        multi_encoder_precision == "int31") {
      precision = "int31";
      VLOG(3) << "Use int31 in XPUFcOp";
    } else if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int8" ||
               multi_encoder_precision == "int8") {
      precision = "int8";
      VLOG(3) << "Use int8 in XPUFcOp";
    }
#endif
    if (with_bias_) {
      op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
    }
    op_desc.SetAttr<bool>("has_bias", with_bias_);
    std::string output_name, output_node_name;
    if (act_type_ != "linear") {
      output_name = matched.at("act_out")->arg()->name;
      output_node_name = "act_out";
    } else if (with_bias_) {
      output_name = matched.at("add_out")->arg()->name;
      output_node_name = "add_out";
    } else {
      output_name = matched.at("mul_out")->arg()->name;
      output_node_name = "mul_out";
    }
    op_desc.SetOutput("Output", {output_name});
    op_desc.SetAttr<std::string>("precision", precision);
    std::map<std::string, int> act_map{{"linear", 0},
                                       {"relu", 1},
                                       {"sigmoid", 2},
                                       {"tanh", 3},
                                       {"leaky_relu", 5},
                                       {"hard_swish", 14},
                                       {"hard_sigmoid", 15},
                                       {"relu6", 17}};

    float act_param_ = 0.0f;
    if (act_type_ == "leaky_relu") {
      auto act_op_desc = *matched.at("act")->stmt()->op_info();
      act_param_ = act_op_desc.GetAttr<float>("alpha");
    } else if (act_type_ == "hard_sigmoid") {
      auto act_op_desc = *matched.at("act")->stmt()->op_info();
      act_param_ = act_op_desc.GetAttr<float>("slope");
    }
    op_desc.SetAttr<int>("act_type", act_map[act_type_]);
    op_desc.SetAttr<float>("act_param", act_param_);
    op_desc.SetAttr(
        "in_num_col_dims",
        matched.at("mul")->stmt()->op_info()->GetAttr<int>("x_num_col_dims"));

    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    auto fc_op = LiteOpRegistry::Global().Create("__xpu__fc");
    auto& valid_places = mul->valid_places();
    fc_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

    IR_NODE_LINK_TO(matched.at("W"), new_op_node);
    IR_NODE_LINK_TO(matched.at("x"), new_op_node);
    if (with_bias_) {
      IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at(output_node_name));
    DirectedLink(new_op_node, max_output_node);
  }

 private:
  bool with_bias_;
  std::string act_type_;
};

}  // namespace fusion

class XPUFcFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    // TODO(weihaoji) support with_no_bias and more activation types
    for (auto with_bias : {true, /*false*/}) {
      for (auto act_type : {"relu",
                            /*"sigmoid",
                            "tanh",
                            "leaky_relu",
                            "hard_swish",
                            "hard_sigmoid",
                            "relu6",*/
                            "linear"}) {
        fusion::XPUFcFuser fuser(with_bias, act_type);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__fc_fuse_pass, paddle::lite::mir::XPUFcFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__fc");
