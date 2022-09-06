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

class XPUFcFuser : public FuseBase {
 public:
  explicit XPUFcFuser(bool with_bias,
                      const std::string& act_type,
                      const std::string& mul_type) {
    with_bias_ = with_bias;
    act_type_ = act_type;
    mul_type_ = mul_type;
  }

  void BuildPattern() override {
    auto* x = VarNode("x")->assert_is_op_input(mul_type_, "X")->AsInput();
    auto* W = VarNode("W")->assert_is_op_input(mul_type_, "Y")->AsInput();
    auto* mul = OpNode("mul", mul_type_)->AsIntermediate();
    auto* mul_out = VarNode("mul_out")->assert_is_op_output(mul_type_, "Out");
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
    bool per_channel = false;
    int weight_scale_size = 1;
    auto* op_info = matched.at("mul")->stmt()->op_info();
    auto mul_input_y_name = op_info->Input("Y").front();
    auto mul_y_shape = scope->FindMutableTensor(mul_input_y_name)->dims();
    CHECK_EQ(mul_y_shape.size(), 2) << "mul_y_shape.size: "
                                    << mul_y_shape.size();
    const bool quant = op_info->HasAttr("enable_int8") &&
                       op_info->GetAttr<bool>("enable_int8");
    op_desc.SetAttr<bool>("enable_int8", quant);
    // int 8
    // X0_scale is already in op_desc when copy from mul
    if (quant) {
      CHECK(op_info->HasAttr("Y0_scale")) << "quant model no Y0_scale";
      weight_scale_size =
          op_info->GetAttr<std::vector<float>>("Y0_scale").size();
      CHECK_EQ(weight_scale_size, mul_y_shape[1])
          << "weight_scale_size: " << weight_scale_size
          << ", mul_y_shape:" << mul_y_shape;
      CHECK_GE(weight_scale_size, 1) << weight_scale_size;
      std::vector<float> weight_max;
      if (is_per_tensor(op_info->GetAttr<std::vector<float>>("Y0_scale"))) {
        per_channel = false;
        VLOG(3) << "xpu fc per tensor";
        weight_max.push_back(
            op_info->GetAttr<std::vector<float>>("Y0_scale")[0] * 127);
      } else {
        per_channel = true;
        VLOG(3) << "xpu fc per channel, first channel max:"
                << op_info->GetAttr<std::vector<float>>("Y0_scale")[0] * 127
                << ", last channel max: "
                << op_info->GetAttr<std::vector<float>>(
                       "Y0_scale")[weight_scale_size - 1] *
                       127;
        for (auto wm : op_info->GetAttr<std::vector<float>>("Y0_scale")) {
          weight_max.push_back(wm * 127);
        }
      }
      VLOG(3) << "weight_max size:" << weight_max.size();
      op_desc.SetAttr<std::vector<float>>("Filter0_scale", weight_max);
      op_desc.SetAttr<bool>("per_channel", per_channel);

      op_desc.SetAttr<std::vector<float>>(
          "Input0_scale",
          {127 *
           matched.at("mul")->stmt()->op_info()->GetInputScale(
               matched.at("x")->arg()->name)[0]});
      // don't need * 127
      op_desc.SetAttr<std::vector<float>>(
          "Output0_scale",
          {matched.at("mul")->stmt()->op_info()->GetAttr<float>(
              "out_threshold")});
    }

    // conv2d int16
    if (matched.at("mul")->stmt()->op_info()->HasAttr("enable_int16") &&
        matched.at("mul")->stmt()->op_info()->GetAttr<bool>("enable_int16")) {
      op_desc.SetAttr<bool>("enable_int16", true);
      op_desc.SetAttr<std::vector<float>>(
          "Input0_scale",
          {((2 << 15) - 1) *
           matched.at("mul")->stmt()->op_info()->GetInputScale(
               matched.at("x")->arg()->name)[0]});

      op_desc.SetAttr<std::vector<float>>(
          "Filter0_scale",
          {((2 << 15) - 1) *
           matched.at("mul")->stmt()->op_info()->GetInputScale(
               matched.at("W")->arg()->name)[0]});
    }

    op_desc.SetOutput("Output", {output_name});
    std::map<std::string, int> act_map{{"linear", 0},
                                       {"relu", 1},
                                       {"sigmoid", 2},
                                       {"tanh", 3},
                                       {"gelu", 4},
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

    op_desc.SetAttr<int>("in_num_col_dims", -1);
    if (mul_type_ == "mul") {
      op_desc.SetAttr(
          "in_num_col_dims",
          matched.at("mul")->stmt()->op_info()->GetAttr<int>("x_num_col_dims"));
      op_desc.SetAttr("transpose_x", false);
      op_desc.SetAttr("transpose_w", true);
    } else if (mul_type_ == "matmul") {
      op_desc.SetAttr(
          "transpose_x",
          matched.at("mul")->stmt()->op_info()->GetAttr<bool>("transpose_X"));
      op_desc.SetAttr(
          "transpose_w",
          matched.at("mul")->stmt()->op_info()->GetAttr<bool>("transpose_Y"));
    } else {
      op_desc.SetAttr(
          "transpose_x",
          matched.at("mul")->stmt()->op_info()->GetAttr<bool>("trans_x"));
      op_desc.SetAttr(
          "transpose_w",
          matched.at("mul")->stmt()->op_info()->GetAttr<bool>("trans_y"));
    }

    std::string max_output_name = output_name + "_xpu_max";
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
  std::string mul_type_;
  bool is_per_tensor(const std::vector<float>& weight_max) {
    bool per_tensor = true;
    CHECK_GT(weight_max.size(), 0) << "fc channel size: " << weight_max.size();
    auto first = weight_max[0];
    for (int i = 1; i < weight_max.size(); ++i) {
      if (std::abs(first - weight_max[i]) > 1e-6) {
        per_tensor = false;
        break;
      }
    }
    return per_tensor;
  }
};

}  // namespace fusion

class XPUFcFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    // TODO(weihaoji) support with_no_bias and more activation types
    for (auto with_bias : {true, /*false*/}) {
      for (auto act_type : {"relu",
                            "gelu",
                            /*"sigmoid",
                            "tanh",
                            "leaky_relu",
                            "hard_swish",
                            "hard_sigmoid",
                            "relu6",*/
                            "linear"}) {
        for (auto mul_type : {"mul", "matmul_v2"}) {
          fusion::XPUFcFuser fuser(with_bias, act_type, mul_type);
          fuser(graph.get());
        }
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
