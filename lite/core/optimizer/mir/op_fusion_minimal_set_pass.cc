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

#include "lite/core/optimizer/mir/op_fusion_minimal_set_pass.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

#define SKIP_DELETE_INTERMEDIATE_NODES \
  for (auto& item : key2nodes_) {      \
    if (&item == &matched) {           \
      item.clear();                    \
    }                                  \
  }

class MulElementwiseAddFuser : public FuseBase {
 public:
  void BuildPattern() override {
    // Create the pattern nodes.
    auto mul_node = OpNode("mul", "mul")->AsIntermediate();
    auto mul_x_node = VarNode("mul_x")->assert_is_op_input("mul", "X");
    auto mul_y_node = VarNode("mul_y")
                          ->assert_is_op_input("mul", "Y")
                          ->assert_is_persistable_var();
    auto mul_out_node = VarNode("mul_out")
                            ->assert_is_op_output("mul", "Out")
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto elementwise_add_node =
        OpNode("elementwise_add", "elementwise_add")
            ->assert_node_satisfied([](const Node* node) -> bool {
              auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
              auto y_name = op_desc.Input("Y").front();
              auto scope = const_cast<Node*>(node)->AsStmt().op()->scope();
              auto y_tensor = scope->FindVar(y_name)->Get<lite::Tensor>();
              if (!y_tensor.persistable()) return false;
              return y_tensor.dims().size() == 1;
            })
            ->AsIntermediate();
    auto elementwise_add_y_node =
        VarNode("elementwise_add_y")
            ->assert_is_op_input("elementwise_add", "Y")
            ->assert_is_persistable_var();
    auto elementwise_add_out_node =
        VarNode("elementwise_add_out")
            ->assert_is_op_output("elementwise_add", "Out");
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> mul_inputs{mul_x_node, mul_y_node};
    std::vector<PMNode*> elementwise_add_inputs{mul_out_node,
                                                elementwise_add_y_node};
    mul_inputs >> *mul_node >> *mul_out_node;
    elementwise_add_inputs >> *elementwise_add_node >>
        *elementwise_add_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto mul_node = matched.at("mul");
    auto mul_op = mul_node->stmt()->op();
    auto scope = mul_op->scope();
    auto mul_x_node = matched.at("mul_x");
    auto mul_x_name = mul_x_node->arg()->name;
    auto mul_y_node = matched.at("mul_y");
    auto mul_y_name = mul_y_node->arg()->name;
    auto mul_y_var = scope->FindVar(mul_y_name);
    auto mul_y_dims = mul_y_var->Get<lite::Tensor>().dims();
    auto mul_out_node = matched.at("mul_out");
    auto mul_out_name = mul_out_node->arg()->name;
    auto elementwise_add_node = matched.at("elementwise_add");
    auto elementwise_add_y_node = matched.at("elementwise_add_y");
    auto elementwise_add_y_name = elementwise_add_y_node->arg()->name;
    auto elementwise_add_y_var = scope->FindVar(elementwise_add_y_name);
    auto elementwise_add_y_dims =
        elementwise_add_y_var->Get<lite::Tensor>().dims();
    auto elementwise_add_out_node = matched.at("elementwise_add_out");
    auto elementwise_add_out_name = elementwise_add_out_node->arg()->name;
    auto fc_num_units = mul_y_dims[1];
    if (elementwise_add_y_dims[0] != fc_num_units) {
      SKIP_DELETE_INTERMEDIATE_NODES
      LOG(WARNING) << "Op fusion failed! The dimension of the input Y of "
                      "elementwise_add should be ["
                   << fc_num_units << "], but recieve ["
                   << elementwise_add_y_dims[0] << "]!";
      return;
    }
    // Get the attributes from mul op and elementwise_add op.
    auto fc_desc = *mul_node->stmt()->op_info();
    auto elementwise_add_desc = *elementwise_add_node->stmt()->op_info();
    // Get input scales from mul op.
    std::vector<float> mul_x_scales;
    std::vector<float> mul_y_scales;
    if (fc_desc.HasInputScale(mul_x_name)) {
      mul_x_scales = fc_desc.GetInputScale(mul_x_name);
    }
    if (fc_desc.HasInputScale(mul_y_name)) {
      mul_y_scales = fc_desc.GetInputScale(mul_y_name);
    }
    // Get the output threshold from elementwise_add op.
    float elementwise_add_out_out_threshold = -1.0f;
    if (elementwise_add_desc.HasAttr("out_threshold")) {
      elementwise_add_out_out_threshold =
          elementwise_add_desc.GetAttr<float>("out_threshold");
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    float elementwise_add_out_Out0_threshold = -1.0f;
    if (elementwise_add_desc.HasAttr("Out0_threshold")) {
      elementwise_add_out_Out0_threshold =
          elementwise_add_desc.GetAttr<float>("Out0_threshold");
    }
    // Modify the mul op desc for a new fc op.
    fc_desc.mutable_inputs()->clear();
    fc_desc.mutable_outputs()->clear();
    fc_desc.SetType("fc");
    fc_desc.SetInput("Input", {mul_x_name});
    fc_desc.SetInput("W", {mul_y_name});
    fc_desc.SetInput("Bias", {elementwise_add_y_name});
    fc_desc.SetOutput("Out", {elementwise_add_out_name});
    fc_desc.SetAttr("in_num_col_dims", fc_desc.GetAttr<int>("x_num_col_dims"));
    if (!mul_x_scales.empty()) {
      fc_desc.SetInputScale(mul_x_name, mul_x_scales);
    }
    if (!mul_y_scales.empty()) {
      fc_desc.SetInputScale(mul_y_name, mul_y_scales);
    }
    if (elementwise_add_out_out_threshold > 0.f) {
      fc_desc.SetAttr("out_threshold", elementwise_add_out_out_threshold);
    }
    if (elementwise_add_out_Out0_threshold > 0.f) {
      fc_desc.SetAttr("Out0_threshold", elementwise_add_out_Out0_threshold);
    }
    // Create a new fc op with the op desc, and replace the matched subgraph
    // nodes.
    auto fc_op = LiteOpRegistry::Global().Create("fc");
    auto& valid_places = mul_op->valid_places();
    fc_op->Attach(fc_desc, scope);
    auto fc_node = graph->GraphCreateInstructNode(fc_op, valid_places);
    IR_NODE_LINK_TO(mul_x_node, fc_node);
    IR_NODE_LINK_TO(mul_y_node, fc_node);
    IR_NODE_LINK_TO(elementwise_add_y_node, fc_node);
    IR_OP_VAR_LINK(fc_node, elementwise_add_out_node);
  }
};

class Conv2dElementwiseAddFuser : public FuseBase {
 public:
  explicit Conv2dElementwiseAddFuser(const std::string& conv2d_type,
                                     bool has_bias)
      : conv2d_type_(conv2d_type), has_bias_(has_bias) {}

  void BuildPattern() override {
    // Create the pattern nodes.
    auto conv2d_node = OpNode("conv2d", conv2d_type_);
    auto conv2d_input_node =
        VarNode("conv2d_input")->assert_is_op_input(conv2d_type_, "Input");
    auto conv2d_filter_node =
        VarNode("conv2d_filter")->assert_is_op_input(conv2d_type_, "Filter");
    auto conv2d_output_node = VarNode("conv2d_output")
                                  ->assert_is_op_output(conv2d_type_, "Output")
                                  ->assert_is_op_input("elementwise_add", "X")
                                  ->AsIntermediate();
    auto elementwise_add_node =
        OpNode("elementwise_add", "elementwise_add")
            ->assert_node_satisfied([](const Node* node) -> bool {
              auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
              auto y_name = op_desc.Input("Y").front();
              auto scope = const_cast<Node*>(node)->AsStmt().op()->scope();
              auto y_tensor = scope->FindVar(y_name)->Get<lite::Tensor>();
              if (!y_tensor.persistable()) return false;
              return y_tensor.dims().size() == 1;
            })
            ->AsIntermediate();
    auto elementwise_add_y_node =
        VarNode("elementwise_add_y")
            ->assert_is_op_input("elementwise_add", "Y")
            ->assert_is_persistable_var();
    auto elementwise_add_out_node =
        VarNode("elementwise_add_out")
            ->assert_is_op_output("elementwise_add", "Out");
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> conv2d_inputs{conv2d_input_node, conv2d_filter_node};
    if (has_bias_) {
      auto conv2d_bias_node = VarNode("conv2d_bias")
                                  ->assert_is_op_input(conv2d_type_, "Bias")
                                  ->assert_is_persistable_var()
                                  ->AsIntermediate();
      conv2d_inputs.emplace_back(conv2d_bias_node);
    }
    std::vector<PMNode*> elementwise_add_inputs{conv2d_output_node,
                                                elementwise_add_y_node};
    conv2d_inputs >> *conv2d_node >> *conv2d_output_node;
    elementwise_add_inputs >> *elementwise_add_node >>
        *elementwise_add_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv2d_node = matched.at("conv2d");
    auto conv2d_op = conv2d_node->stmt()->op();
    auto scope = conv2d_op->scope();
    auto conv2d_filter_node = matched.at("conv2d_filter");
    auto conv2d_filter_name = conv2d_filter_node->arg()->name;
    auto conv2d_filter_var = scope->FindVar(conv2d_filter_name);
    auto conv2d_filter_dims = conv2d_filter_var->Get<lite::Tensor>().dims();
    auto elementwise_add_node = matched.at("elementwise_add");
    auto elementwise_add_op = elementwise_add_node->stmt()->op();
    auto elementwise_add_y_node = matched.at("elementwise_add_y");
    auto elementwise_add_y_name = elementwise_add_y_node->arg()->name;
    auto elementwise_add_y_var = scope->FindVar(elementwise_add_y_name);
    auto elementwise_add_y_tensor =
        elementwise_add_y_var->GetMutable<lite::Tensor>();
    auto elementwise_add_y_dims = elementwise_add_y_tensor->dims();
    auto elementwise_add_out_node = matched.at("elementwise_add_out");
    auto elementwise_add_out_name = elementwise_add_out_node->arg()->name;
    // Get the attributes from conv2d op and elementwise_add op.
    auto conv2d_desc = *conv2d_node->stmt()->op_info();
    auto elementwise_add_desc = *elementwise_add_node->stmt()->op_info();
    auto conv2d_groups = conv2d_desc.GetAttr<int>("groups");
    size_t conv2d_output_channel_size = conv2d_filter_dims[0];
    if (conv2d_type_ == "conv2d_transpose") {
      conv2d_output_channel_size = conv2d_filter_dims[1] * conv2d_groups;
    }
    if (elementwise_add_y_dims[0] != conv2d_output_channel_size) {
      SKIP_DELETE_INTERMEDIATE_NODES
      LOG(WARNING) << "Op fusion failed! The dimension of the input Y of "
                      "elementwise_add should be "
                   << conv2d_output_channel_size << ", but recieve ["
                   << elementwise_add_y_dims[0] << "]!";
      return;
    }
    // Merge bias values if bias already exists in conv2d
    if (has_bias_) {
      auto conv2d_bias_node = matched.at("conv2d_bias");
      auto conv2d_bias_name = conv2d_bias_node->arg()->name;
      auto conv2d_bias_var = scope->FindVar(conv2d_bias_name);
      auto conv2d_bias_tensor = conv2d_bias_var->Get<lite::Tensor>();
      auto conv2d_bias_dims = conv2d_bias_tensor.dims();
      CHECK_EQ(conv2d_bias_dims.size(), 1);
      CHECK_EQ(conv2d_bias_dims[0], conv2d_output_channel_size);
      auto conv2d_bias_data = conv2d_bias_tensor.data<float>();
      auto elementwise_add_y_data =
          elementwise_add_y_tensor->mutable_data<float>();
      for (size_t i = 0; i < conv2d_output_channel_size; i++) {
        elementwise_add_y_data[i] += conv2d_bias_data[i];
      }
    }
    // Get the output threshold from elementwise_add op.
    float elementwise_add_out_out_threshold = -1.0f;
    if (elementwise_add_desc.HasAttr("out_threshold")) {
      elementwise_add_out_out_threshold =
          elementwise_add_desc.GetAttr<float>("out_threshold");
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    float elementwise_add_out_Out0_threshold = -1.0f;
    if (elementwise_add_desc.HasAttr("Out0_threshold")) {
      elementwise_add_out_Out0_threshold =
          elementwise_add_desc.GetAttr<float>("Out0_threshold");
    }
    // Update the conv2d op desc and links
    conv2d_desc.SetInput("Bias", {elementwise_add_y_name});
    conv2d_desc.SetOutput("Output", {elementwise_add_out_name});
    if (elementwise_add_out_out_threshold > 0.f) {
      conv2d_desc.SetAttr("out_threshold", elementwise_add_out_out_threshold);
    }
    if (elementwise_add_out_Out0_threshold > 0.f) {
      conv2d_desc.SetAttr("Out0_threshold", elementwise_add_out_Out0_threshold);
    }
    auto& valid_places = conv2d_op->valid_places();
    conv2d_node->stmt()->ResetOp(conv2d_desc, valid_places);
    IR_NODE_LINK_TO(elementwise_add_y_node, conv2d_node);
    IR_OP_VAR_LINK(conv2d_node, elementwise_add_out_node);
  }

 private:
  std::string conv2d_type_{"conv2d"};
  bool has_bias_{false};
};

class Conv2dActivationFuser : public FuseBase {
 public:
  explicit Conv2dActivationFuser(const std::string& conv2d_type,
                                 const std::string& act_type,
                                 bool has_bias)
      : conv2d_type_(conv2d_type), act_type_(act_type), has_bias_(has_bias) {}

  void BuildPattern() override {
    // Create the pattern nodes.
    auto conv2d_node = OpNode("conv2d", conv2d_type_);
    auto conv2d_input_node =
        VarNode("conv2d_input")->assert_is_op_input(conv2d_type_, "Input");
    auto conv2d_filter_node =
        VarNode("conv2d_filter")->assert_is_op_input(conv2d_type_, "Filter");
    auto conv2d_output_node = VarNode("conv2d_output")
                                  ->assert_is_op_output(conv2d_type_, "Output")
                                  ->assert_is_op_input(act_type_, "X")
                                  ->AsIntermediate();
    auto act_node = OpNode("act", act_type_)->AsIntermediate();
    auto act_out_node =
        VarNode("act_out")->assert_is_op_output(act_type_, "Out");
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> conv2d_inputs{conv2d_input_node, conv2d_filter_node};
    if (has_bias_) {
      auto conv2d_bias_node = VarNode("conv2d_bias")
                                  ->assert_is_op_input(conv2d_type_, "Bias")
                                  ->assert_is_persistable_var();
      conv2d_inputs.emplace_back(conv2d_bias_node);
    }
    conv2d_inputs >> *conv2d_node >> *conv2d_output_node;
    *conv2d_output_node >> *act_node >> *act_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv2d_node = matched.at("conv2d");
    auto conv2d_op = conv2d_node->stmt()->op();
    auto act_node = matched.at("act");
    auto act_out_node = matched.at("act_out");
    auto act_out_name = act_out_node->arg()->name;
    // Get the attributes from conv2d op and act op.
    auto conv2d_desc = *conv2d_node->stmt()->op_info();
    auto act_desc = *act_node->stmt()->op_info();
    // Get the output threshold from act op.
    float act_out_out_threshold = -1.0f;
    if (act_desc.HasAttr("out_threshold")) {
      act_out_out_threshold = act_desc.GetAttr<float>("out_threshold");
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    float act_out_Out0_threshold = -1.0f;
    if (act_desc.HasAttr("Out0_threshold")) {
      act_out_Out0_threshold = act_desc.GetAttr<float>("Out0_threshold");
    }
    // Update the conv2d op desc and links
    conv2d_desc.SetAttr("with_act", true);
    conv2d_desc.SetAttr("act_type", act_type_);
    if (act_type_ == "relu") {
      conv2d_desc.SetAttr("fuse_relu", true);
    }
    if (act_type_ == "relu6") {
      float alpha = act_desc.GetAttr<float>("threshold");
      conv2d_desc.SetAttr("fuse_brelu_threshold", alpha);
    }
    conv2d_desc.SetOutput("Output", {act_out_name});
    if (act_out_out_threshold > 0.f) {
      conv2d_desc.SetAttr("out_threshold", act_out_out_threshold);
    }
    if (act_out_Out0_threshold > 0.f) {
      conv2d_desc.SetAttr("Out0_threshold", act_out_Out0_threshold);
    }
    auto& valid_places = conv2d_op->valid_places();
    conv2d_node->stmt()->ResetOp(conv2d_desc, valid_places);
    IR_OP_VAR_LINK(conv2d_node, act_out_node);
  }

 private:
  std::string conv2d_type_{"conv2d"};
  std::string act_type_{"relu"};
  bool has_bias_{false};
};

void ApplyMulElementwiseAddFuser(SSAGraph* graph) {
  MulElementwiseAddFuser fuser;
  fuser(graph);
}

void ApplyMatmulElementwiseAddFuser(SSAGraph* graph) {
  // MatmulElementwiseAddFuser fuser;
  // fuser(graph);
}

void ApplyFCActivationFuser(SSAGraph* graph) {
  // FCActivationFuser fuser;
  // fuser(graph);
}

void ApplyConv2dBatchNormFuser(SSAGraph* graph) {
  // Conv2dBatchNormFuser fuser;
  // fuser(graph);
}

void ApplyConv2dElementwiseAddFuser(SSAGraph* graph) {
  // The case of has_bias=true should be handled first
  for (auto has_bias : {true, false}) {
    for (auto conv_type : {"conv2d", "depthwise_conv2d", "conv2d_transpose"}) {
      VLOG(5) << "conv_type:" << conv_type << " has_bias:" << has_bias;
      Conv2dElementwiseAddFuser fuser(conv_type, has_bias);
      fuser(graph);
    }
  }
}

void ApplyConv2dActivationFuser(SSAGraph* graph) {
  for (auto conv_type : {"conv2d", "depthwise_conv2d", "conv2d_transpose"}) {
    for (auto act_type : {"relu", "relu1", "relu6"}) {
      for (auto has_bias : {true, false}) {
        VLOG(5) << "conv_type:" << conv_type << "act_type:" << act_type
                << " has_bias:" << has_bias;
        Conv2dActivationFuser fuser(conv_type, act_type, has_bias);
        fuser(graph);
      }
    }
  }
}

void ApplyReshapeTransposeReshapeFuser(SSAGraph* graph) {
  // ReshapeTransposeReshapeFuser fuser;
  // fuser(graph);
}

void OpFusionMinimalSetPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  ApplyMulElementwiseAddFuser(graph.get());
  ApplyMatmulElementwiseAddFuser(graph.get());
  ApplyFCActivationFuser(graph.get());
  ApplyConv2dBatchNormFuser(graph.get());
  ApplyConv2dElementwiseAddFuser(graph.get());
  ApplyConv2dActivationFuser(graph.get());
  // Since some hardware does not support 5-D inputs and outputs, and the
  // shuffle channel op is more general and friendly to hardware manufacturers,
  // it is necessary to convert reshape+transpose+reshape to shuffle channel op.
  ApplyReshapeTransposeReshapeFuser(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(op_fusion_minimal_set_pass,
                  paddle::lite::mir::OpFusionMinimalSetPass)
    .BindTargets({TARGET(kNNAdapter)});
