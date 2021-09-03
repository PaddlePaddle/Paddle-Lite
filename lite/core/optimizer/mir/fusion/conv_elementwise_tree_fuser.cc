// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fusion/conv_elementwise_tree_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvElementwiseTreeFuser::BuildPattern() {
  // create input nodes.
  auto* conv_input =
      VarNode("conv_input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* conv_filter = VarNode("conv_filter")
                          ->assert_is_persistable_var()
                          ->assert_is_op_input(conv_type_, "Filter")
                          ->AsInput();
  auto* elementwise_input = VarNode("elementwise_input")
                                ->assert_is_op_input(elementwise_type_, "X")
                                ->AsInput();

  // create intermediate nodes
  auto* conv_output = VarNode("conv_output")
                          ->assert_is_op_output(conv_type_, "Output")
                          ->assert_is_op_input(elementwise_type_, "Y")
                          ->assert_only_one_output();

  // create op nodes
  // The pass will not been applied if conv1x1 has already applied this pass.
  auto conv_teller = [](const Node* node) -> bool {
    bool has_fuse_elementwise_op_type =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr(
            "fuse_elementwise_op_type");
    return (!has_fuse_elementwise_op_type);
  };
  // Limitation of elementwise
  auto elementwise_teller = [](const Node* node) -> bool {
    int axis =
        const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<int>("axis");
    bool fuse_scale =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr("fuse_scale");
    bool has_act_type =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr("act_type");
    std::string act_type{""};
    if (has_act_type) {
      act_type =
          const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<std::string>(
              "act_type");
    }
    return (axis == -1) && (!fuse_scale) &&
           ((!has_act_type) || (has_act_type && act_type == "relu"));
  };

  auto* conv = OpNode("conv", conv_type_)
                   ->assert_is_op(conv_type_)
                   ->assert_node_satisfied(conv_teller);
  auto* elementwise = OpNode("elementwise", elementwise_type_)
                          ->assert_is_op(elementwise_type_)
                          ->assert_node_satisfied(elementwise_teller);

  // create output node
  auto* elementwise_output = VarNode("elementwise_output")
                                 ->assert_is_op_output(elementwise_type_, "Out")
                                 ->AsOutput();

  // create topology.
  // consider two special cases: conv with bias, conv with prelu alpha
  std::vector<PMNode*> conv_inputs{conv_input, conv_filter};
  if (conv_has_bias_) {
    auto* conv_bias = VarNode("conv_bias")
                          ->assert_is_op_input(conv_type_, "Bias")
                          ->assert_is_persistable_var();
    conv_inputs.push_back(conv_bias);
  }
  if (conv_has_prelu_alpha_) {
    auto* conv_alpha = VarNode("conv_alpha")
                           ->assert_is_op_input(conv_type_, "Prelu_alpha")
                           ->assert_is_persistable_var()
                           ->AsInput();
    conv_inputs.push_back(conv_alpha);
  }
  conv->LinksFrom(conv_inputs).LinksTo({conv_output});
  elementwise->LinksFrom({elementwise_input, conv_output})
      .LinksTo({elementwise_output});
}

void ConvElementwiseTreeFuser::InsertNewNode(SSAGraph* graph,
                                             const key2nodes_t& matched) {
  auto GetTensorDims = [](const key2nodes_t& matched,
                          const std::string key,
                          const std::string out_or_filter,
                          DDimLite& dims) {
    std::string var_name;
    auto* inst = matched.at(key)->stmt();
    const auto op = inst->op();
    const auto* op_info = inst->op_info();
    if (out_or_filter == "out") {
      auto var_names = op_info->output_names();
      CHECK_EQ(var_names.size(), 1);
      var_name = var_names[0];
    } else if (out_or_filter == "filter") {
      var_name = op_info->Input("Filter").front();
    } else {
      LOG(FATAL) << "Illegal request!";
    }
    auto* scope = op->scope();
    auto* var = scope->FindVar(var_name);
    if (var == nullptr) {
      LOG(WARNING) << "var is nullptr! var_name: " << var_name;
      return;
    }
    const auto& tensor = var->Get<Tensor>();
    dims = tensor.dims();
  };

  // Check output dims.
  DDimLite conv_out_dims, elementwise_out_dims;
  GetTensorDims(matched, "conv", "out", conv_out_dims);
  GetTensorDims(matched, "elementwise", "out", elementwise_out_dims);
  if (conv_out_dims != elementwise_out_dims) {
    VLOG(4) << "Output dims is not the same between " << elementwise_type_
            << " and " << conv_type_
            << ". Skip this pass! Output tensor dims of elementwise is "
            << elementwise_out_dims << ", while output tensor dims of conv is "
            << conv_out_dims;
    return;
  }

  // Check filter dims as this pass only support conv1x1 by now.
  DDimLite conv_filter_dims;
  GetTensorDims(matched, "conv", "filter", conv_filter_dims);
  if (!(conv_filter_dims[2] == 1 && conv_filter_dims[3] == 1)) {
    VLOG(4) << "This pass only support conv1x1, while the conv filter dims is "
            << conv_filter_dims << ". Skip this pass!";
    return;
  }

  // NOTE: push these note to nodes2rm_.
  nodes2rm_.insert(matched.at("conv"));
  nodes2rm_.insert(matched.at("conv_output"));
  nodes2rm_.insert(matched.at("elementwise"));

  auto op_desc = GenOpDesc(matched);
  auto conv_op_new = LiteOpRegistry::Global().Create(conv_type_);
  auto conv_op_old = matched.at("conv")->stmt()->op();
  auto* scope = conv_op_old->scope();
  auto& valid_places = conv_op_old->valid_places();
  conv_op_new->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(conv_op_new, valid_places);

  IR_NODE_LINK_TO(matched.at("elementwise_input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("conv_input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("conv_filter"), new_op_node);
  if (conv_has_bias_) {
    IR_NODE_LINK_TO(matched.at("conv_bias"), new_op_node);
  }
  if (conv_has_prelu_alpha_) {
    IR_NODE_LINK_TO(matched.at("conv_alpha"), new_op_node);
  }
  IR_NODE_LINK_TO(new_op_node, matched.at("elementwise_output"));
}

cpp::OpDesc ConvElementwiseTreeFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("conv")->stmt()->op_info();
  op_desc.SetType(conv_type_);
  op_desc.SetInput("Filter", {matched.at("conv_filter")->arg()->name});
  if (conv_has_bias_) {
    op_desc.SetInput("Bias", {matched.at("conv_bias")->arg()->name});
  }
  if (conv_has_prelu_alpha_) {
    op_desc.SetInput("Prelu_alpha", {matched.at("conv_alpha")->arg()->name});
  }
  op_desc.SetAttr("fuse_elementwise_op_type", elementwise_type_);
  op_desc.SetInput("SecondInput",
                   {matched.at("elementwise_input")->arg()->name});
  op_desc.SetOutput("Output", {matched.at("elementwise_output")->arg()->name});

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
