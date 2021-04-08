// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
/* Squeeze and Excitation Block Fusion for SE-ResNet */
/* graph[1]: sub block                               */
/*               in_Input                            */
/*                |     \                            */
/*                |       |                          */
/*                |  Global Pooling                  */
/*                |       |                          */
/*                |       |                          */
/*                |       FC                         */
/*                |       |                          */
/*                |       |                          */
/*                |       FC                         */
/*                \       |                          */
/*                  \     |                          */
/*                    scale                          */
/*                      |                            */
/*                   out_Output                      */

class XPUSqueezeExcitationFuser : public FuseBase {
 public:
  explicit XPUSqueezeExcitationFuser(const std::string& excitation_act_type1,
                                     const std::string& excitation_act_type2,
                                     const std::string& block_act_type,
                                     bool with_branch) {
    excitation_act_type1_ = excitation_act_type1;
    excitation_act_type2_ = excitation_act_type2;
    block_act_type_ = block_act_type;
    with_branch_ = with_branch;
  }
  void BuildPattern() override {
    PMNode* ew_branch_add = nullptr;
    PMNode* ew_branch_add_in = nullptr;
    PMNode* ew_branch_add_out = nullptr;
    PMNode* block_act = nullptr;
    PMNode* block_act_out = nullptr;

    auto* input = VarNode("input")
                      ->assert_is_op_input("pool2d", "X")
                      ->assert_is_op_input("elementwise_mul", "X")
                      ->AsInput();
    auto* pool = OpNode("pool", "pool2d")
                     ->assert_op_attr<bool>("global_pooling", true)
                     ->assert_op_attr<std::string>("pooling_type", "avg")
                     ->AsIntermediate();
    auto* pool_out = VarNode("pool_out")
                         ->assert_is_op_output("pool2d", "Out")
                         ->assert_is_op_input("mul", "X")
                         ->AsIntermediate();
    auto* mul_1 = OpNode("mul_1", "mul")->AsIntermediate();
    auto* mul_1_w =
        VarNode("mul_1_w")->assert_is_op_input("mul", "Y")->AsIntermediate();
    auto* mul_1_out = VarNode("mul_1_out")
                          ->assert_is_op_output("mul", "Out")
                          ->assert_is_op_input(excitation_act_type1_, "X")
                          ->AsIntermediate();
    auto* act_1 = OpNode("act_1", excitation_act_type1_)->AsIntermediate();
    auto* act_1_out = VarNode("act_1_out")
                          ->assert_is_op_output(excitation_act_type1_, "Out")
                          ->assert_is_op_input("mul", "X")
                          ->AsIntermediate();
    auto* mul_2 = OpNode("mul_2", "mul")->AsIntermediate();
    auto* mul_2_w =
        VarNode("mul_2_w")->assert_is_op_input("mul", "Y")->AsIntermediate();
    auto* mul_2_out = VarNode("mul_2_out")
                          ->assert_is_op_output("mul", "Out")
                          ->assert_is_op_input(excitation_act_type2_, "X")
                          ->AsIntermediate();
    auto* act_2 = OpNode("act_2", excitation_act_type2_)->AsIntermediate();
    auto* act_2_out = VarNode("act_2_out")
                          ->assert_is_op_output(excitation_act_type2_, "Out")
                          ->assert_is_op_input("elementwise_mul", "Y")
                          ->AsIntermediate();
    auto* ew_mul = OpNode("ew_mul", "elementwise_mul")->AsIntermediate();
    auto* ew_mul_out =
        VarNode("ew_mul_out")->assert_is_op_output("elementwise_mul", "Out");
    // branch
    if (with_branch_) {
      ew_branch_add_in = VarNode("ew_branch_add_in")
                             ->assert_is_op_input("elementwise_add", "X")
                             ->AsInput();
      ew_branch_add =
          OpNode("ew_branch_add", "elementwise_add")->AsIntermediate();
      ew_branch_add_out = VarNode("ew_branch_add_out")
                              ->assert_is_op_output("elementwise_add", "Out");
    }
    // act
    if (block_act_type_ != "linear") {
      block_act = OpNode("block_act", block_act_type_)->AsIntermediate();
      block_act_out =
          VarNode("block_act_out")->assert_is_op_output(block_act_type_, "Out");
    }
    // pass
    *input >> *pool >> *pool_out >> *mul_1 >> *mul_1_out >> *act_1 >>
        *act_1_out >> *mul_2 >> *mul_2_out >> *act_2 >> *act_2_out >> *ew_mul;
    *input >> *ew_mul;
    *ew_mul >> *ew_mul_out;

    if (with_branch_) {
      ew_mul_out->assert_is_op_input("elementwise_add", "Y")->AsIntermediate();
      *ew_mul_out >> *ew_branch_add >> *ew_branch_add_out;
      *ew_branch_add_in >> *ew_branch_add;
    } else {
      ew_branch_add_out = ew_mul_out;
    }
    if (block_act_type_ != "linear") {
      ew_branch_add_out->assert_is_op_input(block_act_type_, "X")
          ->AsIntermediate();
      *ew_branch_add_out >> *block_act >> *block_act_out;
    } else {
      block_act_out = ew_branch_add_out;
    }
    block_act_out->AsOutput();
    *mul_1_w >> *mul_1;
    *mul_2_w >> *mul_2;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    auto pool_op = matched.at("pool")->stmt()->op();
    auto* scope = pool_op->scope();
    op_desc.SetType("__xpu__squeeze_excitation_block");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    if (with_branch_) {
      op_desc.SetInput("Branch", {matched.at("ew_branch_add_in")->arg()->name});
    }

    auto mul_1_w_name = matched.at("mul_1_w")->arg()->name;
    auto* mul_1_w_t = scope->FindMutableTensor(mul_1_w_name);
    auto mul_1_w_dims = mul_1_w_t->dims();
    int mul_1_w_len = mul_1_w_t->numel();
    float* mul_1_w_on_host = mul_1_w_t->mutable_data<float>();
    auto mul_2_w_name = matched.at("mul_2_w")->arg()->name;
    auto* mul_2_w_t = scope->FindMutableTensor(mul_2_w_name);
    auto mul_2_w_dims = mul_2_w_t->dims();
    int mul_2_w_len = mul_2_w_t->numel();
    float* mul_2_w_on_host = mul_2_w_t->mutable_data<float>();
    if (mul_1_w_dims[0] != mul_2_w_dims[1] ||
        mul_1_w_dims[1] != mul_2_w_dims[0]) {
      LOG(FATAL) << "Error: Dims of excitation mul1 weight is: "
                 << mul_1_w_dims[0] << ", " << mul_1_w_dims[1]
                 << ", but get dims of excitation mul2 weight is: "
                 << mul_2_w_dims[0] << ", " << mul_2_w_dims[1];
    }
    if (mul_1_w_dims[0] % mul_2_w_dims[1] != 0) {
      LOG(FATAL) << "Error: Reduction ratio of excitation is not and integer.";
    }

    op_desc.SetAttr<std::vector<int>>(
        "filter_dims",
        {static_cast<int>(mul_1_w_dims[0] / mul_1_w_dims[1]),
         static_cast<int>(mul_1_w_dims[0])});
    op_desc.SetAttr<std::vector<int>>("op_type", std::vector<int>{4});
    op_desc.SetAttr<std::vector<int>>("place_x", std::vector<int>{0});
    op_desc.SetAttr<std::vector<int>>("place_y", std::vector<int>{9});
    op_desc.SetAttr<std::vector<int>>("place_z", std::vector<int>{10});
    op_desc.SetAttr<std::vector<int>>("strides", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("paddings", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("dilations", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("groups", std::vector<int>{});
    op_desc.SetAttr<std::vector<int>>("block_lod", std::vector<int>{1});
    op_desc.SetAttr<std::vector<int>>("conv_bias", std::vector<int>{});
    op_desc.SetAttr<bool>("has_bias", false);

    std::unique_ptr<float[]> encode_filter_float(
        new float[mul_1_w_len + mul_2_w_len]);
    memcpy(encode_filter_float.get(),
           mul_1_w_on_host,
           mul_1_w_len * sizeof(float));
    memcpy(encode_filter_float.get() + mul_1_w_len,
           mul_2_w_on_host,
           mul_2_w_len * sizeof(float));
    std::string new_filter_name = "se_" + mul_1_w_name;
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
    new_filter_t->set_persistable(true);
    new_filter_t->Resize({mul_1_w_len + mul_2_w_len});
    float* new_filter_ptr = new_filter_t->mutable_data<float>();
    memcpy(new_filter_ptr,
           encode_filter_float.get(),
           (mul_1_w_len + mul_2_w_len) * sizeof(float));
    op_desc.SetInput("Filter", {new_filter_name});

    std::string output_name;
    if (block_act_type_ != "linear") {
      output_name = matched.at("block_act_out")->arg()->name;
    } else if (with_branch_) {
      output_name = matched.at("ew_branch_add_out")->arg()->name;
    } else {
      output_name = matched.at("ew_mul_out")->arg()->name;
    }
    op_desc.SetOutput("Output", {output_name});

    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* max_output_tensor = scope->NewTensor(max_output_name);
    max_output_tensor->set_precision(paddle::lite_api::PrecisionType::kFloat);
    max_output_tensor->set_persistable(true);
    op_desc.SetOutput("OutputMax", {max_output_name});

    std::map<std::string, int> act_map{
        {"linear", 0}, {"relu", 1}, {"sigmoid", 2}};

    op_desc.SetAttr<bool>("has_branch", with_branch_);
    op_desc.SetAttr<std::vector<int>>(
        "act_type",
        std::vector<int>{act_map[excitation_act_type1_],
                         act_map[excitation_act_type2_],
                         act_map[block_act_type_]});
    op_desc.SetAttr<std::vector<float>>("act_param",
                                        std::vector<float>{0, 0, 0});

    auto& valid_places = pool_op->valid_places();
    auto se_op = LiteOpRegistry::Global().Create(op_desc.Type());
    se_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(se_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    if (with_branch_) {
      DirectedLink(matched.at("ew_branch_add_in"), new_op_node);
    }
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    if (block_act_type_ != "linear") {
      IR_NODE_LINK_TO(new_op_node, matched.at("block_act_out"));
    } else if (with_branch_) {
      IR_NODE_LINK_TO(new_op_node, matched.at("ew_branch_add_out"));
    } else {
      IR_NODE_LINK_TO(new_op_node, matched.at("ew_mul_out"));
    }
    IR_NODE_LINK_TO(new_op_node, max_output_node);
  }

 private:
  std::string excitation_act_type1_;
  std::string excitation_act_type2_;
  std::string block_act_type_;
  bool with_branch_;
};

}  // namespace fusion

class XPUSqueezeExcitationFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    for (auto with_branch : {true, false}) {
      for (auto excitation_act_type1 : {"relu", "sigmoid"}) {
        for (auto excitation_act_type2 : {"relu", "sigmoid"}) {
          for (auto block_act_type : {"relu", "sigmoid", "linear"}) {
            fusion::XPUSqueezeExcitationFuser fuser(excitation_act_type1,
                                                    excitation_act_type2,
                                                    block_act_type,
                                                    with_branch);
            fuser(graph.get());
          }
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__squeeze_excitation_fuse_pass,
                  paddle::lite::mir::XPUSqueezeExcitationFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__squeeze_excitation_block");
