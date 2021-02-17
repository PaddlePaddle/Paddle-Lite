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

/* fuse xpu_conv2d and pool2d                        */
/*                                                   */
/*                   in_Input                        */
/*        in_Filter      |     in_FilterMax          */
/*                  \    |    /                      */
/*                   \   |   /                       */
/*     in_Bias ------- __xpu__conv2d                 */
/*                       |    \                      */
/*                       |     \                     */
/*                       |      out_OutputMax        */
/*                     pool2d                        */
/*                       |                           */
/*                       |                           */
/*                   pool2d_out                      */
/*                                                   */
/*-------------------------------------------------- */
/*                                                   */
/* After the pass is applied:                        */
/*                     in_Input                      */
/*        in_Filter      |     in_FilterMax          */
/*                  \    |    /                      */
/*                   \   |   /                       */
/*     in_Bias ------- __xpu__block_fuse             */
/*                       |    \                      */
/*                       |     \                     */
/*                       |      out_OutputMax        */
/*                 out_Output                        */
/*                                                   */

class XPUConv2dPool2dFuser : public FuseBase {
 public:
  explicit XPUConv2dPool2dFuser(bool with_conv_bias) {
    with_conv_bias_ = with_conv_bias;
  }

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    auto* weight = VarNode("weight")
                       ->assert_is_op_input("__xpu__conv2d", "Filter")
                       ->assert_is_persistable_var()
                       ->AsInput();
    PMNode* bias = nullptr;
    if (with_conv_bias_) {
      bias = VarNode("bias")
                 ->assert_is_persistable_var()
                 ->assert_is_op_input("__xpu__conv2d", "Bias")
                 ->AsInput();
    }
    auto* xpu_conv =
        OpNode("xpu_conv", "__xpu__conv2d")
            ->assert_op_attr<bool>("has_branch", false)
            ->assert_op_attr<bool>("has_bias", with_conv_bias_)
            ->assert_op_attr_satisfied<int>(
                "act_type",
                [](const int& attr) {
                  return attr == 1 || attr == 2; /* support relu and sigmoid */
                })
            ->AsIntermediate();
    auto* conv_out = VarNode("conv_out")
                         ->assert_is_op_output("__xpu__conv2d", "Output")
                         ->assert_is_op_input("pool2d", "X")
                         ->AsIntermediate();
    auto* conv_out_max = VarNode("conv_out_max")
                             ->assert_is_op_output("__xpu__conv2d", "OutputMax")
                             ->AsIntermediate();
    auto pool2d_teller = [](const Node* x) -> bool {
      if (x && x->IsStmt()) {
        auto* op_info = x->stmt()->op_info();
        if (op_info->HasAttr("adaptive") &&
            op_info->GetAttr<bool>("adaptive")) {
          return false;
        }
        if (op_info->HasAttr("padding_algorithm") &&
            op_info->GetAttr<std::string>("padding_algorithm") == "SAME") {
          return false;
        }
      }
      return true;
    };
    auto* pool2d = OpNode("pool2d", "pool2d")
                       ->assert_op_attr<bool>("global_pooling", false)
                       ->assert_node_satisfied(pool2d_teller)
                       ->AsIntermediate();
    auto* pool2d_out =
        VarNode("pool2d_out")->assert_is_op_output("pool2d", "Out")->AsOutput();

    *input >> *xpu_conv >> *conv_out >> *pool2d >> *pool2d_out;
    *weight >> *xpu_conv;
    if (with_conv_bias_) {
      *bias >> *xpu_conv;
    }
    *xpu_conv >> *conv_out_max;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{"xpu_conv"};
    std::vector<std::string> filter_name{matched.at("weight")->arg()->name};

    cpp::OpDesc op_desc;
    auto conv = matched.at("xpu_conv")->stmt()->op();
    auto* scope = conv->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    auto output_name = matched.at("pool2d_out")->arg()->name;
    if (with_conv_bias_) {
      op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
    }
    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->NewTensor(max_output_name);

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter", {filter_name});
    op_desc.SetOutput("Output", {output_name});
    op_desc.SetOutput("OutputMax", {max_output_name});

    std::vector<int> place_x{0, 0};
    std::vector<int> place_y{9, 9};
    std::vector<int> place_z{10, 10};
    std::vector<int> block_lod{1, 1};
    std::vector<int> conv_bias{with_conv_bias_};
    int pooling_type = -1;
    if (matched.at("pool2d")->stmt()->op_info()->GetAttr<std::string>(
            "pooling_type") == "avg") {
      if (matched.at("pool2d")->stmt()->op_info()->GetAttr<bool>("exclusive") ==
          true) {
        pooling_type = 1;
      } else {
        pooling_type = 2;
      }
    } else {
      pooling_type = 3;
    }
    std::vector<int> op_type{0, pooling_type};
    auto conv_filter_dims = matched.at("xpu_conv")
                                ->stmt()
                                ->op_info()
                                ->GetAttr<std::vector<int>>("filter_dims");
    auto pool_kernel =
        matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "ksize");
    std::vector<int> filter_dims{conv_filter_dims[0],
                                 conv_filter_dims[1],
                                 conv_filter_dims[2],
                                 conv_filter_dims[3],
                                 pool_kernel[0],
                                 pool_kernel[1]};
    auto conv_strides = matched.at("xpu_conv")
                            ->stmt()
                            ->op_info()
                            ->GetAttr<std::vector<int>>("strides");
    auto pool_strides =
        matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "strides");
    std::vector<int> strides{
        conv_strides[0], conv_strides[1], pool_strides[0], pool_strides[1]};
    auto conv_paddings = matched.at("xpu_conv")
                             ->stmt()
                             ->op_info()
                             ->GetAttr<std::vector<int>>("paddings");
    if (conv_paddings.size() == 2) {
      for (size_t i = 0; i < conv_strides.size(); ++i) {
        int copy_pad = *(conv_paddings.begin() + 2 * i);
        conv_paddings.insert(conv_paddings.begin() + 2 * i + 1, copy_pad);
      }
    }
    CHECK_EQ(conv_paddings.size(), 4UL)
        << "Paddings size should be 2 or 4, But received paddings size: "
        << conv_paddings.size();
    auto pool_paddings =
        matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
            "paddings");
    if (pool_paddings.size() == 2) {
      for (size_t i = 0; i < pool_strides.size(); ++i) {
        int copy_pad = *(pool_paddings.begin() + 2 * i);
        pool_paddings.insert(pool_paddings.begin() + 2 * i + 1, copy_pad);
      }
    }
    CHECK_EQ(conv_paddings.size(), 4UL)
        << "Paddings size should be 2 or 4, But received paddings size: "
        << conv_paddings.size();
    if ((matched.at("pool2d")->stmt()->op_info()->HasAttr(
            "padding_algorithm")) &&
        (matched.at("pool2d")->stmt()->op_info()->GetAttr<std::string>(
             "padding_algorithm") == "VALID")) {
      pool_paddings[0] = 0;
      pool_paddings[1] = 0;
      pool_paddings[2] = 0;
      pool_paddings[3] = 0;
    }
    if ((matched.at("pool2d")->stmt()->op_info()->HasAttr("ceil_mode")) &&
        (matched.at("pool2d")->stmt()->op_info()->GetAttr<bool>("ceil_mode"))) {
      pool_paddings[1] += pool_strides[0] - 1;
      pool_paddings[3] += pool_strides[1] - 1;
    }
    std::vector<int> paddings{conv_paddings[0],
                              conv_paddings[1],
                              conv_paddings[2],
                              conv_paddings[3],
                              pool_paddings[0],
                              pool_paddings[1],
                              pool_paddings[2],
                              pool_paddings[3]};
    auto conv_dilations = matched.at("xpu_conv")
                              ->stmt()
                              ->op_info()
                              ->GetAttr<std::vector<int>>("dilations");
    std::vector<int> dilations{conv_dilations[0], conv_dilations[1]};
    auto conv_groups =
        matched.at("xpu_conv")->stmt()->op_info()->GetAttr<int>("groups");
    std::vector<int> groups{conv_groups};
    auto conv_act_type =
        matched.at("xpu_conv")->stmt()->op_info()->GetAttr<int>("act_type");
    std::vector<int> act_type{conv_act_type};
    auto conv_act_param =
        matched.at("xpu_conv")->stmt()->op_info()->GetAttr<float>("act_param");
    std::vector<float> act_param{conv_act_param};
    op_desc.SetAttr("op_type", op_type);
    op_desc.SetAttr("place_x", place_x);
    op_desc.SetAttr("place_y", place_y);
    op_desc.SetAttr("place_z", place_z);
    op_desc.SetAttr("filter_dims", filter_dims);
    op_desc.SetAttr("strides", strides);
    op_desc.SetAttr("paddings", paddings);
    op_desc.SetAttr("dilations", dilations);
    op_desc.SetAttr("groups", groups);
    op_desc.SetAttr("act_type", act_type);
    op_desc.SetAttr("act_param", act_param);
    op_desc.SetAttr("block_lod", block_lod);
    op_desc.SetAttr("conv_bias", conv_bias);
    op_desc.SetAttr<bool>("has_bias", with_conv_bias_);

    auto& valid_places = conv->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(matched.at("weight"), new_op_node);
    if (with_conv_bias_) {
      IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at("pool2d_out"));
    IR_NODE_LINK_TO(new_op_node, max_output_node);
  }

 private:
  bool with_conv_bias_;
};

}  // namespace fusion

class XPUConv2dPool2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto with_conv_bias : {true, false}) {
      fusion::XPUConv2dPool2dFuser fuser(with_conv_bias);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_pool2d_fuse_pass,
                  paddle::lite::mir::XPUConv2dPool2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");
