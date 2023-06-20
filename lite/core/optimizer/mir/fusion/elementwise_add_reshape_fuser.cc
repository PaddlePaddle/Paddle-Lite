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

#include "lite/core/optimizer/mir/fusion/elementwise_add_reshape_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ElementwiseReshapeFuser::BuildPattern() {
 
  LOG(INFO) << "===》 lkk BuildPattern reshape_type_ " << reshape_type_ << " elt_type_ " << eltwise_type_;

  auto* reshape_in = VarNode("reshape_in")->assert_is_op_input(reshape_type_,"X");

  auto* x = VarNode("x")
            ->assert_is_op_input(eltwise_type_, "X")
            ->assert_is_op_output("conv2d","Output")
            ->AsInput();
  
  // auto* reshape2_xshape = VarNode("reshape2_xt_type_
  //                                 ->assert_is_op_output("reshape2", "XShape")
  //                                 ->AsIntermediate();


  // create intermediate nodes
  auto* y = VarNode("y")
                      ->assert_is_op_output(reshape_type_, "Out")
                      ->assert_is_op_input(eltwise_type_, "Y")
                      // ->assert_is_persistable_var()
                      // ->assert_only_one_output()
                      ->AsIntermediate();
  auto* reshape_xshape = VarNode("reshape_xshape");

  // create op nodes
  auto* reshape = OpNode("reshape", reshape_type_)
                  ->assert_is_op(reshape_type_)
                  // ->assert_op_attr<int>("axis", -1)
                  // ->assert_op_attr_satisfied<std::vector<int>>("shape",
                  //     [](const std::vector<int>& attr) { return attr.size() == 2; })
                  ->AsIntermediate();
  auto* elt = OpNode("elt", eltwise_type_)
                  ->assert_is_op(eltwise_type_)
                  ->AsIntermediate();


  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(eltwise_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> reshape_outputs{y, reshape_xshape};
  std::vector<PMNode*> elt_inputs{x, y};
  *reshape_in >> *reshape >> reshape_outputs;
  elt_inputs >> *elt >> *out;
  // *reshape >> *reshape2_xshape;
}

void ElementwiseReshapeFuser::InsertNewNode(SSAGraph* graph,
                                          const key2nodes_t& matched) {
  LOG(INFO) << "===》 lkk InsertNewNode  ";

  auto op_desc = GenOpDesc(matched);
  std::shared_ptr<lite::OpLite> op;
  if (eltwise_type_ == "elementwise_add") {
    op = LiteOpRegistry::Global().Create("elementwise_add");
  } else {
    LOG(FATAL) << "not supported elementwise_type: " << eltwise_type_;
  }
  

  auto old_op = matched.at("elt")->stmt()->op();
  auto* scope = old_op->scope(); 

  auto filter_name = matched.at("reshape_in")->arg()->name;
  auto* filter_t = scope->FindMutableTensor(filter_name);
  auto& f_dims = filter_t->dims();
  LOG(INFO) << "===> lkk filter " << filter_name << " dims: " << f_dims[0] << " " << f_dims[1] << " " << f_dims[2] << " " << f_dims[3];

  std::string fusion_bias_name = filter_name;
  auto* fusion_bias_node = graph->NewArgumentNode(fusion_bias_name);
  fusion_bias_node->arg()->is_weight = true;
  fusion_bias_node->arg()->type = LiteType::GetTensorTy(
      TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
  auto* fusion_bias_t = scope->MutableParent()->NewTensor(fusion_bias_name);
  fusion_bias_t->set_precision(paddle::lite_api::PrecisionType::kFloat);
  fusion_bias_t->Resize({f_dims[0]});

  float* fusion_bias_ptr = fusion_bias_t->mutable_data<float>();
  auto ew_bias_add_y_name = matched.at("y")->arg()->name;
  auto* ew_bias_add_y_t = scope->FindMutableTensor(ew_bias_add_y_name);
  float* ew_bias_add_y_on_host = ew_bias_add_y_t->mutable_data<float>();
  auto ew_bias_add_y_size = ew_bias_add_y_t->numel();
  if (ew_bias_add_y_size != f_dims[0] && ew_bias_add_y_size == 1) {
    for (int i = 0; i < f_dims[0]; ++i) {
      fusion_bias_ptr[i] = ew_bias_add_y_on_host[0];
    }
  } else if (ew_bias_add_y_size == f_dims[0]) {
    for (int i = 0; i < f_dims[0]; ++i) {
      fusion_bias_ptr[i] = ew_bias_add_y_on_host[i];
    }
  } else {
    LOG(WARNING)
        << "Elements size of `elemwise_bias` and 'conv_filter_channels` "
            "should be the same, but get size of `elemwise_bias` "
            "is: "
        << ew_bias_add_y_size
        << ", size of `conv_filter_channels` is: " << f_dims[0];
    return;
  }
  fusion_bias_t->set_persistable(true);
  op_desc.SetInput("Y", {fusion_bias_name});

  auto& valid_places = old_op->valid_places();
  op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("reshape_in"), new_op_node);
  DirectedLink(fusion_bias_node, new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ElementwiseReshapeFuser::GenOpDesc(const key2nodes_t& matched) {
    LOG(INFO) << "===》 lkk GenOpDesc  ";
  // 获取reshape_op_desc 的属性新增到op_desc中并进行返回
  auto op_desc = *matched.at("elt")->stmt()->op_info();
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  // auto* reshape_op_desc = matched.at("reshape")->stmt()->op_info();
  op_desc.SetAttr("axis", 1);
  // op_desc.SetAttr("shape", reshape_op_desc->GetAttr<>("shape")); // czh delete?
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
