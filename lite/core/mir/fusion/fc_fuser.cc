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

#include "lite/core/mir/fusion/fc_fuser.h"
#include <memory>
#include <vector>
#include "lite/kernels/arm/fc_compute.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

///////////////////////////////////////////////////////////////////////////////
// Function: TransFcWeights
// Usage: Judge if GEMM is used. If GEMM is used in FC, corresponding weight
// data will be transposed.
///////////////////////////////////////////////////////////////////////////////
void TransFcWeights(Tensor* weight,
                    Tensor* input,
                    Tensor* output,
                    Tensor* bias,
                    int m_,
                    const std::vector<float>& scale) {
  auto bias_flag = bias->mutable_data<bool>();
  auto x_dims = input->dims();

#define CHECK_FC_USE_GEMM(input_type__, output_type__)                       \
  flag_gemm_ = paddle::lite::kernels::arm::check_fc_use_gemm<input_type__,   \
                                                             output_type__>( \
      m_, scale, bias_flag);

  bool flag_gemm_;
  switch (input->precision()) {
    case PRECISION(kFloat):
      flag_gemm_ = CHECK_FC_USE_GEMM(PRECISION(kFloat), PRECISION(kFloat));
      break;
    case PRECISION(kInt8):
      switch (output->precision()) {
        case PRECISION(kFloat):
          flag_gemm_ = CHECK_FC_USE_GEMM(PRECISION(kInt8), PRECISION(kFloat));
          break;
        case PRECISION(kInt8):
          flag_gemm_ = CHECK_FC_USE_GEMM(PRECISION(kInt8), PRECISION(kInt8));
          break;
        default:
          LOG(FATAL) << "Unsupported output precision type";
      }
      break;
    default:
      LOG(FATAL) << "Unsupported input precision type";
  }

  if (!flag_gemm_) {
    Tensor tmp_tensor;
    paddle::lite::kernels::arm::fc_trans_weights<PRECISION(kInt8)>(*weight,
                                                                   &tmp_tensor);
    weight->CopyDataFrom(tmp_tensor);
  }
}

void FcFuser::BuildPattern() {
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

void FcFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  auto mul = matched.at("mul")->stmt()->op();
  auto* scope = mul->scope();
  auto& valid_places = mul->valid_places();
  fc_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("b"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

cpp::OpDesc FcFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("mul")->stmt()->op_info();

  // Get the input scale from mul
  std::vector<float> x_scale_vct;
  std::vector<float> y_scale_vct;
  auto input_x_name = op_desc.Input("X").front();
  auto input_y_name = op_desc.Input("Y").front();
  bool is_quantized_op = op_desc.HasInputScale(input_x_name) &&
                         op_desc.HasInputScale(input_y_name);
  if (is_quantized_op) {
    x_scale_vct = op_desc.GetInputScale(input_x_name);
    y_scale_vct = op_desc.GetInputScale(op_desc.Input("Y").front());
  }

  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("fc");
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetInput("W", {matched.at("W")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  op_desc.SetAttr(
      "in_num_col_dims",
      matched.at("mul")->stmt()->op_info()->GetAttr<int>("x_num_col_dims"));
  if (with_relu_) {
    op_desc.SetAttr("activation_type", std::string{"relu"});
  }

  // Set the input scale into fc
  if (is_quantized_op) {
    op_desc.SetInputScale(matched.at("x")->arg()->name, x_scale_vct);
    op_desc.SetInputScale(matched.at("W")->arg()->name, y_scale_vct);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Judge if GEMM is used in FC.
  ///////////////////////////////////////////////////////////////////////////////
  auto* scope = matched.at("W")->stmt()->op()->scope();
  auto* weight =
      scope->FindVar(matched.at("W")->arg()->name)->GetMutable<lite::Tensor>();
  auto* input = scope->FindVar(matched.at("Input")->arg()->name)
                    ->GetMutable<lite::Tensor>();
  auto* output = scope->FindVar(matched.at("Out")->arg()->name)
                     ->GetMutable<lite::Tensor>();
  auto* bias =
      scope->FindVar(matched.at("b")->arg()->name)->GetMutable<lite::Tensor>();
  auto x_dims = input->dims();
  auto m_ = x_dims
                .Slice(0,
                       matched.at("mul")->stmt()->op_info()->GetAttr<int>(
                           "x_num_col_dims"))
                .production();
  auto weight_scale = op_desc.GetInputScale(matched.at("W")->arg()->name);
  TransFcWeights(weight, input, output, bias, m_, weight_scale);

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
