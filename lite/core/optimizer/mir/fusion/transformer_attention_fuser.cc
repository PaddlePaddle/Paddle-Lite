// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fusion/transformer_attention_fuser.h"
#include <memory>
#include <string>
#include <vector>
namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/*
*                       input
*           /             |             \
*          /              |              \
*         /               |               \
*       fc               fc               fc
*       |                 |                |
*       |                 |                |
*     reshape2        reshape2          reshape2
*       |                 |                |
*       |                 |                |
*   transpose2       transpose2        transpose2
*       |                 |                |
*       |                 |                |
*     scale               |                |
*         \              /                 |
*          \            /                  |
*          matmul_v2/matmul                |
*              \                           /
*               \                         /
*         elementwise_add                /
*                 \                     /
*                  \                   /
*                softmax              /
*                   |                /
*                   |               /
*                dropout           /
*                    \            /
*                     \          /
*                    matmul_v2/matmul
*                           |
*                           |
*                         output
*/

void TransformerAttentionFuser::BuildPattern() {
  auto matmul0_attr_teller = [](const Node* node) -> bool {
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    bool trans_x;
    bool trans_y;
    if (op_desc.Type() == "matmul") {
      trans_x = op_desc.GetAttr<bool>("transpose_X");
      trans_y = op_desc.GetAttr<bool>("transpose_Y");
    } else {
      trans_x = op_desc.GetAttr<bool>("trans_x");
      trans_y = op_desc.GetAttr<bool>("trans_y");
    }
    auto res = (trans_x == false && trans_y == true);
    return res;
  };
  auto matmul1_attr_teller = [](const Node* node) -> bool {
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    bool trans_x;
    bool trans_y;
    if (op_desc.Type() == "matmul") {
      trans_x = op_desc.GetAttr<bool>("transpose_X");
      trans_y = op_desc.GetAttr<bool>("transpose_Y");
    } else {
      trans_x = op_desc.GetAttr<bool>("trans_x");
      trans_y = op_desc.GetAttr<bool>("trans_y");
    }
    auto res = (trans_x == false && trans_y == false);
    return res;
  };
  auto* input = VarNode("input")->assert_is_op_input("fc", "Input")->AsInput();
  // fc
  auto* fc0_w = VarNode("fc0_w")->assert_is_op_input("fc", "W");
  auto* fc0_bias = VarNode("fc0_bias")->assert_is_op_input("fc", "Bias");
  auto* fc0 = OpNode("fc0", "fc");
  auto* fc0_out = VarNode("fc0_out")->assert_is_op_output("fc", "Out");

  auto* fc1_w = VarNode("fc1_w")->assert_is_op_input("fc", "W");
  auto* fc1_bias = VarNode("fc1_bias")->assert_is_op_input("fc", "Bias");
  auto* fc1 = OpNode("fc1", "fc");
  auto* fc1_out = VarNode("fc1_out")->assert_is_op_output("fc", "Out");

  auto* fc2_w = VarNode("fc2_w")->assert_is_op_input("fc", "W");
  auto* fc2_bias = VarNode("fc2_bias")->assert_is_op_input("fc", "Bias");
  auto* fc2 = OpNode("fc2", "fc");
  auto* fc2_out = VarNode("fc2_out")->assert_is_op_output("fc", "Out");

  // reshape2
  auto* reshape0 = OpNode("reshape0", "reshape2");
  auto* reshape0_out =
      VarNode("reshape0_out")->assert_is_op_output("reshape2", "Out");

  auto* reshape1 = OpNode("reshape1", "reshape2");
  auto* reshape1_out =
      VarNode("reshape1_out")->assert_is_op_output("reshape2", "Out");

  auto* reshape2 = OpNode("reshape2", "reshape2");
  auto* reshape2_out =
      VarNode("reshape2_out")->assert_is_op_output("reshape2", "Out");

  PMNode* xshape0 = nullptr;
  PMNode* xshape1 = nullptr;
  PMNode* xshape2 = nullptr;
  if (reshape_has_xshape_) {
    xshape0 = VarNode("xshape0")->assert_is_op_output("reshape2", "XShape");
    xshape1 = VarNode("xshape1")->assert_is_op_output("reshape2", "XShape");
    xshape2 = VarNode("xshape2")->assert_is_op_output("reshape2", "XShape");
  }

  // transpose2
  auto* transpose0 = OpNode("transpose0", "transpose2")
                         ->assert_op_attr("axis", std::vector<int>{0, 2, 1, 3});
  auto* transpose0_out =
      VarNode("transpose0_out")->assert_is_op_output("transpose2", "Out");

  auto* transpose1 = OpNode("transpose1", "transpose2")
                         ->assert_op_attr("axis", std::vector<int>{0, 2, 1, 3});
  auto* transpose1_out =
      VarNode("transpose1_out")->assert_is_op_output("transpose2", "Out");

  auto* transpose2 = OpNode("transpose2", "transpose2")
                         ->assert_op_attr("axis", std::vector<int>{0, 2, 1, 3});
  auto* transpose2_out =
      VarNode("transpose2_out")->assert_is_op_output("transpose2", "Out");

  PMNode* xshape3 = nullptr;
  PMNode* xshape4 = nullptr;
  PMNode* xshape5 = nullptr;
  if (transpose_has_xshape_) {
    xshape3 = VarNode("xshape3")->assert_is_op_output("transpose2", "XShape");
    xshape4 = VarNode("xshape4")->assert_is_op_output("transpose2", "XShape");
    xshape5 = VarNode("xshape5")->assert_is_op_output("transpose2", "XShape");
  }

  // scale
  auto* scale0 = OpNode("scale0", "scale");
  auto* scale0_out = VarNode("scale0_out")->assert_is_op_output("scale", "Out");

  // matmul
  auto* matmul0 =
      OpNode("matmul0", mul_type_)->assert_node_satisfied(matmul0_attr_teller);
  auto* matmul0_out =
      VarNode("matmul0_out")->assert_is_op_output(mul_type_, "Out");

  // elementwise_add
  auto* residual = VarNode("residual")
                       ->assert_is_op_input("elementwise_add", "Y")
                       ->AsInput();
  auto* add = OpNode("add", "elementwise_add");
  auto* add0_out =
      VarNode("add0_out")->assert_is_op_output("elementwise_add", "Out");

  // softmax
  auto* softmax0 = OpNode("softmax0", "softmax");
  auto* softmax0_out =
      VarNode("softmax0_out")->assert_is_op_output("softmax", "Out");

  // dropout
  auto* dropout = OpNode("dropout", "dropout");
  auto* dropout_out =
      VarNode("dropout_out")->assert_is_op_output("dropout", "Out");
  PMNode* mask_out = nullptr;
  if (dropout_mask_) {
    mask_out = VarNode("mask_out")->assert_is_op_output("dropout", "Mask");
  }

  // matmul
  auto* matmul1 =
      OpNode("matmul1", mul_type_)->assert_node_satisfied(matmul1_attr_teller);

  auto* Out = VarNode("Out");

  std::vector<PMNode*> fc0_inputs{input, fc0_w, fc0_bias};
  std::vector<PMNode*> fc1_inputs{input, fc1_w, fc1_bias};
  std::vector<PMNode*> fc2_inputs{input, fc2_w, fc2_bias};
  fc0_inputs >> *fc0 >> *fc0_out >> *reshape0 >> *reshape0_out >> *transpose0 >>
      *transpose0_out >> *scale0 >> *scale0_out;
  fc1_inputs >> *fc1 >> *fc1_out >> *reshape1 >> *reshape1_out >> *transpose1 >>
      *transpose1_out;
  fc2_inputs >> *fc2 >> *fc2_out >> *reshape2 >> *reshape2_out >> *transpose2 >>
      *transpose2_out;
  if (reshape_has_xshape_) {
    *reshape0 >> *xshape0;
    *reshape1 >> *xshape1;
    *reshape2 >> *xshape2;
  }
  if (transpose_has_xshape_) {
    *transpose0 >> *xshape3;
    *transpose1 >> *xshape4;
    *transpose2 >> *xshape5;
  }

  std::vector<PMNode*> matmul0_inputs{scale0_out, transpose1_out};
  matmul0_inputs >> *matmul0 >> *matmul0_out;
  std::vector<PMNode*> add0_inputs{matmul0_out, residual};
  add0_inputs >> *add >> *add0_out >> *softmax0 >> *softmax0_out >> *dropout >>
      *dropout_out;

  if (dropout_mask_) {
    *dropout >> *mask_out;
  }

  std::vector<PMNode*> matmul1_inputs{dropout_out, transpose2_out};
  matmul1_inputs >> *matmul1 >> *Out;

  if (reshape_has_xshape_) {
    xshape0->AsIntermediate();
    xshape1->AsIntermediate();
    xshape2->AsIntermediate();
  }
  if (transpose_has_xshape_) {
    xshape3->AsIntermediate();
    xshape4->AsIntermediate();
    xshape5->AsIntermediate();
  }
  if (dropout_mask_) {
    mask_out->AsIntermediate();
  }
  fc0->AsIntermediate();
  fc0_out->AsIntermediate();
  reshape0->AsIntermediate();
  reshape0_out->AsIntermediate();
  transpose0->AsIntermediate();
  transpose0_out->AsIntermediate();
  fc1->AsIntermediate();
  fc1_out->AsIntermediate();
  reshape1->AsIntermediate();
  reshape1_out->AsIntermediate();
  transpose1->AsIntermediate();
  transpose1_out->AsIntermediate();
  fc2->AsIntermediate();
  fc2_out->AsIntermediate();
  reshape2->AsIntermediate();
  reshape2_out->AsIntermediate();
  transpose2->AsIntermediate();
  transpose2_out->AsIntermediate();
  scale0->AsIntermediate();
  scale0_out->AsIntermediate();
  matmul0->AsIntermediate();
  matmul0_out->AsIntermediate();
  add->AsIntermediate();
  add0_out->AsIntermediate();
  softmax0->AsIntermediate();
  softmax0_out->AsIntermediate();
  dropout->AsIntermediate();
  dropout_out->AsIntermediate();
  matmul1->AsIntermediate();
}

template <typename T>
void ComputeNewWeight(lite::Tensor* dout,
                      lite::Tensor* din0,
                      lite::Tensor* din1,
                      lite::Tensor* din2,
                      int ih,
                      int iw) {
  T* fuse_weights = dout->mutable_data<T>();
  const T* weight0_data = din0->data<T>();
  const T* weight1_data = din1->data<T>();
  const T* weight2_data = din2->data<T>();
  for (int h = 0; h < ih; h++) {
    memcpy(fuse_weights, weight0_data, iw * sizeof(T));
    fuse_weights += iw;
    weight0_data += iw;
    memcpy(fuse_weights, weight1_data, iw * sizeof(T));
    fuse_weights += iw;
    weight1_data += iw;
    memcpy(fuse_weights, weight2_data, iw * sizeof(T));
    fuse_weights += iw;
    weight2_data += iw;
  }
}

void ComputeNewBias(lite::Tensor* dout,
                    lite::Tensor* din0,
                    lite::Tensor* din1,
                    lite::Tensor* din2,
                    int iw) {
  auto bias0_data = din0->data<float>();
  auto bias1_data = din1->data<float>();
  auto bias2_data = din2->data<float>();
  float* fuse_bias = dout->mutable_data<float>();
  memcpy(fuse_bias, bias0_data, iw * sizeof(float));
  fuse_bias += iw;
  memcpy(fuse_bias, bias1_data, iw * sizeof(float));
  fuse_bias += iw;
  memcpy(fuse_bias, bias2_data, iw * sizeof(float));
}

void ComputeNewBias(lite::Tensor* dout,
                    lite::Tensor* din0,
                    lite::Tensor* din1,
                    lite::Tensor* din2,
                    const float scale0_scale,
                    const float* calib3_scale,
                    const float* calib4_scale,
                    const float* calib5_scale,
                    int iw) {
  auto bias0_data = din0->data<float>();
  auto bias1_data = din1->data<float>();
  auto bias2_data = din2->data<float>();
  float* fuse_bias = dout->mutable_data<float>();
  for (int i = 0; i < iw; i++) {
    fuse_bias[i] = bias0_data[i] * scale0_scale / calib3_scale[0];
  }
  fuse_bias += iw;
  for (int i = 0; i < iw; i++) {
    fuse_bias[i] = bias1_data[i] / calib4_scale[0];
  }
  fuse_bias += iw;
  for (int i = 0; i < iw; i++) {
    fuse_bias[i] = bias2_data[i] / calib5_scale[0];
  }
}

void ComputeNewScale(float* fuse_scales,
                     const float* fc0_scale_x,
                     const float* fc0_scale_y,
                     const float* fc1_scale_x,
                     const float* fc1_scale_y,
                     const float* fc2_scale_x,
                     const float* fc2_scale_y,
                     const float* calib3_scale,
                     const float* calib4_scale,
                     const float* calib5_scale,
                     const float scale0_scale,
                     int iw) {
  for (int w = 0; w < iw; w++) {
    fuse_scales[w] =
        fc0_scale_x[0] * fc0_scale_y[w] * scale0_scale / calib3_scale[0];
  }
  fuse_scales += iw;
  for (int w = 0; w < iw; w++) {
    fuse_scales[w] = fc1_scale_x[0] * fc1_scale_y[w] / calib4_scale[0];
  }
  fuse_scales += iw;
  for (int w = 0; w < iw; w++) {
    fuse_scales[w] = fc2_scale_x[0] * fc2_scale_y[w] / calib5_scale[0];
  }
}

void TransformerAttentionFuser::InsertNewNode(SSAGraph* graph,
                                              const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("fused_attention");

  auto fc = matched.at("fc0")->stmt()->op();
  auto* scope = fc->scope();

  // set input
  op_desc.SetInput("Input", {matched.at("input")->arg()->name});
  op_desc.SetInput("Residual", {matched.at("residual")->arg()->name});

  // fc
  auto fc0_op_desc = matched.at("fc0")->stmt()->op_info();
  auto fc1_op_desc = matched.at("fc1")->stmt()->op_info();
  auto fc2_op_desc = matched.at("fc2")->stmt()->op_info();
  bool enable_int8 = fc0_op_desc->HasAttr("enable_int8") &&
                     fc0_op_desc->GetAttr<bool>("enable_int8");
  // concat fc0_w fc1_w fc2_w
  auto weight0_t = scope->FindMutableTensor(matched.at("fc0_w")->arg()->name);
  auto weight1_t = scope->FindMutableTensor(matched.at("fc1_w")->arg()->name);
  auto weight2_t = scope->FindMutableTensor(matched.at("fc2_w")->arg()->name);
  auto weight0_dims = weight0_t->dims();
  Tensor weight_tensor;
  weight_tensor.Resize({weight0_dims[0], weight0_dims[1] * 3});
  // concat fc0_bias fc1_bias fc2_bias
  auto bias0_t = scope->FindMutableTensor(matched.at("fc0_bias")->arg()->name);
  auto bias1_t = scope->FindMutableTensor(matched.at("fc1_bias")->arg()->name);
  auto bias2_t = scope->FindMutableTensor(matched.at("fc2_bias")->arg()->name);
  auto bias0_dims = bias0_t->dims();
  Tensor bias_tensor;
  bias_tensor.Resize({bias0_dims[0] * 3});

  auto matmul0_op_desc = matched.at("matmul0")->stmt()->op_info();
  auto matmul1_op_desc = matched.at("matmul1")->stmt()->op_info();
  auto scale0_op_desc = matched.at("scale0")->stmt()->op_info();
  auto scale0_scale = scale0_op_desc->GetAttr<float>("scale");
  if (enable_int8) {
    op_desc.SetAttr<bool>("enable_int8", true);
    op_desc.SetAttr<std::vector<float>>(
        "calib0_scale", fc0_op_desc->GetAttr<std::vector<float>>("X0_scale"));
    ComputeNewWeight<int8_t>(&weight_tensor,
                             weight0_t,
                             weight1_t,
                             weight2_t,
                             weight0_dims[0],
                             weight0_dims[1]);
    auto calib3_scale =
        matmul0_op_desc->GetAttr<std::vector<float>>("X0_scale");
    auto calib4_scale =
        matmul0_op_desc->GetAttr<std::vector<float>>("Y0_scale");
    auto matmul1_scale_x =
        matmul1_op_desc->GetAttr<std::vector<float>>("X0_scale");
    auto matmul1_scale_y =
        matmul1_op_desc->GetAttr<std::vector<float>>("Y0_scale");
    op_desc.SetAttr<std::vector<float>>("calib1_scale", matmul1_scale_x);
    auto calib5_scale = matmul1_scale_y;
    ComputeNewBias(&bias_tensor,
                   bias0_t,
                   bias1_t,
                   bias2_t,
                   scale0_scale,
                   calib3_scale.data(),
                   calib4_scale.data(),
                   calib5_scale.data(),
                   bias0_dims[0]);
    // concat scale
    auto fc0_scale_x = fc0_op_desc->GetAttr<std::vector<float>>("X0_scale");
    auto fc0_scale_y = fc0_op_desc->GetAttr<std::vector<float>>("Y0_scale");

    auto fc1_scale_x = fc1_op_desc->GetAttr<std::vector<float>>("X0_scale");
    auto fc1_scale_y = fc1_op_desc->GetAttr<std::vector<float>>("Y0_scale");

    auto fc2_scale_x = fc2_op_desc->GetAttr<std::vector<float>>("X0_scale");
    auto fc2_scale_y = fc2_op_desc->GetAttr<std::vector<float>>("Y0_scale");

    std::vector<float> fuse_scales(bias0_dims[0] * 3);
    ComputeNewScale(fuse_scales.data(),
                    fc0_scale_x.data(),
                    fc0_scale_y.data(),
                    fc1_scale_x.data(),
                    fc1_scale_y.data(),
                    fc2_scale_x.data(),
                    fc2_scale_y.data(),
                    calib3_scale.data(),
                    calib4_scale.data(),
                    calib5_scale.data(),
                    scale0_scale,
                    bias0_dims[0]);
    op_desc.SetAttr<std::vector<float>>("fc0_scale", fuse_scales);
    op_desc.SetAttr<std::vector<float>>("Input0_scale", fc0_scale_x);
    // fc 1
    auto matmul0_scale_x =
        matmul0_op_desc->GetAttr<std::vector<float>>("X0_scale");
    auto matmul0_scale_y =
        matmul0_op_desc->GetAttr<std::vector<float>>("Y0_scale");
    std::vector<float> fc1_scale_data;
    fc1_scale_data.push_back(matmul0_scale_x[0] * matmul0_scale_y[0]);
    op_desc.SetAttr<std::vector<float>>("fc1_scale", fc1_scale_data);
    // fc 2 == matmul 1
    std::vector<float> fc2_scale_data;
    fc2_scale_data.push_back(matmul1_scale_x[0] * matmul1_scale_y[0]);
    op_desc.SetAttr<std::vector<float>>("fc2_scale", fc2_scale_data);
  } else {
    ComputeNewWeight<float>(&weight_tensor,
                            weight0_t,
                            weight1_t,
                            weight2_t,
                            weight0_dims[0],
                            weight0_dims[1]);
    ComputeNewBias(&bias_tensor, bias0_t, bias1_t, bias2_t, bias0_dims[0]);
    op_desc.SetAttr<float>("scale", scale0_scale);
  }
  // update weight bias
  weight0_t->Resize({weight0_dims[0], weight0_dims[1] * 3});
  weight0_t->CopyDataFrom(weight_tensor);
  bias0_t->Resize({bias0_dims[0] * 3});
  bias0_t->CopyDataFrom(bias_tensor);
  op_desc.SetInput("W", {matched.at("fc0_w")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("fc0_bias")->arg()->name});
  op_desc.SetAttr<std::string>("op_type",
                               fc0_op_desc->GetAttr<std::string>("op_type"));
  op_desc.SetAttr<int32_t>("in_num_col_dims",
                           fc0_op_desc->GetAttr<int32_t>("in_num_col_dims"));
  // reshape
  auto reshape_op_desc = matched.at("reshape0")->stmt()->op_info();
  op_desc.SetAttr<std::vector<int>>(
      "reshape_shape", reshape_op_desc->GetAttr<std::vector<int>>("shape"));

  // softmax
  auto softmax_op_desc = matched.at("softmax0")->stmt()->op_info();
  op_desc.SetAttr<int>("softmax_axis", softmax_op_desc->GetAttr<int>("axis"));

  // set output
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  auto fused_attention_op = LiteOpRegistry::Global().Create("fused_attention");
  auto& valid_places = fc->valid_places();
  fused_attention_op->Attach(op_desc, scope);
  auto* new_op_node =
      graph->GraphCreateInstructNode(fused_attention_op, valid_places);
  DirectedLink(matched.at("input"), new_op_node);
  DirectedLink(matched.at("residual"), new_op_node);
  DirectedLink(matched.at("fc0_w"), new_op_node);
  DirectedLink(matched.at("fc0_bias"), new_op_node);
  DirectedLink(new_op_node, matched.at("Out"));
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
