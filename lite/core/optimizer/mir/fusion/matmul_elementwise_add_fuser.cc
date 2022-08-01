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

#include "lite/core/optimizer/mir/fusion/matmul_elementwise_add_fuser.h"
#include <cmath>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void MatmulElementwiseAddFuser::CreatePattern() {
  // create nodes.
  auto* x = VarNode("x")->assert_is_op_input("matmul", "X");
  auto* W = VarNode("W")->assert_is_persistable_var()->assert_is_op_input(
      "matmul", "Y");
  auto* b = VarNode("b")->assert_is_persistable_var();
  auto* matmul = OpNode("matmul", "matmul")
                     ->assert_op_attr_satisfied<float>("alpha", [](float attr) {
                       return (std::fabs(attr - 1.0) < 1e-5);
                     });
  auto* matmul_out = VarNode("matmul_out");
  auto* add = OpNode("add", "elementwise_add");
  auto* Out = VarNode("Out");

  // create topology.
  std::vector<PMNode*> matmul_inputs{W, x};
  std::vector<PMNode*> add_inputs{matmul_out, b};
  matmul_inputs >> *matmul >> *matmul_out;

  // Some op specialities.
  matmul_out->AsIntermediate();
  matmul->AsIntermediate();
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

void MatmulElementwiseAddFuser::BuildPattern() {
  for (auto& node : graph_->StmtTopologicalOrder()) {
    if (node->IsStmt() && node->AsStmt().op_type() == "matmul") {
      auto* scope = node->stmt()->op()->scope();
      auto op_desc = node->stmt()->mutable_op_info();

      bool transpose_x = op_desc->GetAttr<bool>("transpose_X");
      bool transpose_y = op_desc->GetAttr<bool>("transpose_Y");
      auto arg_y_name = op_desc->Input("Y").front();
      auto& tensor_y = scope->FindVar(arg_y_name)->Get<lite::Tensor>();
      bool is_persist = tensor_y.persistable();
      if ((!transpose_x && !transpose_y) ||
          (!transpose_x && transpose_y && is_persist)) {
        CreatePattern();
        return;
      }
    }
  }
}

void MatmulElementwiseAddFuser::InsertNewNode(SSAGraph* graph,
                                              const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  auto mul = matched.at("matmul")->stmt()->op();
  auto* scope = mul->scope();
  auto& valid_places = mul->valid_places();
  fc_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("b"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

template <typename T>
void transpose(T* dst, const T* src, const int src_rows, const int src_cols) {
  CHECK(src && dst && src_rows > 0 && src_cols > 0);
  for (int r = 0; r < src_rows; ++r) {
    for (int c = 0; c < src_cols; ++c) {
      dst[c * src_rows + r] = src[r * src_cols + c];
    }
  }
}

cpp::OpDesc MatmulElementwiseAddFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("matmul")->stmt()->op_info();

  // Get the input scale from mul
  std::vector<float> x_scale_vct;
  std::vector<float> y_scale_vct;
  auto input_x_name = op_desc.Input("X").front();
  auto input_y_name = op_desc.Input("Y").front();
  bool is_quantized_op = op_desc.HasInputScale(input_x_name) &&
                         op_desc.HasInputScale(input_y_name);
  if (is_quantized_op) {
    x_scale_vct = op_desc.GetInputScale(input_x_name);
    y_scale_vct = op_desc.GetInputScale(input_y_name);
  }
  auto* scope = matched.at("matmul")->stmt()->op()->scope();
  auto x_shape = scope->FindVar(input_x_name)->Get<lite::Tensor>().dims();
  int x_num_col_dims = x_shape.size() - 1;
  VLOG(4) << "x_shape: " << x_shape;
  VLOG(4) << "y_shape: "
          << scope->FindVar(input_y_name)->Get<lite::Tensor>().dims();
  VLOG(4) << "x_num_col_dims: " << x_num_col_dims;

  bool transpose_y = op_desc.GetAttr<bool>("transpose_Y");
  if (transpose_y) {
    auto* y_t = scope->FindVar(input_y_name)->GetMutable<lite::Tensor>();
    auto y_dims = y_t->dims();
    Tensor y_t_tmp;
    y_t_tmp.CopyDataFrom(*y_t);  // in order to copy y_t's
                                 // target_,lod_,precision_, etc,. to y_t_tmp
    y_t_tmp.Resize({y_dims[1], y_dims[0]});
    const float* src = y_t->data<float>();
    float* dst = y_t_tmp.mutable_data<float>();
    transpose<float>(dst, src, y_dims[0], y_dims[1]);
    y_t->CopyDataFrom(y_t_tmp);
  }

  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("fc");
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetInput("W", {matched.at("W")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  op_desc.SetAttr("in_num_col_dims", x_num_col_dims);
  if (with_relu_) {
    op_desc.SetAttr("activation_type", std::string{"relu"});
  }

  // Set the input scale into fc
  if (is_quantized_op) {
    op_desc.SetInputScale(matched.at("x")->arg()->name, x_scale_vct);
    op_desc.SetInputScale(matched.at("W")->arg()->name, y_scale_vct);
  }

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
