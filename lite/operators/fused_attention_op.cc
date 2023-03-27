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

#include "lite/operators/fused_attention_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FusedAttentionOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.residual);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.fc_w);

  const auto input_dims = param_.input->dims();
  const auto w_dims = param_.fc_w->dims();
  CHECK_EQ_OR_FALSE(w_dims.size(), 2UL);
  int64_t w_dims_1 = param_.padding_weights ? w_dims[1] - 4 : w_dims[1];
  if (param_.fc_bias) {
    const auto bias_dims = param_.fc_bias->dims();
    if (bias_dims.size() == 2) {
      CHECK_EQ_OR_FALSE(bias_dims[0], 1);
      CHECK_EQ_OR_FALSE(bias_dims[1], w_dims_1);
    } else if (bias_dims.size() == 1) {
      CHECK_EQ_OR_FALSE(bias_dims[0], w_dims_1);
    }
  }
  std::string op_type = param_.op_type;
  if (op_type == "matmul" || op_type == "matmul_v2") {
    CHECK_GE_OR_FALSE(input_dims.size(),
                      static_cast<size_t>(param_.in_num_col_dims));
    CHECK_EQ_OR_FALSE(w_dims[0], input_dims[input_dims.size() - 1]);
  } else {
    CHECK_GT_OR_FALSE(input_dims.size(),
                      static_cast<size_t>(param_.in_num_col_dims));
  }
  return true;
}

static bool CheckPositive(const DDim &dims) {
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) {
      return false;
    }
  }
  return true;
}

bool FusedAttentionOpLite::InferShape() {
  lite::DDim x_dims = param_.input->dims();

  // infer fc
  int in_num_col_dims = param_.in_num_col_dims;
  std::string op_type = param_.op_type;
  const auto &w_dims = param_.fc_w->dims();
  int64_t w_dims_1 = w_dims[1] / 3;

  if (op_type == "matmul" || op_type == "matmul_v2") {
    in_num_col_dims = x_dims.size() - 1;
  }
  DDim::value_type fc_output_size = 1;
  std::vector<DDim::value_type> fc_output_dims(in_num_col_dims + 1);
  for (int i = 0; i < in_num_col_dims; ++i) {
    fc_output_dims[i] = x_dims[i];
    fc_output_size *= fc_output_dims[i];
  }
  fc_output_dims[in_num_col_dims] = w_dims_1;
  fc_output_size *= fc_output_dims[in_num_col_dims];

  std::vector<int> shape = param_.reshape_shape;
  std::vector<DDim::value_type> reshape_output_dims(shape.size());
  DDim::value_type capacity = 1;
  const int unk_dim_val = -1;
  const int copy_dim_val = 0;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(i, fc_output_dims.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }

    DDim::value_type output_dim_i =
        shape[i] ? static_cast<DDim::value_type>(shape[i]) : fc_output_dims[i];
    reshape_output_dims[i] = output_dim_i;
    capacity *= output_dim_i;
  }
  if (unk_dim_idx != -1) {
    if (CheckPositive(lite::DDim(fc_output_dims))) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      reshape_output_dims[unk_dim_idx] = -fc_output_size / capacity;
      CHECK_EQ(reshape_output_dims[unk_dim_idx] * capacity, -fc_output_size)
          << "Invalid shape is given.";
    } else {
      reshape_output_dims[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, fc_output_size) << "Invalid shape is given.";
  }
  lite::DDim out_dims = lite::DDim({reshape_output_dims[0],
                                    reshape_output_dims[2],
                                    reshape_output_dims[1],
                                    reshape_output_dims[3]});
  param_.output->Resize(out_dims);

  // share LoD
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool FusedAttentionOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                      lite::Scope *scope) {
  auto input = op_desc.Input("Input").front();
  auto residual = op_desc.Input("Residual").front();
  auto fc_w = op_desc.Input("W").front();
  auto output = op_desc.Output("Out").front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.residual = scope->FindVar(residual)->GetMutable<lite::Tensor>();
  param_.fc_w = scope->FindVar(fc_w)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(output)->GetMutable<lite::Tensor>();

  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    auto bias_arguments = op_desc.Input("Bias");
    if (bias_arguments.size() > 0) {
      auto bias_var = scope->FindVar(bias_arguments.front());
      if (bias_var != nullptr) {
        param_.fc_bias = bias_var->GetMutable<lite::Tensor>();
      }
    }
  }
  param_.in_num_col_dims = op_desc.GetAttr<int>("in_num_col_dims");
  param_.reshape_shape = op_desc.GetAttr<std::vector<int>>("reshape_shape");
  param_.softmax_axis = op_desc.GetAttr<int>("softmax_axis");

  if (op_desc.HasAttr("activation_type")) {
    param_.activation_type = op_desc.GetAttr<std::string>("activation_type");
  }
  if (param_.activation_type == "relu6") {
    param_.alpha = op_desc.GetAttr<float>("alpha");
  }

  // For Int8
  const OpInfo *op_info = static_cast<const OpInfo *>(&op_desc);
  if (op_info != nullptr && op_info->HasAttr("enable_int8")) {
    param_.calib0_scale = op_desc.GetAttr<std::vector<float>>("calib0_scale");
    param_.calib1_scale = op_desc.GetAttr<std::vector<float>>("calib1_scale");
    param_.fc0_scale = op_desc.GetAttr<std::vector<float>>("fc0_scale");
    param_.fc1_scale = op_desc.GetAttr<std::vector<float>>("fc1_scale");
    param_.fc2_scale = op_desc.GetAttr<std::vector<float>>("fc2_scale");
    param_.enable_int8 = op_info->GetAttr<bool>("enable_int8");
  } else {
    param_.scale = op_info->GetAttr<float>("scale");
  }
  if (op_desc.HasAttr("op_type")) {
    param_.op_type = op_desc.GetAttr<std::string>("op_type");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fused_attention,
                 paddle::lite::operators::FusedAttentionOpLite);
