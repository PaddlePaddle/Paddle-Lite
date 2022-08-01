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

#include "lite/operators/fc_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FcOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.w);
  // bias is optional.

  const auto input_dims = param_.input->dims();
  const auto w_dims = param_.w->dims();
  CHECK_EQ_OR_FALSE(w_dims.size(), 2UL);

  int64_t w_dims_1 = param_.padding_weights ? w_dims[1] - 4 : w_dims[1];
  if (param_.bias) {
    const auto bias_dims = param_.bias->dims();
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
  param_.in_mat_dims = input_dims.Flatten2D(param_.in_num_col_dims);
  // CHECK_EQ_OR_FALSE(param_.in_mat_dims[1], w_dims[0]);

  return true;
}

bool FcOpLite::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  int64_t w_dims_1;
  if (param_.w_dims.empty()) {
    const auto& w_dims = param_.w->dims();
    w_dims_1 = param_.padding_weights ? w_dims[1] - 4 : w_dims[1];
  } else {
    const auto& w_dims = param_.w_dims;
    w_dims_1 = param_.padding_weights ? w_dims[1] - 4 : w_dims[1];
  }
  int in_num_col_dims = param_.in_num_col_dims;
  std::string op_type = param_.op_type;
  if (op_type == "matmul" || op_type == "matmul_v2") {
    in_num_col_dims = input_dims.size() - 1;
  }
  param_.in_num_col_dims = in_num_col_dims;

  // Set output dims
  std::vector<DDim::value_type> output_dims(in_num_col_dims + 1);
  for (int i = 0; i < in_num_col_dims; ++i) {
    output_dims[i] = input_dims[i];
  }
  output_dims[in_num_col_dims] = w_dims_1;
  param_.output->Resize(output_dims);

  // share LoD
  param_.output->set_lod(param_.input->lod());

  return true;
}

bool FcOpLite::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto input = op_desc.Input("Input").front();
  auto W = op_desc.Input("W").front();
  auto out = op_desc.Output("Out").front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.w = scope->FindVar(W)->GetMutable<lite::Tensor>();
  param_.w_dims = param_.w->dims();
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    auto bias_arguments = op_desc.Input("Bias");
    if (bias_arguments.size() > 0) {
      auto bias_var = scope->FindVar(bias_arguments.front());
      if (bias_var != nullptr) {
        param_.bias = bias_var->GetMutable<lite::Tensor>();
      }
    }
  }
  CHECK(scope->FindVar(out));
  param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.in_num_col_dims = op_desc.GetAttr<int>("in_num_col_dims");
  input_tensor_ptrs_cache_.push_back(param_.input);
  output_tensor_ptrs_cache_.push_back(param_.output);

  if (op_desc.HasAttr("activation_type")) {
    param_.activation_type = op_desc.GetAttr<std::string>("activation_type");
  }
  if (op_desc.HasAttr("padding_weights")) {
    param_.padding_weights = op_desc.GetAttr<bool>("padding_weights");
  } else {
    param_.padding_weights = false;
  }

  if (param_.activation_type == "prelu") {
    param_.Prelu_mode = op_desc.GetAttr<std::string>("prelu_mode");
    auto prelu_alpha_name = op_desc.Input("Alpha").front();
    auto prelu_alpha_var = scope->FindVar(prelu_alpha_name);
    param_.Prelu_alpha =
        const_cast<lite::Tensor*>(&(prelu_alpha_var->Get<lite::Tensor>()));
  }
  if (param_.activation_type == "relu6") {
    param_.alpha = op_desc.GetAttr<float>("alpha");
  }

  // For Int8
  const OpInfo* op_info = static_cast<const OpInfo*>(&op_desc);
  if (op_info != nullptr && op_info->HasAttr("enable_int8")) {
    param_.enable_int8 = op_info->GetAttr<bool>("enable_int8");
    auto input_scale_name = "Input0_scale";
    auto weight_scale_name = "W0_scale";
    auto out_scale_name = "Out0_scale";
    if (op_info->HasInputScale(input_scale_name, true))
      param_.input_scale = op_info->GetInputScale(input_scale_name, true)[0];
    if (op_info->HasInputScale(weight_scale_name, true))
      param_.weight_scale = op_info->GetInputScale(weight_scale_name, true);
    if (op_info->HasOutputScale(out_scale_name, true))
      param_.output_scale = op_info->GetOutputScale(out_scale_name, true)[0];
  }
  if (op_desc.HasAttr("op_type")) {
    param_.op_type = op_desc.GetAttr<std::string>("op_type");
  }

#ifdef LITE_WITH_FPGA
  if (op_info != nullptr && op_info->HasAttr("fpga_static_quant")) {
    param_.enable_int8 = op_info->GetAttr<bool>("fpga_static_quant");
    auto input_scale_name = "Input0_scale";
    if (op_info->HasInputScale(input_scale_name, true)) {
      param_.input_scale = op_info->GetInputScale(input_scale_name, true)[0];
    }
  }
#endif

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fc, paddle::lite::operators::FcOpLite);
