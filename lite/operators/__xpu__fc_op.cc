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

#include "lite/operators/__xpu__fc_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUFcOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.w);
  // bias is optional.

  const auto input_dims = param_.input->dims();
  const auto w_dims = param_.w->dims();
  CHECK_EQ_OR_FALSE(w_dims.size(), 2UL);

  int64_t w_dims_1 = w_dims[1];
  if (param_.bias) {
    const auto bias_dims = param_.bias->dims();
    if (bias_dims.size() == 2) {
      CHECK_EQ_OR_FALSE(bias_dims[0], 1);
      CHECK_EQ_OR_FALSE(bias_dims[1], w_dims_1);
    } else if (bias_dims.size() == 1) {
      CHECK_EQ_OR_FALSE(bias_dims[0], w_dims_1);
    }
  }

  CHECK_GT_OR_FALSE(input_dims.size(),
                    static_cast<size_t>(param_.in_num_col_dims));
  param_.in_mat_dims = input_dims.Flatten2D(param_.in_num_col_dims);
  CHECK_EQ_OR_FALSE(param_.in_mat_dims[1], w_dims[0]);

  return true;
}

bool XPUFcOp::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  const auto& w_dims = param_.w->dims();
  int in_num_col_dims = param_.in_num_col_dims;
  int64_t w_dims_1 = w_dims[1];

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

bool XPUFcOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto input = op_desc.Input("Input").front();
  auto W = op_desc.Input("W").front();
  auto out = op_desc.Output("Out").front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.w = scope->FindVar(W)->GetMutable<lite::Tensor>();
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
  param_.w_max = op_desc.GetAttr<float>("w_max");

  if (op_desc.HasAttr("activation_type")) {
    param_.activation_type = op_desc.GetAttr<std::string>("activation_type");
  }
  if (op_desc.HasAttr("transpose_w")) {
    param_.transpose_w = op_desc.GetAttr<bool>("transpose_w");
  }
  if (op_desc.HasAttr("precision")) {
    param_.precision = op_desc.GetAttr<std::string>("precision");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__fc, paddle::lite::operators::XPUFcOp);
