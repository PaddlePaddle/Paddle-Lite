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
  if (param_.in_num_col_dims == -1) {
    param_.in_num_col_dims += input_dims.size();
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
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Input("Filter").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));
  CHECK(scope->FindVar(op_desc.Output("OutputMax").front()));

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.w =
      scope->FindVar(op_desc.Input("Filter").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  param_.output_max =
      scope->FindVar(op_desc.Output("OutputMax").front())->GetMutable<Tensor>();

  param_.act_type = op_desc.GetAttr<int>("act_type");
  param_.act_param = op_desc.GetAttr<float>("act_param");
  param_.has_bias = op_desc.GetAttr<bool>("has_bias");
  param_.in_num_col_dims = op_desc.GetAttr<int>("in_num_col_dims");
  param_.transpose_x = op_desc.GetAttr<bool>("transpose_x");
  param_.transpose_w = op_desc.GetAttr<bool>("transpose_w");
  // optional params
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

  if (op_desc.HasAttr("has_input_max") &&
      op_desc.GetAttr<bool>("has_input_max")) {
    CHECK(scope->FindVar(op_desc.Input("InputMax").front()));
    param_.input_max =
        scope->FindVar(op_desc.Input("InputMax").front())->GetMutable<Tensor>();
  }

  if (op_desc.HasAttr("enable_int8") && op_desc.GetAttr<bool>("enable_int8")) {
    param_.enable_int8 = op_desc.GetAttr<bool>("enable_int8");
    // Equivalent use Input0_scale,Filter0_scale
    param_.quant_input_max =
        op_desc.GetAttr<std::vector<float>>("Input0_scale")[0];
    param_.weight_max = op_desc.GetAttr<std::vector<float>>("Filter0_scale");
    param_.quant_output_max =
        op_desc.GetAttr<std::vector<float>>("Output0_scale")[0];
    param_.per_channel = op_desc.GetAttr<bool>("per_channel");
  }

  if (op_desc.HasAttr("enable_int16") &&
      op_desc.GetAttr<bool>("enable_int16")) {
    param_.enable_int16 = true;
    param_.quant_input_max =
        op_desc.GetAttr<std::vector<float>>("Input0_scale")[0];
    param_.weight_max = op_desc.GetAttr<std::vector<float>>("Filter0_scale");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__fc, paddle::lite::operators::XPUFcOp);
