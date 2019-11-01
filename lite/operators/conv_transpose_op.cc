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

#include "lite/operators/conv_transpose_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConvTransposeOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.filter);
  CHECK_OR_FALSE(param_.output);

  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();

  CHECK_OR_FALSE(in_dims.size() == 4 || in_dims.size() == 5);

  CHECK_EQ_OR_FALSE(in_dims.size(), filter_dims.size());
  CHECK_OR_FALSE(in_dims.size() - param_.strides.size() == 2U);
  CHECK_EQ_OR_FALSE(param_.paddings.size(), param_.strides.size());

  CHECK_OR_FALSE(in_dims[1] % param_.groups == 0);
  return true;
}

bool ConvTransposeOpLite::InferShape() const {
  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();

  std::vector<int64_t> output_shape;
  output_shape.push_back(in_dims[0]);
  output_shape.push_back(filter_dims[1] * param_.groups);
  for (int i = 0; i < param_.strides.size(); i++) {
    int kernel_extent = param_.dilations[i] * (filter_dims[i + 2] - 1) + 1;
    int output_len = (in_dims[i + 2] - 1) * param_.strides[i] + kernel_extent -
                     2 * param_.paddings[i];
    output_shape.push_back(output_len);
  }

  // Set output dims
  param_.output->Resize(lite::DDim(output_shape));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool ConvTransposeOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                     lite::Scope *scope) {
  auto X = op_desc.Input("Input").front();
  auto Filter = op_desc.Input("Filter").front();
  auto Out = op_desc.Output("Output").front();
  param_.x = scope->FindVar(X)->GetMutable<lite::Tensor>();
  param_.filter = scope->FindVar(Filter)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(Out)->GetMutable<lite::Tensor>();

  param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  param_.groups = op_desc.GetAttr<int>("groups");
  param_.dilations = op_desc.GetAttr<std::vector<int>>("dilations");

  // optional params
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    auto bias_arguments = op_desc.Input("Bias");
    if (bias_arguments.size() > 0) {
      auto bias_var = scope->FindVar(bias_arguments.front());
      if (bias_var != nullptr) {
        param_.bias =
            const_cast<lite::Tensor *>(&(bias_var->Get<lite::Tensor>()));
      }
    }
  }
  if (op_desc.HasAttr("fuse_relu")) {
    param_.fuse_relu = op_desc.GetAttr<bool>("fuse_relu");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(conv2d_transpose,
                 paddle::lite::operators::ConvTransposeOpLite);
