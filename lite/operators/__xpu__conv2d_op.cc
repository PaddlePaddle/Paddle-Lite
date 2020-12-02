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

#include "lite/operators/__xpu__conv2d_op.h"
#include <memory>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace operators {

std::string padding_algorithm_ = "";  // NOLINT

bool XPUConv2dOp::CheckShape() const {
  CHECK(param_.Input) << "Input(Input) of ConvXPUOp should not be null.";
  CHECK(param_.Output) << "Input(Filter) of ConvXPUOp should not be null.";
  CHECK(param_.Filter) << "Output(Output) of ConvXPUOp should not be null.";
  // bias is optional.

  const auto in_dims = param_.Input->dims();
  const auto filter_dims = param_.Filter->dims();
  int groups = param_.groups;

  CHECK_EQ(in_dims.size(), 4UL) << "Conv intput should be 4-D tensor.";
  CHECK_EQ(in_dims.size(), filter_dims.size())
      << "Conv input dimension and filter dimension should be the same.";
  CHECK_EQ(in_dims.size() - param_.strides.size(), 2U)
      << "Conv input dimension and strides dimension should be consistent.";
  CHECK_EQ(filter_dims.size(), 4UL) << "Conv filter should be 4-D tensor.";
  CHECK_EQ(in_dims[1], filter_dims[1] * groups)
      << "The number of input channels should be equal to filter channels * "
         "groups.";
  CHECK_EQ(filter_dims[0] % groups, 0)
      << "The number of output channels should be divided by groups.";

  return true;
}

// copy from conv_op.cc
inline int ConvOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int pad_left,
                          int pad_right,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
}

// copy from conv_op.cc
bool XPUConv2dOp::InferShapeImpl() const {
  const auto in_dims = param_.Input->dims();
  const auto filter_dims = param_.Filter->dims();

  operators::UpdatePaddingAndDilation(param_.paddings.get(),
                                      param_.dilations.get(),
                                      param_.strides,
                                      padding_algorithm_,
                                      in_dims,
                                      filter_dims);
  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  auto paddings = *param_.paddings;
  auto dilations = *param_.dilations;
  for (size_t i = 0; i < param_.strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2],
                                          filter_dims[i + 2],
                                          dilations[i],
                                          paddings[i * 2],
                                          paddings[i * 2 + 1],
                                          param_.strides[i]));
  }

  // Set output and output max dims
  param_.Output->Resize(lite::DDim(output_shape));
  param_.OutputMax->Resize({4});
  // share LoD
  param_.Output->set_lod(param_.Input->lod());

  return true;
}

bool XPUConv2dOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  AttachParam(&param_);
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Input("Filter").front()));
  CHECK(scope->FindVar(op_desc.Input("FilterMax").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));
  CHECK(scope->FindVar(op_desc.Output("OutputMax").front()));

  param_.Input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.Filter =
      scope->FindVar(op_desc.Input("Filter").front())->GetMutable<Tensor>();
  param_.FilterMax =
      scope->FindVar(op_desc.Input("FilterMax").front())->GetMutable<Tensor>();
  param_.Output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  param_.OutputMax =
      scope->FindVar(op_desc.Output("OutputMax").front())->GetMutable<Tensor>();

  param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  param_.dilations = std::make_shared<std::vector<int>>(dilations);
  param_.groups = op_desc.GetAttr<int>("groups");
  param_.act_type = op_desc.GetAttr<int>("act_type");
  param_.act_param = op_desc.GetAttr<float>("act_param");
  param_.filter_dims = op_desc.GetAttr<std::vector<int>>("filter_dims");

  // optional params
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Branch") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("Branch");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.Branch =
            const_cast<lite::Tensor*>(&(arg_var->Get<lite::Tensor>()));
      }
    }
  }
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("Bias");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.Bias =
            const_cast<lite::Tensor*>(&(arg_var->Get<lite::Tensor>()));
      }
    }
  }

  if (op_desc.HasAttr("has_input_max") &&
      op_desc.GetAttr<bool>("has_input_max")) {
    CHECK(scope->FindVar(op_desc.Input("InputMax").front()));
    param_.InputMax =
        scope->FindVar(op_desc.Input("InputMax").front())->GetMutable<Tensor>();
  }

  if (op_desc.HasAttr("padding_algorithm")) {
    padding_algorithm_ = op_desc.GetAttr<std::string>("padding_algorithm");
  }

  // 2-pad to 4-pad
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < param_.strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  } else {
    if (paddings.size() != 4L) {
      LOG(FATAL)
          << "Paddings size should be the same or twice as the input size.";
    }
  }
  param_.paddings = std::make_shared<std::vector<int>>(paddings);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__conv2d, paddle::lite::operators::XPUConv2dOp);
