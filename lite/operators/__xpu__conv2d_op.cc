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
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace operators {

std::string padding_algorithm_ = "";

bool XPUConv2dOp::CheckShape() const {
  CHECK(param_.Input) << "Input(Input) of ConvXPUOp should not be null.";
  CHECK(param_.Output) << "Input(Filter) of ConvXPUOp should not be null.";
  CHECK(param_.Filter) << "Output(Output) of ConvXPUOp should not be null.";
  // bias is optional.

  const auto in_dims = param_.Input->dims();
  const auto filter_dims = param_.Filter->dims();
  int groups = param_.groups;

  CHECK_EQ(in_dims.size(), 4UL) << "Conv intput should be 4-D tensor.";
  CHECK_EQ(in_dims.size(), filter_dims.size()) << "Conv input dimension and filter dimension should be the same.";
  CHECK_EQ(in_dims.size() - param_.strides.size(), 2U) << "Conv input dimension and strides dimension should be consistent.";
  CHECK_EQ(filter_dims.size(), 4UL) << "Conv filter should be 4-D tensor.";
  //CHECK_EQ(param_.paddings.size(), param_.strides.size()) << "Conv paddings dimension and Conv strides dimension should be the same.";
  CHECK_EQ(in_dims[1], filter_dims[1] * groups) << "The number of input channels should be equal to filter channels * groups.";
  CHECK_EQ(filter_dims[0] % groups, 0) << "The number of output channels should be divided by groups.";

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
//
//// copy from conv_op.cc
//void UpdatePaddingAndDilation(std::vector<int>* paddings,
//                              std::vector<int>* dilations,
//                              const std::vector<int>& strides,
//                              const std::string padding_algorithm,
//                              const lite::DDim data_dims,
//                              const lite::DDim& ksize) {
//  // when padding_desc is "VALID" or "SAME"
//  if (padding_algorithm == "SAME") {
//    for (size_t i = 0; i < strides.size(); ++i) {
//      int out_size = (data_dims[i + 2] + strides[i] - 1) / strides[i];
//      int pad_sum = std::max(
//          (out_size - 1) * strides[i] + ksize[i + 2] - data_dims[i + 2],
//          (int64_t)0);
//      int pad_0 = pad_sum / 2;
//      int pad_1 = pad_sum - pad_0;
//      // pad
//      *(paddings->begin() + i * 2) = pad_0;
//      *(paddings->begin() + i * 2 + 1) = pad_1;
//      // dilation
//      *(dilations->begin() + i) = 1;
//    }
//  } else if (padding_algorithm == "VALID") {
//    for (auto& it : *paddings) {
//      it = 0;
//    }
//  }
//}

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

  // Set output dims
  param_.Output->Resize(lite::DDim(output_shape));
  // share LoD
  param_.Output->set_lod(param_.Input->lod());

  return true;
}

bool XPUConv2dOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  AttachParam(&param_);
  param_.Input = scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.Filter = scope->FindVar(op_desc.Input("Filter").front())->GetMutable<Tensor>();
  param_.FilterMax = scope->FindVar(op_desc.Input("FilterMax").front())->GetMutable<Tensor>();
  auto bias = scope->FindVar(op_desc.Input("Bias").front());
  if (bias != nullptr) {
    param_.Bias = bias->GetMutable<Tensor>();
  }
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

  param_.Output = scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();

  // optional params
  std::vector<std::string> output_arg_names = op_desc.OutputArgumentNames();
  if (std::find(output_arg_names.begin(), output_arg_names.end(), "OutputMax") !=
        output_arg_names.end()) {
    auto arguments = op_desc.Output("OutputMax");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.OutputMax = arg_var->GetMutable<lite::Tensor>();
      }
    }
  }

  param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  param_.dilations = std::make_shared<std::vector<int>>(dilations);
  param_.groups = op_desc.GetAttr<int>("groups");
  if (op_desc.HasAttr("act_type") && op_desc.GetAttr<bool>("act_type")) {
    param_.act_type = op_desc.GetAttr<int>("act_type");
  } else {
    param_.act_type = xdnn::Activation_t::RELU;
  }
  if (op_desc.HasAttr("filter_type") && op_desc.GetAttr<bool>("filter_type")) {
    param_.filter_type = op_desc.GetAttr<std::string>("filter_type");
  } else {
    param_.filter_type = "int16";
  }

  if (op_desc.HasAttr("has_input_max") && op_desc.GetAttr<bool>("has_input_max")) {
    param_.InputMax = scope->FindVar(op_desc.Input("InputMax").front())->GetMutable<Tensor>();
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
