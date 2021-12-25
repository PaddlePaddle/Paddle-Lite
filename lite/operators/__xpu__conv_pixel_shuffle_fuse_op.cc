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

#include "lite/operators/__xpu__conv_pixel_shuffle_fuse_op.h"
#include <memory>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace operators {

std::string padding_algorithm_0_ = "";  // NOLINT
std::string padding_algorithm_1_ = "";  // NOLINT

bool XPUConvPixelShuffleOp::CheckShape() const {
  CHECK(param_.input) << "Input(Input) of ConvXPUOp should not be null.";
  CHECK(param_.output) << "Input(Filter) of ConvXPUOp should not be null.";
  CHECK(param_.filter_0) << "Output(Output) of ConvXPUOp should not be null.";
  CHECK(param_.filter_1) << "Output(Output) of ConvXPUOp should not be null.";

  const auto in_dims = param_.input->dims();
  const auto filter_0_dims = param_.filter_0->dims();
  const auto filter_1_dims = param_.filter_1->dims();
  const auto upscale_factor = param_.upscale_factor;
  int groups_0 = param_.groups_0.front();
  int groups_1 = param_.groups_1.front();

  CHECK_EQ(in_dims.size(), 4UL) << "Conv intput should be 4-D tensor.";
  CHECK_EQ(filter_0_dims.size(), 4UL) << "Conv filter should be 4-D tensor.";
  CHECK_EQ(filter_1_dims.size(), 4UL) << "Conv filter should be 4-D tensor.";
  CHECK_EQ(in_dims[1], filter_0_dims[1] * groups_0)
      << "The number of input channels should be equal to filter channels * "
         "groups.";
  CHECK_EQ(filter_0_dims[0] % groups_0, 0)
      << "The number of output channels should be divided by groups.";
  CHECK_EQ(filter_1_dims[0] % groups_1, 0)
      << "The number of output channels should be divided by groups.";
  CHECK_EQ_OR_FALSE(filter_0_dims[0] % (upscale_factor * upscale_factor), 0);
  CHECK_EQ(filter_0_dims[0] / upscale_factor / upscale_factor,
           filter_1_dims[1] * groups_1)
      << "The number of input channels should be equal to filter channels * "
         "groups.";

  return true;
}

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

bool XPUConvPixelShuffleOp::InferShapeImpl() const {
  const auto in_dims = param_.input->dims();
  const auto filter_0_dims = param_.filter_0->dims();

  operators::UpdatePaddingAndDilation(param_.paddings_0.get(),
                                      param_.dilations_0.get(),
                                      param_.strides_0,
                                      padding_algorithm_0_,
                                      in_dims,
                                      filter_0_dims);

  // get mid shape
  auto paddings_0 = *param_.paddings_0;
  auto dilations_0 = *param_.dilations_0;
  std::vector<int64_t> mid_shape({in_dims[0], filter_0_dims[0]});
  for (size_t i = 0; i < param_.strides_0.size(); ++i) {
    mid_shape.push_back(ConvOutputSize(in_dims[i + 2],
                                       filter_0_dims[i + 2],
                                       dilations_0[i],
                                       paddings_0[i * 2],
                                       paddings_0[i * 2 + 1],
                                       param_.strides_0[i]));
  }
  const auto upscale_factor = param_.upscale_factor;
  mid_shape = {mid_shape[0],
               mid_shape[1] / upscale_factor / upscale_factor,
               mid_shape[2] * upscale_factor,
               mid_shape[3] * upscale_factor};
  const auto filter_1_dims = param_.filter_1->dims();

  operators::UpdatePaddingAndDilation(param_.paddings_1.get(),
                                      param_.dilations_1.get(),
                                      param_.strides_1,
                                      padding_algorithm_1_,
                                      lite::DDim(mid_shape),
                                      filter_1_dims);
  std::vector<int64_t> output_shape({mid_shape[0], filter_1_dims[0]});
  auto paddings_1 = *param_.paddings_1;
  auto dilations_1 = *param_.dilations_1;
  for (size_t i = 0; i < param_.strides_1.size(); ++i) {
    output_shape.push_back(ConvOutputSize(mid_shape[i + 2],
                                          filter_1_dims[i + 2],
                                          dilations_1[i],
                                          paddings_1[i * 2],
                                          paddings_1[i * 2 + 1],
                                          param_.strides_1[i]));
  }
  // Set output and output max dims
  param_.output->Resize(lite::DDim(output_shape));
  param_.output_max->Resize({4});
  // share LoD
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool XPUConvPixelShuffleOp::AttachImpl(const cpp::OpDesc& op_desc,
                                       lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Input("Filter_0").front()));
  CHECK(scope->FindVar(op_desc.Input("Filter_1").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));
  CHECK(scope->FindVar(op_desc.Output("OutputMax").front()));

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.filter_0 =
      scope->FindVar(op_desc.Input("Filter_0").front())->GetMutable<Tensor>();
  param_.filter_1 =
      scope->FindVar(op_desc.Input("Filter_1").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  param_.output_max =
      scope->FindVar(op_desc.Output("OutputMax").front())->GetMutable<Tensor>();

  param_.strides_0 = op_desc.GetAttr<std::vector<int>>("strides_0");
  CHECK_EQ(param_.strides_0.size(), 2UL);
  param_.strides_1 = op_desc.GetAttr<std::vector<int>>("strides_1");
  CHECK_EQ(param_.strides_1.size(), 2UL);
  std::vector<int> paddings_0 = op_desc.GetAttr<std::vector<int>>("paddings_0");
  std::vector<int> paddings_1 = op_desc.GetAttr<std::vector<int>>("paddings_1");
  auto dilations_0 = op_desc.GetAttr<std::vector<int>>("dilations_0");
  CHECK_EQ(dilations_0.size(), 2UL);
  param_.dilations_0 = std::make_shared<std::vector<int>>(dilations_0);
  auto dilations_1 = op_desc.GetAttr<std::vector<int>>("dilations_1");
  CHECK_EQ(dilations_1.size(), 2UL);
  param_.dilations_1 = std::make_shared<std::vector<int>>(dilations_1);
  param_.groups_0 = op_desc.GetAttr<std::vector<int>>("groups_0");
  CHECK_EQ(param_.groups_0.size(), 1UL);
  param_.groups_1 = op_desc.GetAttr<std::vector<int>>("groups_1");
  CHECK_EQ(param_.groups_1.size(), 1UL);
  param_.act_type_0 = op_desc.GetAttr<std::vector<int>>("act_type_0");
  CHECK_EQ(param_.act_type_0.size(), 1UL);
  param_.act_type_1 = op_desc.GetAttr<std::vector<int>>("act_type_1");
  CHECK_EQ(param_.act_type_1.size(), 1UL);
  param_.act_param_0 = op_desc.GetAttr<std::vector<float>>("act_param_0");
  CHECK_EQ(param_.act_param_0.size(), 1UL);
  param_.act_param_1 = op_desc.GetAttr<std::vector<float>>("act_param_1");
  CHECK_EQ(param_.act_param_1.size(), 1UL);
  param_.has_bias_0 = op_desc.GetAttr<bool>("has_bias_0");
  param_.has_bias_1 = op_desc.GetAttr<bool>("has_bias_1");
  param_.upscale_factor = op_desc.GetAttr<int>("upscale_factor");

  // optional params
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias_0") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("Bias_0");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.bias_0 =
            const_cast<lite::Tensor*>(&(arg_var->Get<lite::Tensor>()));
      }
    }
  }
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias_1") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("Bias_1");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.bias_1 =
            const_cast<lite::Tensor*>(&(arg_var->Get<lite::Tensor>()));
      }
    }
  }
  if (op_desc.HasAttr("padding_algorithm_0")) {
    padding_algorithm_0_ = op_desc.GetAttr<std::string>("padding_algorithm_0");
  }
  if (op_desc.HasAttr("padding_algorithm_1")) {
    padding_algorithm_1_ = op_desc.GetAttr<std::string>("padding_algorithm_1");
  }

  // 2-pad to 4-pad
  if (paddings_0.size() == 2L) {
    for (size_t i = 0; i < param_.strides_0.size(); ++i) {
      int copy_pad = *(paddings_0.begin() + 2 * i);
      paddings_0.insert(paddings_0.begin() + 2 * i + 1, copy_pad);
    }
  } else {
    if (paddings_0.size() != 4L) {
      LOG(FATAL)
          << "Paddings size should be the same or twice as the input size.";
    }
  }
  param_.paddings_0 = std::make_shared<std::vector<int>>(paddings_0);

  if (paddings_1.size() == 2L) {
    for (size_t i = 0; i < param_.strides_1.size(); ++i) {
      int copy_pad = *(paddings_1.begin() + 2 * i);
      paddings_1.insert(paddings_1.begin() + 2 * i + 1, copy_pad);
    }
  } else {
    if (paddings_1.size() != 4L) {
      LOG(FATAL)
          << "Paddings size should be the same or twice as the input size.";
    }
  }
  param_.paddings_1 = std::make_shared<std::vector<int>>(paddings_1);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__conv_pixel_shuffle_fuse_op,
                 paddle::lite::operators::XPUConvPixelShuffleOp);
