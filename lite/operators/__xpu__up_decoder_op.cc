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

#include "lite/operators/__xpu__up_decoder_op.h"
#include <utility>
#include <vector>
#include "lite/backends/xpu/vec_trans.h"
#include "lite/core/op_registry.h"

#define SEPERATE_NUM -1000

namespace paddle {
namespace lite {
namespace operators {

bool XPUUpDecoderOp::CheckShape() const { return true; }

bool XPUUpDecoderOp::InferShapeImpl() const {
  auto input_shape = param_.input1->dims();
  auto batch_size = input_shape[0];
  auto channel_out = -1;
  auto h = input_shape[2];
  auto w = input_shape[3];
  if (param_.has_post_interp_conv) {
    channel_out = param_.post_filter_dims[0];
    auto Scale = param_.interp_scale;
    if (Scale.size() > 0) {
      auto scale_h = Scale[0];
      auto scale_w = Scale[1];
      h = static_cast<int>(h * scale_h);
      w = static_cast<int>(w * scale_w);
    } else {
      h = param_.interp_out_h;
      w = param_.interp_out_w;
    }
  } else {
    channel_out = param_.resblock_filter_dims[param_.num_resblocks - 1][0][0];
  }
  param_.output->Resize({batch_size, channel_out, h, w});
  return true;
}

bool XPUUpDecoderOp::AttachImpl(const cpp::OpDesc& op_desc,
                                lite::Scope* scope) {
  // All attributes
  param_.resblock_conv_fix =
      op_desc.GetAttr<std::vector<int>>("ResblockConvFix");
  std::vector<int> res_conv_groups =
      op_desc.GetAttr<std::vector<int>>("ResblockConvGroups");
  param_.resblock_conv_groups =
      paddle::lite::xpu::vec::Vec3DTo2D(paddle::lite::xpu::vec::Vec1DTo3D(
          res_conv_groups, param_.resblock_conv_fix, 1));
  std::vector<int> res_conv_strides =
      op_desc.GetAttr<std::vector<int>>("ResblockConvStrides");
  param_.resblock_conv_strides = paddle::lite::xpu::vec::Vec1DTo3D(
      res_conv_strides, param_.resblock_conv_fix, 2);
  std::vector<int> res_conv_paddings =
      op_desc.GetAttr<std::vector<int>>("ResblockConvPaddings");
  param_.resblock_conv_paddings = paddle::lite::xpu::vec::Vec1DTo3D(
      res_conv_paddings, param_.resblock_conv_fix, 4);
  std::vector<int> res_conv_dilations =
      op_desc.GetAttr<std::vector<int>>("ResblockConvDilations");
  param_.resblock_conv_dilations = paddle::lite::xpu::vec::Vec1DTo3D(
      res_conv_dilations, param_.resblock_conv_fix, 2);
  std::vector<float> res_gn_eps =
      op_desc.GetAttr<std::vector<float>>("ResblockGNEps");
  std::vector<int> res_gn_groups =
      op_desc.GetAttr<std::vector<int>>("ResblockGNGroups");
  std::vector<int> res_conv_filter_dims =
      op_desc.GetAttr<std::vector<int>>("ResblockConvFilterDims");
  param_.resblock_gn_eps = paddle::lite::xpu::vec::Vec1DTo2D(res_gn_eps, 2);
  param_.resblock_gn_groups =
      paddle::lite::xpu::vec::Vec1DTo2D(res_gn_groups, 2);
  param_.resblock_filter_dims = paddle::lite::xpu::vec::Vec1DTo3D(
      res_conv_filter_dims, param_.resblock_conv_fix, 4);
  param_.num_resblocks = op_desc.GetAttr<int>("NumResblocks");
  param_.has_post_interp_conv = op_desc.GetAttr<bool>("PostInterp");
  if (param_.has_post_interp_conv) {
    param_.post_conv_strides =
        op_desc.GetAttr<std::vector<int>>("PostConvStrides");
    param_.post_conv_paddings =
        op_desc.GetAttr<std::vector<int>>("PostConvPaddings");
    param_.post_conv_dilations =
        op_desc.GetAttr<std::vector<int>>("PostConvDilations");
    param_.post_filter_dims =
        op_desc.GetAttr<std::vector<int>>("PostConvFilterDims");
    param_.post_conv_groups =
        op_desc.GetAttr<std::vector<int>>("PostConvGroups");
    param_.interp_align_corners = op_desc.GetAttr<bool>("interp_align_corners");
    param_.interp_method = op_desc.GetAttr<std::string>("interp_method");
    param_.interp_scale = op_desc.GetAttr<std::vector<float>>("interp_scale");
    param_.interp_out_d = op_desc.GetAttr<int>("interp_out_d");
    param_.interp_out_h = op_desc.GetAttr<int>("interp_out_h");
    param_.interp_out_w = op_desc.GetAttr<int>("interp_out_w");
  }
  // Input
  param_.input1 = scope->FindTensor(op_desc.Input("Input").front());
  // All resblocks conv filters
  param_.resblock_conv_filter.clear();
  for (auto& name : op_desc.Input("ResblockConvFilter")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.resblock_conv_filter.push_back(t);
  }
  // All resblocks conv bias
  param_.resblock_conv_bias.clear();
  for (auto& name : op_desc.Input("ResblockConvBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.resblock_conv_bias.push_back(t);
  }
  // All resblocks conv filters max
  param_.resblock_filter_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("ResblockConvFilterMaxs")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.resblock_filter_max.push_back(tensor);
  }
  // All resblocks gn scale
  param_.resblock_gn_scale.clear();
  for (auto& name : op_desc.Input("ResblockGNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.resblock_gn_scale.push_back(t);
  }
  // All resblocks gn bias
  param_.resblock_gn_bias.clear();
  for (auto& name : op_desc.Input("ResblockGNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.resblock_gn_bias.push_back(t);
  }
  // All resblocks conv input max
  param_.resblock_conv_input_max.clear();
  for (auto& name : op_desc.Input("ResblockInputMax")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.resblock_conv_input_max.push_back(t);
  }
  // pos conv filter and bias
  if (param_.has_post_interp_conv) {
    param_.post_conv_filter =
        scope->FindVar(op_desc.Input("PostConvFilter").front())
            ->GetMutable<Tensor>();
    param_.post_conv_bias =
        scope->FindVar(op_desc.Input("PostConvBias").front())
            ->GetMutable<Tensor>();
    // post conv filter max
    param_.post_filter_max.clear();
    for (const auto& weight_max_tensor :
         op_desc.GetAttr<std::vector<std::string>>("PostConvFilterMax")) {
      auto tensor = scope->FindMutableTensor(weight_max_tensor);
      CHECK(tensor != nullptr);
      param_.post_filter_max.push_back(tensor);
    }
  }
  param_.output = scope->FindMutableTensor(op_desc.Output("Output").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__up_decoder, paddle::lite::operators::XPUUpDecoderOp);
