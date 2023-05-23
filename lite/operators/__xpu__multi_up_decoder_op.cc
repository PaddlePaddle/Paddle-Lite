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

#include "lite/operators/__xpu__multi_up_decoder_op.h"
#include <utility>
#include <vector>
#include "lite/backends/xpu/vec_trans.h"
#include "lite/core/op_registry.h"

#define SEPERATE_NUM -1000
#define RESBLOCK_CONV_NUM 2
#define RESBLOCK_GN_NUM 2

namespace paddle {
namespace lite {
namespace operators {

bool XPUMultiUpDecoderOp::CheckShape() const { return true; }

bool XPUMultiUpDecoderOp::InferShapeImpl() const {
  auto input_shape = param_.input1->dims();
  auto batch_size = input_shape[0];
  int last_up_decoder_idx = param_.num_up_decoders - 1;
  int last_resblock_idx =
      param_.num_resblocks_per_up_decoder[last_up_decoder_idx] - 1;
  int last_conv_idx =
      param_.all_resblock_filter_dims[last_up_decoder_idx][last_resblock_idx]
          .size() -
      1;
  auto channel_out =
      param_.all_resblock_filter_dims[last_up_decoder_idx][last_resblock_idx]
                                     [last_conv_idx][0];
  auto h = input_shape[2];
  auto w = input_shape[3];

  for (int i = 0; i < param_.has_post_interp.size(); ++i) {
    if (param_.has_post_interp[i]) {
      if (param_.post_interp_scale[i].size() > 0) {
        h *= static_cast<int>(param_.post_interp_scale[i][0]);
        w *= static_cast<int>(param_.post_interp_scale[i][1]);
      } else {
        h = param_.post_interp_out_dhw[i][1];
        w = param_.post_interp_out_dhw[i][2];
      }
    }
  }
  param_.output->Resize({batch_size, channel_out, h, w});
  return true;
}

bool XPUMultiUpDecoderOp::AttachImpl(const cpp::OpDesc& op_desc,
                                     lite::Scope* scope) {
  // All attributes
  param_.num_resblocks_per_up_decoder =
      op_desc.GetAttr<std::vector<int>>("ResblockNumPerUpDecoder");
  param_.num_up_decoders = param_.num_resblocks_per_up_decoder.size();
  int tmp_idx = 0;
  param_.resblock_start_idx.clear();
  param_.resblock_end_idx.clear();
  for (int i = 0; i < param_.num_resblocks_per_up_decoder.size(); ++i) {
    param_.resblock_start_idx.push_back(tmp_idx);
    tmp_idx += param_.num_resblocks_per_up_decoder[i];
    param_.resblock_end_idx.push_back(tmp_idx - 1);
  }
  std::vector<int> res_conv_fix_info =
      op_desc.GetAttr<std::vector<int>>("ResConvFixInfo");
  param_.all_resblock_conv_fix = paddle::lite::xpu::vec::Vec1DTo2DWithIdx(
      res_conv_fix_info, param_.resblock_start_idx, param_.resblock_end_idx);

  tmp_idx = 0;
  // Resblock related attributes
  std::vector<int> res_conv_groups =
      op_desc.GetAttr<std::vector<int>>("ResConvGroups");
  param_.all_resblock_conv_groups =
      paddle::lite::xpu::vec::Vec1DTo3DWithExtraInfo(
          res_conv_groups,
          param_.all_resblock_conv_fix,
          RESBLOCK_CONV_NUM,
          true);
  std::vector<float> res_gn_eps =
      op_desc.GetAttr<std::vector<float>>("ResblockGNEps");
  param_.all_resblock_gn_eps = paddle::lite::xpu::vec::Vec1DTo3DWithExtraInfo(
      res_gn_eps, param_.all_resblock_conv_fix, RESBLOCK_GN_NUM, false);
  std::vector<int> res_gn_groups =
      op_desc.GetAttr<std::vector<int>>("ResblockGNGroups");
  param_.all_resblock_gn_groups =
      paddle::lite::xpu::vec::Vec1DTo3DWithExtraInfo(
          res_gn_groups, param_.all_resblock_conv_fix, RESBLOCK_GN_NUM, false);
  std::vector<int> res_conv_dilations =
      op_desc.GetAttr<std::vector<int>>("ResConvDilations");
  param_.all_resblock_conv_dilations =
      paddle::lite::xpu::vec::Vec1DTo4DWithExtraInfo(
          res_conv_dilations,
          param_.all_resblock_conv_fix,
          RESBLOCK_CONV_NUM,
          2);
  std::vector<int> res_conv_filter_dims =
      op_desc.GetAttr<std::vector<int>>("ResConvFilterDims");
  param_.all_resblock_filter_dims =
      paddle::lite::xpu::vec::Vec1DTo4DWithExtraInfo(
          res_conv_filter_dims,
          param_.all_resblock_conv_fix,
          RESBLOCK_CONV_NUM,
          4);
  std::vector<int> res_conv_paddings =
      op_desc.GetAttr<std::vector<int>>("ResConvPaddings");
  param_.all_resblock_conv_paddings =
      paddle::lite::xpu::vec::Vec1DTo4DWithExtraInfo(
          res_conv_paddings,
          param_.all_resblock_conv_fix,
          RESBLOCK_CONV_NUM,
          4);
  std::vector<int> res_conv_strides =
      op_desc.GetAttr<std::vector<int>>("ResConvStrides");
  param_.all_resblock_conv_strides =
      paddle::lite::xpu::vec::Vec1DTo4DWithExtraInfo(
          res_conv_strides, param_.all_resblock_conv_fix, RESBLOCK_CONV_NUM, 2);

  // Post interp+conv related attribute
  param_.has_post_interp = op_desc.GetAttr<std::vector<int>>("HasInterp");
  param_.post_interp_align_corners =
      op_desc.GetAttr<std::vector<int>>("PostInterpAlignCorners");
  param_.post_interp_method =
      op_desc.GetAttr<std::vector<std::string>>("PostInterpMethods");
  std::vector<int> post_interp_dhw =
      op_desc.GetAttr<std::vector<int>>("PostInterpOutDHW");
  param_.post_interp_out_dhw =
      paddle::lite::xpu::vec::Vec1DTo2D(post_interp_dhw, 3);
  std::vector<float> post_interp_scale =
      op_desc.GetAttr<std::vector<float>>("PostInterpOutScale");
  param_.post_interp_scale =
      paddle::lite::xpu::vec::Vec1DTo2D(post_interp_scale, 2);

  std::vector<int> post_conv_strides =
      op_desc.GetAttr<std::vector<int>>("PostConvStrides");
  std::vector<int> post_conv_paddings =
      op_desc.GetAttr<std::vector<int>>("PostConvPaddings");
  std::vector<int> post_conv_filter_dims =
      op_desc.GetAttr<std::vector<int>>("PostConvFilterDims");
  std::vector<int> post_conv_dilations =
      op_desc.GetAttr<std::vector<int>>("PostConvDilations");
  param_.all_post_conv_strides =
      paddle::lite::xpu::vec::Vec1DTo2D(post_conv_strides, 2);
  param_.all_post_conv_paddings =
      paddle::lite::xpu::vec::Vec1DTo2D(post_conv_paddings, 4);
  param_.all_post_filter_dims =
      paddle::lite::xpu::vec::Vec1DTo2D(post_conv_filter_dims, 4);
  param_.all_post_conv_dilations =
      paddle::lite::xpu::vec::Vec1DTo2D(post_conv_dilations, 2);
  param_.all_post_conv_groups =
      op_desc.GetAttr<std::vector<int>>("PostConvGroups");
  // Inputs
  param_.input1 = scope->FindTensor(op_desc.Input("Input").front());

  param_.all_resblock_conv_filter.clear();
  for (auto& name : op_desc.Input("AllUpDecoderResConvFilter")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_resblock_conv_filter.push_back(t);
  }

  param_.all_resblock_conv_bias.clear();
  for (auto& name : op_desc.Input("AllUpDecoderResConvBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_resblock_conv_bias.push_back(t);
  }

  param_.all_resblock_conv_input_max.clear();
  for (auto& name : op_desc.Input("AllUpDecoderInputMax")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_resblock_conv_input_max.push_back(t);
  }

  param_.all_resblock_gn_scale.clear();
  for (auto& name : op_desc.Input("AllUpDecoderGNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_resblock_gn_scale.push_back(t);
  }

  param_.all_resblock_gn_bias.clear();
  for (auto& name : op_desc.Input("AllUpDecoderGNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_resblock_gn_bias.push_back(t);
  }

  param_.all_post_conv_input_max.clear();
  for (auto& name : op_desc.Input("AllUpDecoderPostConvInputMax")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_resblock_conv_input_max.push_back(t);
  }
  param_.all_post_conv_filter.clear();
  for (auto& name : op_desc.Input("AllUpDecoderPostConvFilter")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_post_conv_filter.push_back(t);
  }

  param_.all_post_conv_bias.clear();
  for (auto& name : op_desc.Input("AllUpDecoderPostConvBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.all_post_conv_bias.push_back(t);
  }

  // All resblocks conv filters max
  param_.all_resblock_filter_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("ResConvFilterMaxs")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.all_resblock_filter_max.push_back(tensor);
  }

  // All post conv filters max
  param_.all_post_filter_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("PostConvFilterMaxs")) {
    if (weight_max_tensor != "none") {
      auto tensor = scope->FindMutableTensor(weight_max_tensor);
      CHECK(tensor != nullptr);
      param_.all_post_filter_max.push_back(tensor);
    } else {
      param_.all_post_filter_max.push_back(nullptr);
    }
  }
  // Last GN Scale / GN bias
  param_.has_last_gn_silu = op_desc.GetAttr<bool>("HasLastGNSilu");
  if (param_.has_last_gn_silu) {
    param_.last_gn_bias = scope->FindVar(op_desc.Input("LastGNBias").front())
                              ->GetMutable<Tensor>();
    param_.last_gn_scale = scope->FindVar(op_desc.Input("LastGNScale").front())
                               ->GetMutable<Tensor>();
    param_.last_gn_eps = op_desc.GetAttr<float>("LastGNEps");
    param_.last_gn_groups = op_desc.GetAttr<int>("LastGNGroups");
  }
  // Output
  param_.output = scope->FindMutableTensor(op_desc.Output("Output").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__multi_up_decoder,
                 paddle::lite::operators::XPUMultiUpDecoderOp);
