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

#include "lite/kernels/xpu/__xpu__multi_up_decoder_compute.h"
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> PreparePostConvWeight(
    const std::vector<lite::Tensor*>& op_weight) {
  std::vector<const T*> result;
  for (auto* weight : op_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

template <typename T>
static std::vector<std::vector<std::vector<const T*>>> PrepareResConvWeight(
    const std::vector<lite::Tensor*>& op_weight,
    const std::vector<std::vector<int>>& extra_info,
    int conv_offset) {
  int tmp_idx = 0;
  std::vector<std::vector<std::vector<const T*>>> result;
  for (int i = 0; i < extra_info.size(); i++) {
    std::vector<std::vector<const T*>> tmp_1;
    for (int j = 0; j < extra_info[i].size(); j++) {
      std::vector<const T*> tmp_2;
      int tmp_idx_buff = tmp_idx;
      tmp_idx += (conv_offset + extra_info[i][j]);
      for (int k = tmp_idx_buff; k < tmp_idx; k++) {
        tmp_2.push_back(
            reinterpret_cast<const T*>(op_weight[k]->data<float>()));
      }
      tmp_1.push_back(tmp_2);
    }
    result.push_back(tmp_1);
  }
  return result;
}

template <typename InType, PrecisionType PType>
void XPUMultiUpDecoderCompute<InType, PType>::PreparePostConvFilterMax(
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : weight_max) {
    if (max_tensor) {
      max_value_num += max_tensor->numel();
    }
  }
  VLOG(3) << "Total post conv weight max value number: " << max_value_num;
  post_conv_filter_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* weight_max_ptr =
      reinterpret_cast<float*>(post_conv_filter_max_guard_->addr_);

  int offset = 0;
  for (auto max_tensor : weight_max) {
    if (max_tensor) {
      float* cur_weight_max_ptr = weight_max_ptr + offset;
      auto len = max_tensor->numel();
      VLOG(6) << "weight max value: " << max_tensor->data<float>()[0] << " "
              << max_tensor->data<float>()[len - 1];
      std::vector<float> cpu_max(max_ptr_len, max_tensor->data<float>()[0]);
      lite::TargetWrapperXPU::MemcpySync(cur_weight_max_ptr,
                                         cpu_max.data(),
                                         sizeof(float) * max_ptr_len,
                                         IoDirection::HtoD);
      max_xpu_ptrs->push_back(cur_weight_max_ptr);
      offset += max_ptr_len;
    } else {
      max_xpu_ptrs->push_back(nullptr);
    }
  }
}

template <typename InType, PrecisionType PType>
void XPUMultiUpDecoderCompute<InType, PType>::PrepareAllResConvFilterMax(
    const std::vector<lite::Tensor*>& filter_max,
    const std::vector<std::vector<int>>& extra_info,
    int max_ptr_len,
    int conv_offset,
    std::vector<std::vector<std::vector<const float*>>>* max_xpu_ptrs) {
  int max_value_num = 0;
  std::vector<const float*> max_xpu_ptrs_buff;
  for (auto max_tensor : filter_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total resblock conv weight max value number: " << max_value_num;
  resblock_conv_filter_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* filter_max_ptr =
      reinterpret_cast<float*>(resblock_conv_filter_max_guard_->addr_);

  int offset = 0;
  for (auto max_tensor : filter_max) {
    float* cur_filter_max_ptr = filter_max_ptr + offset;
    auto len = max_tensor->numel();
    VLOG(6) << "weight max value: " << max_tensor->data<float>()[0] << " "
            << max_tensor->data<float>()[len - 1];
    std::vector<float> cpu_max(max_ptr_len, max_tensor->data<float>()[0]);
    lite::TargetWrapperXPU::MemcpySync(cur_filter_max_ptr,
                                       cpu_max.data(),
                                       sizeof(float) * max_ptr_len,
                                       IoDirection::HtoD);
    max_xpu_ptrs_buff.push_back(cur_filter_max_ptr);
    offset += max_ptr_len;
  }
  int tmp_idx = 0;
  for (int i = 0; i < extra_info.size(); i++) {
    std::vector<std::vector<const float*>> tmp_1;
    for (int j = 0; j < extra_info[i].size(); j++) {
      std::vector<const float*> tmp_2;
      int tmp_idx_buff = tmp_idx;
      tmp_idx += (conv_offset + extra_info[i][j]);
      for (int k = tmp_idx_buff; k < tmp_idx; k++) {
        tmp_2.push_back(max_xpu_ptrs_buff[k]);
      }
      tmp_1.push_back(tmp_2);
    }
    max_xpu_ptrs->push_back(tmp_1);
  }
}

template <typename InType, PrecisionType PType>
void XPUMultiUpDecoderCompute<InType, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();

  // prepare params
  multi_up_decoder_param_.num_up_decoders = param.num_up_decoders;
  multi_up_decoder_param_.num_resblocks_per_up_decoder =
      param.num_resblocks_per_up_decoder;
  // Upsample params
  for (int i = 0; i < param.has_post_interp.size(); i++) {
    xft::UpSampleParam up_sample_tmp;
    up_sample_tmp.interpolate2d_scale = param.post_interp_scale[i];
    up_sample_tmp.is_nearest = (param.post_interp_method[i] == "nearest");
    up_sample_tmp.is_nchw = true;
    up_sample_tmp.conv_groups = {param.all_post_conv_groups[i]};
    up_sample_tmp.kernel_dims = {param.all_post_filter_dims[i]};
    up_sample_tmp.strides = {param.all_post_conv_strides[i]};
    up_sample_tmp.dilations = {param.all_post_conv_dilations[i]};
    up_sample_tmp.paddings = {param.all_post_conv_paddings[i]};
    multi_up_decoder_param_.upsample_params.push_back(up_sample_tmp);
  }

  // Resblock params
  for (int i = 0; i < param.all_resblock_conv_fix.size(); i++) {
    std::vector<xft::STResBlockParam> resblocks_tmp1;
    for (int j = 0; j < param.all_resblock_conv_fix[i].size(); j++) {
      xft::STResBlockParam resblock_tmp2;
      resblock_tmp2.conv_fix =
          static_cast<bool>(param.all_resblock_conv_fix[i][j]);
      resblock_tmp2.has_silu_fc_input = false;
      resblock_tmp2.conv_groups = param.all_resblock_conv_groups[i][j];
      resblock_tmp2.kernel_dims = param.all_resblock_filter_dims[i][j];
      resblock_tmp2.dilations = param.all_resblock_conv_dilations[i][j];
      resblock_tmp2.paddings = param.all_resblock_conv_paddings[i][j];
      resblock_tmp2.strides = param.all_resblock_conv_strides[i][j];
      resblock_tmp2.gn_groups = param.all_resblock_gn_groups[i][j];
      resblock_tmp2.gn_eps = param.all_resblock_gn_eps[i][j];
      resblocks_tmp1.push_back(resblock_tmp2);
    }
    multi_up_decoder_param_.resblock_params.push_back(resblocks_tmp1);
  }

  // last gn silu param
  multi_up_decoder_param_.last_gn_eps = param.last_gn_eps;
  multi_up_decoder_param_.last_gn_groups = param.last_gn_groups;
  multi_up_decoder_param_.last_gn_silu = param.has_last_gn_silu;

  // prepare reslobck conv bias/gn scale/gn bias.
  int tmp_idx_res_conv = 0;
  int tmp_idx_res_gn = 0;
  const int offset = 2;
  for (int i = 0; i < param.all_resblock_conv_fix.size(); i++) {
    std::vector<std::vector<xft::xftVec<float>>> tmp_1_conv_bias;
    std::vector<std::vector<xft::xftVec<float>>> tmp_1_gn_scale;
    std::vector<std::vector<xft::xftVec<float>>> tmp_1_gn_bias;
    for (int j = 0; j < param.all_resblock_conv_fix[i].size(); j++) {
      std::vector<xft::xftVec<float>> tmp_2_conv_bias;
      std::vector<xft::xftVec<float>> tmp_2_gn_scale;
      std::vector<xft::xftVec<float>> tmp_2_gn_bias;
      int tmp_idx_res_conv_buff = tmp_idx_res_conv;
      int tmp_idx_res_gn_buff = tmp_idx_res_gn;
      tmp_idx_res_conv += offset + param.all_resblock_conv_fix[i][j];
      tmp_idx_res_gn += offset;
      for (int k = tmp_idx_res_conv_buff; k < tmp_idx_res_conv; k++) {
        tmp_2_conv_bias.emplace_back(
            const_cast<float*>(
                param.all_resblock_conv_bias[k]->template data<float>()),
            xft::xftVec<float>::dim_t{
                param.all_resblock_conv_bias[k]->dims()[0]});
      }
      for (int k = tmp_idx_res_gn_buff; k < tmp_idx_res_gn; k++) {
        tmp_2_gn_scale.emplace_back(
            const_cast<float*>(
                param.all_resblock_gn_scale[k]->template data<float>()),
            xft::xftVec<float>::dim_t{
                param.all_resblock_gn_scale[k]->dims()[0]});
        tmp_2_gn_bias.emplace_back(
            const_cast<float*>(
                param.all_resblock_gn_bias[k]->template data<float>()),
            xft::xftVec<float>::dim_t{
                param.all_resblock_gn_bias[k]->dims()[0]});
      }
      tmp_1_conv_bias.push_back(std::move(tmp_2_conv_bias));
      tmp_1_gn_bias.push_back(std::move(tmp_2_gn_bias));
      tmp_1_gn_scale.push_back(std::move(tmp_2_gn_scale));
    }
    xft_all_res_conv_bias_.push_back(std::move(tmp_1_conv_bias));
    xft_all_res_gn_bias_.push_back(std::move(tmp_1_gn_bias));
    xft_all_res_gn_weight_.push_back(std::move(tmp_1_gn_scale));
  }

  if (param.has_last_gn_silu) {
    xft_last_gn_weight_ = xft::xftVec<float>(
        const_cast<float*>(param.last_gn_scale->template data<float>()),
        xft::xftVec<float>::dim_t{param.last_gn_scale->dims()[0]});
    xft_last_gn_bias_ = xft::xftVec<float>(
        const_cast<float*>(param.last_gn_bias->template data<float>()),
        xft::xftVec<float>::dim_t{param.last_gn_bias->dims()[0]});
  }

  // prepare resblock conv weights
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  arg_all_res_conv_filter_int16_ = PrepareResConvWeight<int16_t>(
      param.all_resblock_conv_filter, param.all_resblock_conv_fix, 2);
  PrepareAllResConvFilterMax(param.all_resblock_filter_max,
                             param.all_resblock_conv_fix,
                             XPU_QUANT_SCALE_NUM,
                             2,
                             &all_res_conv_filter_max_);
  for (int i = 0; i < arg_all_res_conv_filter_int16_.size(); i++) {
    std::vector<std::vector<xft::xftTensor<int16_t, 4>>> tmp_1_conv_weight;
    for (int j = 0; j < arg_all_res_conv_filter_int16_[i].size(); j++) {
      std::vector<xft::xftTensor<int16_t, 4>> tmp_2_conv_weight;
      for (int k = 0; k < arg_all_res_conv_filter_int16_[i][j].size(); k++) {
        tmp_2_conv_weight.emplace_back(
            const_cast<int16_t*>(arg_all_res_conv_filter_int16_[i][j][k]),
            const_cast<float*>(all_res_conv_filter_max_[i][j][k]),
            xft::xftTensor<int16_t, 4>::dim_t{
                param.all_resblock_filter_dims[i][j][k][0],
                param.all_resblock_filter_dims[i][j][k][1],
                param.all_resblock_filter_dims[i][j][k][2],
                param.all_resblock_filter_dims[i][j][k][3]});
      }

      tmp_1_conv_weight.push_back(std::move(tmp_2_conv_weight));
    }
    xft_all_res_conv_weight_.emplace_back(std::move(tmp_1_conv_weight));
  }
  // prepare post conv weight
  arg_all_post_conv_filter_int16_ =
      PreparePostConvWeight<int16_t>(param.all_post_conv_filter);
  PreparePostConvFilterMax(param.all_post_filter_max,
                           XPU_QUANT_SCALE_NUM,
                           &all_post_conv_filter_max_);
  for (int i = 0; i < arg_all_post_conv_filter_int16_.size(); i++) {
    xft_all_post_conv_weight_.emplace_back(
        const_cast<int16_t*>(arg_all_post_conv_filter_int16_[i]),
        const_cast<float*>(all_post_conv_filter_max_[i]),
        xft::xftTensor<int16_t, 4>::dim_t{param.all_post_filter_dims[i][0],
                                          param.all_post_filter_dims[i][1],
                                          param.all_post_filter_dims[i][2],
                                          param.all_post_filter_dims[i][3]});
  }
  // prepare post conv bias
  for (auto* bias : param.all_post_conv_bias) {
    int bias_dim = bias->dims()[0];
    xft_all_post_conv_bias_.emplace_back(
        const_cast<float*>(bias->template data<float>()),
        xft::xftVec<float>::dim_t{bias_dim});
  }
}

template <typename InType, PrecisionType PType>
void XPUMultiUpDecoderCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const InType* in1 = param.input1->template data<InType>();
  InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  int batch = static_cast<int>(param.input1->dims()[0]);
  int channel = static_cast<int>(param.input1->dims()[1]);
  int nh = static_cast<int>(param.input1->dims()[2]);
  int nw = static_cast<int>(param.input1->dims()[3]);

  int channel_out = static_cast<int>(param.output->dims()[1]);
  int nh_out = static_cast<int>(param.output->dims()[2]);
  int nw_out = static_cast<int>(param.output->dims()[3]);

  // Input
  xft::xftTensor<InType, 4> in_tensor(
      const_cast<InType*>(in1), nullptr, {batch, channel, nh, nw});
  // Output
  xft::xftTensor<InType, 4> output_tensor(out,
                                          {batch, channel_out, nh_out, nw_out});

  int r = xft::multi_up_decoder_fusion<InType, int16_t, int16_t>(
      ctx.GetRawContext(),
      in_tensor,
      xft_all_res_gn_weight_,
      xft_all_res_gn_bias_,
      xft_all_res_conv_weight_,
      xft_all_res_conv_bias_,
      xft_all_post_conv_weight_,
      xft_all_post_conv_bias_,
      xft_last_gn_weight_,
      xft_last_gn_bias_,
      &output_tensor,
      multi_up_decoder_param_);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUMultiUpDecoder_FP32 =
    xpu::XPUMultiUpDecoderCompute<float, PRECISION(kFloat)>;
using XPUMultiUpDecoder_FP16 =
    xpu::XPUMultiUpDecoderCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    __xpu__multi_up_decoder, kXPU, kFloat, kNCHW, XPUMultiUpDecoder_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderResConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderResConvFilter",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderGNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderGNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderInputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderPostConvFilter",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderPostConvBias",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderPostConvInputMax",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LastGNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LastGNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    __xpu__multi_up_decoder, kXPU, kFP16, kNCHW, XPUMultiUpDecoder_FP16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("AllUpDecoderResConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderResConvFilter",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderGNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderGNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderInputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderPostConvFilter",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderPostConvBias",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AllUpDecoderPostConvInputMax",
               {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LastGNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("LastGNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
