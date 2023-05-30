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

#include "lite/kernels/xpu/__xpu__up_decoder_compute.h"
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> PrepareWeight(
    const std::vector<lite::Tensor*>& op_weight) {
  std::vector<const T*> result;
  for (auto* weight : op_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}
template <typename T>
static std::vector<std::vector<const T*>> PrepareResConvWeight(
    const std::vector<lite::Tensor*>& op_weight,
    const std::vector<int>& extra_info,
    int conv_offset) {
  int tmp_idx = 0;
  std::vector<std::vector<const T*>> result;
  for (int j = 0; j < extra_info.size(); j++) {
    std::vector<const T*> tmp_2;
    int tmp_idx_buff = tmp_idx;
    tmp_idx += (conv_offset + extra_info[j]);
    for (int k = tmp_idx_buff; k < tmp_idx; k++) {
      tmp_2.push_back(reinterpret_cast<const T*>(op_weight[k]->data<float>()));
    }
    result.push_back(tmp_2);
  }
  return result;
}

template <typename InType, PrecisionType PType>
void XPUUpDecoderCompute<InType, PType>::PrepareResConvFilterMax(
    const std::vector<lite::Tensor*>& filter_max,
    const std::vector<int>& extra_info,
    int max_ptr_len,
    int conv_offset,
    std::vector<std::vector<const float*>>* max_xpu_ptrs) {
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
  for (int j = 0; j < extra_info.size(); j++) {
    std::vector<const float*> tmp_2;
    int tmp_idx_buff = tmp_idx;
    tmp_idx += (conv_offset + extra_info[j]);
    for (int k = tmp_idx_buff; k < tmp_idx; k++) {
      tmp_2.push_back(max_xpu_ptrs_buff[k]);
    }
    max_xpu_ptrs->push_back(tmp_2);
  }
}

template <typename InType, PrecisionType PType>
void XPUUpDecoderCompute<InType, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();

  // prepare params
  up_decoder_param_.num_resblocks = param.num_resblocks;

  // Resblock params
  for (int j = 0; j < param.resblock_conv_fix.size(); j++) {
    xft::STResBlockParam resblock_tmp2;
    resblock_tmp2.conv_fix = static_cast<bool>(param.resblock_conv_fix[j]);
    resblock_tmp2.has_silu_fc_input = false;
    resblock_tmp2.conv_groups = param.resblock_conv_groups[j];
    resblock_tmp2.kernel_dims = param.resblock_filter_dims[j];
    resblock_tmp2.dilations = param.resblock_conv_dilations[j];
    resblock_tmp2.paddings = param.resblock_conv_paddings[j];
    resblock_tmp2.strides = param.resblock_conv_strides[j];
    resblock_tmp2.gn_groups = param.resblock_gn_groups[j];
    resblock_tmp2.gn_eps = param.resblock_gn_eps[j];
    up_decoder_param_.resblock_params.push_back(resblock_tmp2);
  }

  // prepare reslobck conv bias/gn scale/gn bias.
  int tmp_idx_res_conv = 0;
  int tmp_idx_res_gn = 0;
  const int offset = 2;
  std::vector<std::vector<xft::xftVec<float>>> tmp_1_conv_bias;
  std::vector<std::vector<xft::xftVec<float>>> tmp_1_gn_scale;
  std::vector<std::vector<xft::xftVec<float>>> tmp_1_gn_bias;
  for (int j = 0; j < param.resblock_conv_fix.size(); j++) {
    std::vector<xft::xftVec<float>> tmp_2_conv_bias;
    std::vector<xft::xftVec<float>> tmp_2_gn_scale;
    std::vector<xft::xftVec<float>> tmp_2_gn_bias;
    int tmp_idx_res_conv_buff = tmp_idx_res_conv;
    int tmp_idx_res_gn_buff = tmp_idx_res_gn;
    tmp_idx_res_conv += offset + param.resblock_conv_fix[j];
    tmp_idx_res_gn += offset;
    for (int k = tmp_idx_res_conv_buff; k < tmp_idx_res_conv; k++) {
      tmp_2_conv_bias.emplace_back(
          const_cast<float*>(
              param.resblock_conv_bias[k]->template data<float>()),
          xft::xftVec<float>::dim_t{param.resblock_conv_bias[k]->dims()[0]});
    }
    for (int k = tmp_idx_res_gn_buff; k < tmp_idx_res_gn; k++) {
      tmp_2_gn_scale.emplace_back(
          const_cast<float*>(
              param.resblock_gn_scale[k]->template data<float>()),
          xft::xftVec<float>::dim_t{param.resblock_gn_scale[k]->dims()[0]});
      tmp_2_gn_bias.emplace_back(
          const_cast<float*>(param.resblock_gn_bias[k]->template data<float>()),
          xft::xftVec<float>::dim_t{param.resblock_gn_bias[k]->dims()[0]});
    }
    xft_res_conv_bias_.push_back(std::move(tmp_2_conv_bias));
    xft_res_gn_bias_.push_back(std::move(tmp_2_gn_bias));
    xft_res_gn_weight_.push_back(std::move(tmp_2_gn_scale));
  }

  // prepare resblock conv weights
  auto resblock_conv_fix = param.resblock_conv_fix;
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  res_conv_filter_int16_ = PrepareResConvWeight<int16_t>(
      param.resblock_conv_filter, resblock_conv_fix, 2);
  PrepareResConvFilterMax(param.resblock_filter_max,
                          resblock_conv_fix,
                          XPU_QUANT_SCALE_NUM,
                          2,
                          &res_conv_filter_max_);
  std::vector<std::vector<xft::xftTensor<int16_t, 4>>> tmp_1_conv_weight;
  for (int j = 0; j < res_conv_filter_int16_.size(); j++) {
    std::vector<xft::xftTensor<int16_t, 4>> tmp_2_conv_weight;
    for (int k = 0; k < res_conv_filter_int16_[j].size(); k++) {
      tmp_2_conv_weight.emplace_back(
          const_cast<int16_t*>(res_conv_filter_int16_[j][k]),
          const_cast<float*>(res_conv_filter_max_[j][k]),
          xft::xftTensor<int16_t, 4>::dim_t{
              param.resblock_filter_dims[j][k][0],
              param.resblock_filter_dims[j][k][1],
              param.resblock_filter_dims[j][k][2],
              param.resblock_filter_dims[j][k][3]});
    }

    xft_res_conv_weight_.push_back(std::move(tmp_2_conv_weight));
  }
}

template <typename InType, PrecisionType PType>
void XPUUpDecoderCompute<InType, PType>::Run() {
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

  int r = xft::up_decoder_fusion<InType, int16_t, int16_t>(ctx.GetRawContext(),
                                                           in_tensor,
                                                           xft_res_gn_weight_,
                                                           xft_res_gn_bias_,
                                                           xft_res_conv_weight_,
                                                           xft_res_conv_bias_,
                                                           &output_tensor,
                                                           up_decoder_param_);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUUpDecoder_FP32 = xpu::XPUUpDecoderCompute<float, PRECISION(kFloat)>;
using XPUUpDecoder_FP16 = xpu::XPUUpDecoderCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    __xpu__up_decoder, kXPU, kFloat, kNCHW, XPUUpDecoder_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockConvFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockGNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockGNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockInputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PostConvFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PostConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PostConvInputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    __xpu__up_decoder, kXPU, kFP16, kNCHW, XPUUpDecoder_FP16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("ResblockConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockConvFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockGNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockGNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ResblockInputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PostConvFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PostConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("PostConvInputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
