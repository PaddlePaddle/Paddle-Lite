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

#include "lite/kernels/xpu/__xpu__spatial_transformer_resblock_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
static std::vector<const T*> PrepareWeight(
    const std::vector<lite::Tensor*>& fc_weight) {
  std::vector<const T*> result;
  for (auto* weight : fc_weight) {
    result.push_back(reinterpret_cast<const T*>(weight->data<float>()));
  }
  return result;
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerResBlockCompute<InType, PType>::PrepareWeightMax(
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : weight_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
  weight_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* weight_max_ptr = reinterpret_cast<float*>(weight_max_guard_->addr_);

  int offset = 0;
  for (auto max_tensor : weight_max) {
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
  }
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerResBlockCompute<InType, PType>::PrepareFilterMax(
    const std::vector<lite::Tensor*>& filter_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : filter_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
  filter_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* filter_max_ptr = reinterpret_cast<float*>(filter_max_guard_->addr_);

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
    max_xpu_ptrs->push_back(cur_filter_max_ptr);
    offset += max_ptr_len;
  }
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerResBlockCompute<InType, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();

  // achieve params from model
  resblock_param_.conv_fix = param.conv_fix;
  resblock_param_.has_silu_fc_input = param.has_silu_fc_input;
  resblock_param_.include_silu = param.include_silu;
  resblock_param_.conv_groups = param.groups;
  resblock_param_.kernel_dims = param.filter_dims;
  resblock_param_.dilations = param.dilations;
  resblock_param_.paddings = param.paddings;
  resblock_param_.strides = param.strides;
  resblock_param_.gn_groups = param.gn_groups;
  resblock_param_.gn_eps = param.gn_eps;

  // prepare bias
  for (auto* fc_bias : param.fc_bias) {
    xft_fc_bias_.emplace_back(
        const_cast<float*>(fc_bias->template data<float>()),
        xft::xftVec<float>::dim_t{fc_bias->dims()[0]});
  }
  for (auto* conv_bias : param.conv_bias) {
    xft_conv_bias_.emplace_back(
        const_cast<float*>(conv_bias->template data<float>()),
        xft::xftVec<float>::dim_t{conv_bias->dims()[0]});
  }
  // prepare scale
  for (auto* gn_scale : param.gn_scale) {
    xft_gn_weight_.emplace_back(
        const_cast<float*>(gn_scale->template data<float>()),
        xft::xftVec<float>::dim_t{gn_scale->dims()[0]});
  }
  // prepare gn_bias
  for (auto* gn_bias : param.gn_bias) {
    xft_gn_bias_.emplace_back(
        const_cast<float*>(gn_bias->template data<float>()),
        xft::xftVec<float>::dim_t{gn_bias->dims()[0]});
  }
  for (auto* input_max : param.input_max) {
    input_max_.push_back(input_max->template data<float>());
  }
  if (input_max_.size() == 0) {
    input_max_.push_back(nullptr);
  }

  // prepare conv params
  arg_conv_filter_int16_ = PrepareWeight<int16_t>(param.conv_filter);
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  PrepareFilterMax(param.filter_max, XPU_QUANT_SCALE_NUM, &conv_filter_max_);
  int channel = static_cast<int>(param.input1->dims()[1]);
  for (size_t i = 0; i < arg_conv_filter_int16_.size(); i++) {
    int xn = param.filter_dims[i][0];
    int nh = param.filter_dims[i][2];
    int nw = param.filter_dims[i][3];
    xft_conv_weights_.emplace_back(
        const_cast<int16_t*>(arg_conv_filter_int16_[i]),
        const_cast<float*>(conv_filter_max_[i]),
        xft::xftTensor<int16_t, 4>::dim_t{channel, xn, nh, nw});
  }

  // prepare fc params
  if (resblock_param_.has_silu_fc_input) {
    arg_fc_weight_int16_ = PrepareWeight<int16_t>(param.fc_weight);
    PrepareWeightMax(param.weight_max, XPU_QUANT_SCALE_NUM, &fc_weight_max_);
    int input2_dim = static_cast<int>(param.input2->dims()[1]);
    for (size_t i = 0; i < arg_fc_weight_int16_.size(); i++) {
      int xn = param.filter_dims[i][0];
      xft_fc_weights_.emplace_back(
          const_cast<int16_t*>(arg_fc_weight_int16_[i]),
          const_cast<float*>(fc_weight_max_[i]),
          xft::xftMat<int16_t>::dim_t{xn, input2_dim});
    }
  }
}

template <typename InType, PrecisionType PType>
void XPUSpatialTransformerResBlockCompute<InType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const InType* in1 = param.input1->template data<InType>();
  const InType* in2 = (param.has_silu_fc_input)
                          ? param.input2->template data<InType>()
                          : nullptr;
  InType* out = param.output->template mutable_data<InType>(TARGET(kXPU));
  int batch = static_cast<int>(param.input1->dims()[0]);
  int channel = static_cast<int>(param.input1->dims()[1]);
  int nh = static_cast<int>(param.input1->dims()[2]);
  int nw = static_cast<int>(param.input1->dims()[3]);
  int input2_dim = (param.has_silu_fc_input)
                       ? static_cast<int>(param.input2->dims()[1])
                       : -1;
  // input
  xft::xftTensor<InType, 4> in_tensor(const_cast<InType*>(in1),
                                      const_cast<float*>(input_max_[0]),
                                      {batch, channel, nh, nw});
  xft::xftMat<InType> in_silu_tensor(
      const_cast<InType*>(in2), nullptr, {batch, input2_dim});
  // output
  xft::xftTensor<InType, 4> output_tensor(out, {batch, channel, nh, nw});
  int r = xft::st_resblock_fusion<InType, int16_t, int16_t>(ctx.GetRawContext(),
                                                            in_tensor,
                                                            in_silu_tensor,
                                                            xft_gn_weight_,
                                                            xft_gn_bias_,
                                                            xft_fc_weights_,
                                                            xft_fc_bias_,
                                                            xft_conv_weights_,
                                                            xft_conv_bias_,
                                                            &output_tensor,
                                                            resblock_param_);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using XPUSpatialTransformerResBlock_FP32 =
    xpu::XPUSpatialTransformerResBlockCompute<float, PRECISION(kFloat)>;
using XPUSpatialTransformerResBlock_FP16 =
    xpu::XPUSpatialTransformerResBlockCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(__xpu__spatial_transformer_resblock,
                     kXPU,
                     kFloat,
                     kNCHW,
                     XPUSpatialTransformerResBlock_FP32,
                     def)
    .BindInput("Input1", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Input2", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(__xpu__spatial_transformer_resblock,
                     kXPU,
                     kFP16,
                     kNCHW,
                     XPUSpatialTransformerResBlock_FP16,
                     def)
    .BindInput("Input1",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Input2",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCWeight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FCBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ConvBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNScale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("GNBias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
