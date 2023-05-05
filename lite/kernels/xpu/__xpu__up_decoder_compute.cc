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

template <typename InType, PrecisionType PType>
void XPUUpDecoderCompute<InType, PType>::PreparePostConvFilterMax(
    const std::vector<lite::Tensor*>& weight_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : weight_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
  post_conv_filter_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_value_num * sizeof(float));
  float* weight_max_ptr =
      reinterpret_cast<float*>(post_conv_filter_max_guard_->addr_);

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
void XPUUpDecoderCompute<InType, PType>::PrepareResblockConvFilterMax(
    const std::vector<lite::Tensor*>& filter_max,
    int max_ptr_len,
    std::vector<const float*>* max_xpu_ptrs) {
  int max_value_num = 0;
  for (auto max_tensor : filter_max) {
    max_value_num += max_tensor->numel();
  }
  VLOG(3) << "Total weight max value number: " << max_value_num;
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
    max_xpu_ptrs->push_back(cur_filter_max_ptr);
    offset += max_ptr_len;
  }
}

template <typename InType, PrecisionType PType>
void XPUUpDecoderCompute<InType, PType>::PrepareForRun() {
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->template Param<param_t>();

  for (auto* conv_bias : param.resblock_conv_bias) {
    xft_resblock_conv_bias_.emplace_back(
        const_cast<float*>(conv_bias->template data<float>()),
        xft::xftVec<float>::dim_t{conv_bias->dims()[0]});
  }
  // prepare scale
  for (auto* gn_scale : param.resblock_gn_scale) {
    xft_resblock_gn_weight_.emplace_back(
        const_cast<float*>(gn_scale->template data<float>()),
        xft::xftVec<float>::dim_t{gn_scale->dims()[0]});
  }
  // prepare gn_bias
  for (auto* gn_bias : param.resblock_gn_bias) {
    xft_resblock_gn_bias_.emplace_back(
        const_cast<float*>(gn_bias->template data<float>()),
        xft::xftVec<float>::dim_t{gn_bias->dims()[0]});
  }
  for (auto* input_max : param.resblock_conv_input_max) {
    resblock_input_max_.push_back(input_max->template data<float>());
  }
  if (resblock_input_max_.size() == 0) {
    resblock_input_max_.push_back(nullptr);
  }

  // prepare resblock conv params
  arg_resblock_conv_filter_int16_ =
      PrepareWeight<int16_t>(param.resblock_conv_filter);
  const int XPU_QUANT_SCALE_NUM = ctx.GetRawContext()->max_ptr_size();
  PrepareResblockConvFilterMax(param.resblock_filter_max,
                               XPU_QUANT_SCALE_NUM,
                               &resblock_conv_filter_max_);
  int channel = static_cast<int>(param.input1->dims()[1]);

  // param.resblock_filter_dims has 3 dims: num_resblock; num_conv in each
  // resblock; dim_info for each conv.
  int i_count = 0;
  for (size_t resblock_id = 0; resblock_id < param.resblock_filter_dims.size();
       ++resblock_id) {
    for (size_t conv_id = 0;
         conv_id < param.resblock_filter_dims[resblock_id].size();
         ++conv_id) {
      int xn = param.resblock_filter_dims[resblock_id][conv_id][0];
      int nh = param.resblock_filter_dims[resblock_id][conv_id][2];
      int nw = param.resblock_filter_dims[resblock_id][conv_id][3];
      xft_resblock_conv_weights_.emplace_back(
          const_cast<int16_t*>(arg_resblock_conv_filter_int16_[i_count]),
          const_cast<float*>(resblock_conv_filter_max_[i_count]),
          xft::xftTensor<int16_t, 4>::dim_t{channel, xn, nh, nw});
      i_count++;
    }
  }
}

template <typename InType, PrecisionType PType>
void XPUUpDecoderCompute<InType, PType>::Run() {
  // TODO(shenyijun01): Currently, this op is an intermediate op of
  // multi-up-decoder fused op.
  // The compute of this op will be adapted to XFT interface later on.
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
