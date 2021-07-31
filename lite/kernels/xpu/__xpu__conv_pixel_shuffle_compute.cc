// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__conv_pixel_shuffle_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
bool _QuantFilter(const float* filter_on_host,
                  T* quant_res,
                  float max,
                  int64_t len) {
  return false;
}

template <>
bool _QuantFilter<int16_t>(const float* filter_on_host,
                           int16_t* quant_res,
                           float max,
                           int64_t len) {
  paddle::lite::xpu::math::ConvertFP32ToInt16(
      filter_on_host, quant_res, max, len);
  return true;
}

template <>
bool _QuantFilter<int8_t>(const float* filter_on_host,
                          int8_t* quant_res,
                          float max,
                          int64_t len) {
  paddle::lite::xpu::math::ConvertFP32ToInt8(
      filter_on_host, quant_res, max, len);
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

template <typename TM, typename TW, PrecisionType PType>
void XPUConvPixelShuffleCompute<TM, TW, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto filter_ptr = param.filter_0->template data<float>();
  auto filter_len = param.filter_0->numel();
  // max_0
  float max_f = paddle::lite::xpu::math::FindMaxAbs(filter_ptr, filter_len);
  std::vector<float> max_f_v(4, max_f);
  filter_max_guard_0_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  filter_max_0_ = reinterpret_cast<float*>(filter_max_guard_0_->addr_);
  XPU_CALL(xpu_memcpy(filter_max_0_,
                      max_f_v.data(),
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  // quant_0
  quant_filter_guard_0_ =
      TargetWrapperXPU::MallocScratchPad(filter_len * sizeof(TW));
  quant_filter_0_ = reinterpret_cast<TW*>(quant_filter_guard_0_->addr_);
  std::vector<TW> quant_filter_cpu(filter_len, 0);
  bool ret =
      _QuantFilter<TW>(filter_ptr, quant_filter_cpu.data(), max_f, filter_len);
  CHECK_EQ(ret, true);
  XPU_CALL(xpu_memcpy(quant_filter_0_,
                      quant_filter_cpu.data(),
                      filter_len * sizeof(TW),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  filter_ptr = param.filter_1->template data<float>();
  filter_len = param.filter_1->numel();
  // max_1
  max_f = paddle::lite::xpu::math::FindMaxAbs(filter_ptr, filter_len);
  max_f_v = std::vector<float>(4, max_f);
  filter_max_guard_1_ = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  filter_max_1_ = reinterpret_cast<float*>(filter_max_guard_1_->addr_);
  XPU_CALL(xpu_memcpy(filter_max_1_,
                      max_f_v.data(),
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  // quant_1
  quant_filter_guard_1_ =
      TargetWrapperXPU::MallocScratchPad(filter_len * sizeof(TW));
  quant_filter_1_ = reinterpret_cast<TW*>(quant_filter_guard_1_->addr_);
  quant_filter_cpu = std::vector<TW>(filter_len, 0);
  ret =
      _QuantFilter<TW>(filter_ptr, quant_filter_cpu.data(), max_f, filter_len);
  CHECK_EQ(ret, true);
  XPU_CALL(xpu_memcpy(quant_filter_1_,
                      quant_filter_cpu.data(),
                      filter_len * sizeof(TW),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  // mid max
  mid_max = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  mid_max_addr = reinterpret_cast<float*>(mid_max->addr_);
}

template <typename TM, typename TW, PrecisionType PType>
void XPUConvPixelShuffleCompute<TM, TW, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& input_dims = param.input->dims();
  auto& filter_0_dims = param.filter_0->dims();
  int batch = static_cast<int>(input_dims[0]);
  int img_c = static_cast<int>(input_dims[1]);
  int img_h = static_cast<int>(input_dims[2]);
  int img_w = static_cast<int>(input_dims[3]);
  int filter_num = static_cast<int>(filter_0_dims[0]);
  int win_h = static_cast<int>(filter_0_dims[2]);
  int win_w = static_cast<int>(filter_0_dims[3]);
  auto paddings = *param.paddings_0;
  auto dilations = *param.dilations_0;
  int groups = param.groups_0.front();
  int act_type = param.act_type_0.front();
  const auto* bias_0 =
      param.has_bias_0 ? param.bias_0->template data<float>() : nullptr;
  const float* input_max =
      param.input_max ? param.input_max->template data<float>() : nullptr;
  xdnn::Activation_t act((xdnn::Activation_t::act_enum)act_type);
  if (act_type == 5) {
    act.leaky_alpha = param.act_param_0.front();
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (act_type == 15) {
    act.hard_sigmoid_slope = param.act_param_0.front();
  }
  param.output->clear();

  // get mid shape
  std::vector<int64_t> mid_shape({input_dims[0], filter_0_dims[0]});
  for (size_t i = 0; i < param.strides_0.size(); ++i) {
    mid_shape.push_back(ConvOutputSize(input_dims[i + 2],
                                       filter_0_dims[i + 2],
                                       dilations[i],
                                       paddings[i * 2],
                                       paddings[i * 2 + 1],
                                       param.strides_0[i]));
  }

  size_t mid_size = 1;
  for (auto shape : mid_shape) {
    mid_size *= static_cast<size_t>(shape);
  }

  auto mid_tensor = TargetWrapperXPU::MallocScratchPad(mid_size * sizeof(TM));
  TM* mid_tensor_addr = reinterpret_cast<TM*>(mid_tensor->addr_);
  auto mid_out_tensor =
      TargetWrapperXPU::MallocScratchPad(mid_size * sizeof(TM));
  TM* mid_out_tensor_addr = reinterpret_cast<TM*>(mid_out_tensor->addr_);

  int r = xdnn::conv2d_fusion<float, TW, TM, TW>(
      ctx.GetRawContext(),
      param.input->template data<float>(),
      quant_filter_0_,
      mid_tensor_addr,
      batch,
      img_c,
      img_h,
      img_w,
      filter_num,
      std::vector<int>{win_h, win_w},
      param.strides_0,
      paddings,
      dilations,
      groups,
      input_max,
      filter_max_0_,
      mid_max_addr,
      true,
      bias_0,
      nullptr,
      act);
  CHECK_EQ(r, 0);

  const auto upscale_factor = param.upscale_factor;
  r = xdnn::pixel_shuffle<TM>(ctx.GetRawContext(),
                              mid_tensor_addr,
                              mid_out_tensor_addr,
                              mid_shape[0],
                              mid_shape[1],
                              mid_shape[2],
                              mid_shape[3],
                              upscale_factor,
                              true);
  CHECK_EQ(r, 0);
  mid_tensor.reset();

  mid_shape = {mid_shape[0],
               mid_shape[1] / upscale_factor / upscale_factor,
               mid_shape[2] * upscale_factor,
               mid_shape[3] * upscale_factor};

  auto& filter_1_dims = param.filter_1->dims();
  filter_num = static_cast<int>(filter_1_dims[0]);
  win_h = static_cast<int>(filter_1_dims[2]);
  win_w = static_cast<int>(filter_1_dims[3]);
  paddings = *param.paddings_1;
  dilations = *param.dilations_1;
  groups = param.groups_1.front();
  float* output_max =
      param.output_max->template mutable_data<float>(TARGET(kXPU));
  float* output = param.output->template mutable_data<float>(TARGET(kXPU));
  act_type = param.act_type_1.front();
  const auto* bias_1 =
      param.has_bias_1 ? param.bias_1->template data<float>() : nullptr;
  act = xdnn::Activation_t((xdnn::Activation_t::act_enum)act_type);
  if (act_type == 5) {
    act.leaky_alpha = param.act_param_1.front();
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (act_type == 15) {
    act.hard_sigmoid_slope = param.act_param_1.front();
  }
  r = xdnn::conv2d_fusion<TM, TW, float, TW>(ctx.GetRawContext(),
                                             mid_out_tensor_addr,
                                             quant_filter_1_,
                                             output,
                                             mid_shape[0],
                                             mid_shape[1],
                                             mid_shape[2],
                                             mid_shape[3],
                                             filter_num,
                                             std::vector<int>{win_h, win_w},
                                             param.strides_1,
                                             paddings,
                                             dilations,
                                             groups,
                                             mid_max_addr,
                                             filter_max_1_,
                                             output_max,
                                             true,
                                             bias_1,
                                             nullptr,
                                             act);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using XPUConvPixelShuffleFp32 =
    xpu::XPUConvPixelShuffleCompute<float, int16_t, PRECISION(kFloat)>;
using XPUConvPixelShuffleFp16 =
    xpu::XPUConvPixelShuffleCompute<float16, int16_t, PRECISION(kFP16)>;
using XPUConvPixelShuffleInt8 =
    xpu::XPUConvPixelShuffleCompute<float, int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(__xpu__conv_pixel_shuffle_fuse_op,
                     kXPU,
                     kInt8,
                     kNCHW,
                     XPUConvPixelShuffleInt8,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter_0", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Filter_1", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias_0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias_1", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__conv_pixel_shuffle_fuse_op,
                     kXPU,
                     kFP16,
                     kNCHW,
                     XPUConvPixelShuffleFp16,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter_0", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Filter_1", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias_0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias_1", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__conv_pixel_shuffle_fuse_op,
                     kXPU,
                     kFloat,
                     kNCHW,
                     XPUConvPixelShuffleFp32,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter_0", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Filter_1", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias_0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias_1", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
