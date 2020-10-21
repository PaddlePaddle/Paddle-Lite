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

#include "lite/kernels/fpga/elementwise_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/kernels/fpga/activation_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void ElementwiseAddCompute::PrepareForRun() {
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  auto& param = Param<operators::ElementwiseParam>();

  param.Out->mutable_data<float16>();
  ew_param.inputs = {param.X->ZynqTensor(), param.Y->ZynqTensor()};
  ew_param.output = param.Out->ZynqTensor();
  ew_param.axis = param.axis;
  ew_param.activeParam.type = zynqmp::TYPE_NONE;

  pe_.init();
  pe_.apply();
}
void ElementwiseAddCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  Debugger::get_instance().registerOutput("ew_add", ew_param.output);
#endif
}

void ElementwiseAddActivationCompute::PrepareForRun() {
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  auto& param = Param<operators::FusionElementwiseActivationParam>();

  if (activation_map.count(param.act_type)) {
    ew_param.activeParam.type = activation_map[param.act_type];
  } else {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }

  param.Out->mutable_data<float16>();
  ew_param.inputs = {param.X->ZynqTensor(), param.Y->ZynqTensor()};
  ew_param.output = param.Out->ZynqTensor();
  ew_param.axis = param.axis;
  pe_.init();
  pe_.apply();
}
void ElementwiseAddActivationCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  Debugger::get_instance().registerOutput("ew_add", ew_param.output);
#endif
}

void ElementwiseMulCompute::PrepareForRun() {
  zynqmp::ScaleParam& scale_param = pe_.param();
  auto& param = Param<operators::ElementwiseParam>();
  param.Out->mutable_data<float16>();

  scale_param.input = param.X->ZynqTensor();
  scale_param.output = param.Out->ZynqTensor();
  scale_param.activeParam.type = zynqmp::TYPE_NONE;

  int channel = scale_param.input->shape().channel();
  scale_param.scale = &scale_;
  scale_param.bias = &bias_;
  zynqmp::Shape shape(zynqmp::N, {channel});
  zynqmp::float16* scale_data =
      scale_.mutableData<zynqmp::float16>(zynqmp::FP16, shape);
  zynqmp::float16* bias_data =
      bias_.mutableData<zynqmp::float16>(zynqmp::FP16, shape);
  zynqmp::float16 scale_value = 0;
  if (param.Y->ZynqTensor()->dataType() == zynqmp::FP32) {
    scale_value = zynqmp::float_to_half(param.Y->data<float>()[0]);
  } else {
    scale_value = param.Y->data<zynqmp::float16>()[0];
  }

  for (int i = 0; i < channel; i++) {
    if (param.Y->dims().production() != 1) {
      if (param.Y->ZynqTensor()->dataType() == zynqmp::FP32) {
        scale_value = zynqmp::float_to_half(param.Y->data<float>()[i]);
      } else {
        scale_value = param.Y->data<zynqmp::float16>()[i];
      }
    }
    scale_data[i] = scale_value;
    bias_data[i] = zero_;
  }

  pe_.init();
  pe_.apply();
}

void ElementwiseMulCompute::Run() {
  auto& param = Param<operators::ElementwiseParam>();

  if (!param.Y->persistable()) {
    // TODO(chonwhite) alignment;

    param.Y->ZynqTensor()->invalidate();
    scale_.copyFrom(param.Y->ZynqTensor());
    scale_.flush();
  }
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::ScaleParam& scale_param = pe_.param();
  Debugger::get_instance().registerOutput("ew_mul", scale_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ElementwiseAddCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kFPGA,
    kFP16,
    kNHWC,
    paddle::lite::kernels::fpga::ElementwiseAddActivationCompute,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ElementwiseMulCompute,
                     ew_mul_fpga)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ElementwiseMulCompute,
                     ew_mul_y_arm)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
