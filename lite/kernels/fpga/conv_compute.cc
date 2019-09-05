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

#include "lite/kernels/fpga/conv_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void ConvCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  // ====================================================
  if (param.groups == param.x->ZynqTensor()->shape().channel()) {
    std::cout << "dw_conv \n";
    // exit(-1);
    zynqmp::DepthwiseConvParam& conv_param = dw_conv_pe_.param();
    param.output->mutable_data<float16>();

    conv_param.input = param.x->ZynqTensor();
    conv_param.output = param.output->ZynqTensor();
    conv_param.filter = param.filter->ZynqTensor();
    conv_param.filter->setDataType(zynqmp::FP32);
    conv_param.groups = param.groups;
    conv_param.strides = param.strides;
    conv_param.paddings = param.paddings;
    conv_param.dilations = param.dilations;
    fill_scale_bias_const(&conv_param);
    conv_param.bias()->copyFrom(param.bias->ZynqTensor());
    conv_param.relu.enabled = param.fuse_relu;

    dw_conv_pe_.init();
    dw_conv_pe_.apply();
  } else {
    zynqmp::ConvParam& conv_param = conv_pe_.param();
    param.output->mutable_data<float16>();

    conv_param.input = param.x->ZynqTensor();
    conv_param.output = param.output->ZynqTensor();
    conv_param.filter = param.filter->ZynqTensor();
    conv_param.filter->setDataType(zynqmp::FP32);
    conv_param.groups = param.groups;
    conv_param.strides = param.strides;
    conv_param.paddings = param.paddings;
    conv_param.dilations = param.dilations;
    fill_scale_bias_const(&conv_param);
    conv_param.bias()->copyFrom(param.bias->ZynqTensor());
    conv_param.relu.enabled = param.fuse_relu;

    conv_pe_.init();
    conv_pe_.apply();
  }
}

void ConvCompute::Run() {
  zynqmp::ConvParam& conv_param = pe_.param();
  pe_.dispatch();
  auto& param = this->Param<param_t>();
  if (param.groups == param.x->ZynqTensor()->shape().channel()) {
    dw_conv_pe_.dispatch();
    param.output->ZynqTensor()->saveToFile("dw", true);
  } else {
    zynqmp::ConvParam& conv_param = conv_pe_.param();
    conv_pe_.dispatch();
    param.output->ZynqTensor()->saveToFile("conv", true);
  }
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    conv2d, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::ConvCompute, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
