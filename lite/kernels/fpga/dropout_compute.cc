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

#include "lite/kernels/fpga/dropout_compute.h"
#include <string>

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/backends/fpga/KD/float16.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void DropoutCompute::PrepareForRun() {
  auto& param = Param<operators::DropoutParam>();
  param.output->mutable_data<float16>();

  zynqmp::ScaleParam& scale_param = pe_.param();
  scale_param.input = param.x->ZynqTensor();
  scale_param.output = param.output->ZynqTensor();

  int channel = scale_param.input->shape().channel();
  zynqmp::Tensor* scale = new zynqmp::Tensor();
  zynqmp::Tensor* bias = new zynqmp::Tensor();
  zynqmp::Shape shape(zynqmp::N, {channel});
  float* scale_data = scale->mutableData<float>(zynqmp::FP32, shape);
  float* bias_data = bias->mutableData<float>(zynqmp::FP32, shape);

  float scale_value = 1 - param.dropout_prob;
  for (int i = 0; i < channel; ++i) {
    scale_data[i] = scale_value;
    bias_data[i] = 0.0f;
  }
  scale->flush();
  bias->flush();

  scale_param.bias = bias;
  scale_param.scale = scale;

  pe_.init();
  pe_.apply();
}

void DropoutCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::ScaleParam& scale_param = pe_.param();
  Debugger::get_instance().registerOutput("dropout", scale_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dropout,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::DropoutCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
