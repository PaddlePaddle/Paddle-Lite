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

#include "lite/kernels/fpga/scale_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void ScaleCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  param.output->mutable_data<float16>();

  zynqmp::ScaleParam& scale_param = pe_.param();

  scale_param.input = param.x->ZynqTensor();
  scale_param.output = param.output->ZynqTensor();

  int channel = scale_param.input->shape().channel();
  zynqmp::Tensor* scale = &scale_;
  zynqmp::Tensor* bias = &bias_;
  zynqmp::Shape shape(zynqmp::N, {channel});
  float* scale_data = scale->mutableData<float>(zynqmp::FP32, shape);
  float* bias_data = bias->mutableData<float>(zynqmp::FP32, shape);

  float scale_value = param.scale;
  float bias_value = param.bias_after_scale ? param.bias : 0;

  for (int i = 0; i < channel; ++i) {
    scale_data[i] = scale_value;
    bias_data[i] = bias_value;
  }
  scale_param.scale = scale;
  scale_param.bias = bias;

  pe_.init();
  pe_.apply();
}

void ScaleCompute::Run() { pe_.dispatch(); }

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    scale, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::ScaleCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
