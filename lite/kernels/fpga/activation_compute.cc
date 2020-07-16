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

#include "lite/kernels/fpga/activation_compute.h"
#include "lite/backends/fpga/KD/float16.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void ReluCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto output_data = param.Out->mutable_data<float16>();
  zynqmp::InputParam& relu_param = pe_.param();

  relu_param.input = param.X->ZynqTensor();
  relu_param.output = param.Out->ZynqTensor();
  pe_.init();
  pe_.apply();
}

void ReluCompute::Run() { pe_.dispatch(); }

void SigmoidCompute::Run() {
  // TODO(chonwhite) use fpga and arm implementation;
  auto& param = this->Param<param_t>();
  auto output_data = param.Out->mutable_data<float16>();
  int numel = param.Out->numel();

  float16* in_data = param.X->ZynqTensor()->data<float16>();
  float16* out_data = param.Out->ZynqTensor()->data<float16>();
  param.X->ZynqTensor()->syncToCPU();
  float max = 0.0f;
  for (int i = 0; i < numel; i++) {
    /* code */
    float value = zynqmp::half_to_float(in_data[i]);
    value = 1 / (1 + exp(-value));
    out_data[i] = zynqmp::float_to_half(value);
    max = std::max(std::abs(value), max);
  }
  param.Out->ZynqTensor()->scale()[0] = max / 127.0;
  param.Out->ZynqTensor()->scale()[1] = 127.0 / max;
  param.Out->ZynqTensor()->flush();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    relu, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::ReluCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(sigmoid,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::SigmoidCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
