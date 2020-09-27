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

#include "lite/kernels/fpga/softmax_compute.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/fpga/KD/debugger.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void SoftmaxCompute::PrepareForRun() {
  zynqmp::SoftmaxParam& softmax_param = pe_.param();
  auto& param = Param<operators::SoftmaxParam>();

  param.output->mutable_data<float>();
  softmax_param.input = param.x->ZynqTensor();
  softmax_param.output = param.output->ZynqTensor();
  pe_.init();
  pe_.apply();
}

void SoftmaxCompute::Run() {
  zynqmp::SoftmaxParam& softmax_param = pe_.param();
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  Debugger::get_instance().registerOutput("softmax", softmax_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(softmax,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::SoftmaxCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
