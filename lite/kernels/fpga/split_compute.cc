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

#include "lite/kernels/fpga/split_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void SplitCompute::PrepareForRun() {
  auto& param = Param<operators::SplitParam>();
  zynqmp::SplitParam& split_param = pe_.param();
  split_param.input = param.x->ZynqTensor();
  auto& dout = param.output;
  for (int i = 0; i < dout.size(); i++) {
    dout[i]->mutable_data<zynqmp::float16>();
    split_param.outputs.push_back(dout[i]->ZynqTensor());
  }

  pe_.init();
  pe_.apply();
}

void SplitCompute::Run() {
  zynqmp::SplitParam& split_param = pe_.param();
  pe_.dispatch();

#ifdef FPGA_PRINT_TENSOR
  auto& dout = param.output;
  for (int i = 0; i < dout.size(); i++) {
    Debugger::get_instance().registerOutput("split", split_param.outputs[0]);
  }

#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    split, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::SplitCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SectionsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
