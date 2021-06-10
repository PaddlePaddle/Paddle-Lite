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
#include "lite/kernels/fpga/slice_compute.h"
#include <algorithm>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void SliceCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  auto in_type = param.X->ZynqTensor()->dataType();
  if (in_type == zynqmp::FP32 || in_type == zynqmp::FP16) {
    param.Out->mutable_data<float16>();
  }
  if (in_type == zynqmp::INT32) {
    param.Out->mutable_data<int32_t>();
  }

  zynqmp::SliceParam& slice_param = pe_.param();

  slice_param.input = param.X->ZynqTensor();
  slice_param.output = param.Out->ZynqTensor();
  slice_param.axes = param.axes;
  slice_param.starts = param.starts;
  slice_param.ends = param.ends;
  slice_param.decrease_axis = param.decrease_axis;
  slice_param.infer_flags = param.infer_flags;

  pe_.init();
  pe_.apply();
}

void SliceCompute::Run() { pe_.dispatch(); }

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    slice, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::SliceCompute, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("StartsTensor", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("EndsTensor", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("StartsTensorList", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("EndsTensorList", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
