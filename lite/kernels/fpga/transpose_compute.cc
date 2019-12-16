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

#include <string>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/kernels/fpga/transpose_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

// Transpose
void TransposeCompute::Run() {
  auto& param = this->Param<param_t>();
  // param.output->mutable_data<float16>();
}

// Transpose2
void Transpose2Compute::Run() {
  auto& param = this->Param<param_t>();
  param.output->mutable_data<float>();
  param.x->ZynqTensor()->invalidate();
  param.x->ZynqTensor()->unalignImage();
  if (param.x->dims().size() != 4) {
    // TransposeCompute<float>(param);
    // auto out = param.Out();
    // auto out_data = out->data<half>();

    //   int num = input_x_dims[1];
    // int channel = input_x_dims[2];

    // int index = 0;
    // for (int n = 0; n < num; n++) {
    //   for (int c = 0; c < channel; c++) {
    //     out_data[c * num + n] = input_x_data[n * channel + c];
    //     index++;
    //   }
    // }
  } else {
    param.x->ZynqTensor()->saveToFile("tx", true);
    param.output->ZynqTensor()->copyFrom(param.x->ZynqTensor());
    param.output->ZynqTensor()->saveToFile("to", true);
  }
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// Transpose
REGISTER_LITE_KERNEL(transpose,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransposeCompute,
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

// Transpose2
REGISTER_LITE_KERNEL(transpose2,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::Transpose2Compute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
