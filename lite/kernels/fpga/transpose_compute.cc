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

#include "lite/kernels/fpga/transpose_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

// Transpose
void TransposeCompute::Run() {}

// Transpose2
void Transpose2Compute::Run() {}

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
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
