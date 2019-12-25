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

#include "lite/kernels/fpga/concat_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

#include "lite/backends/fpga/KD/debugger.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void ConcatCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  param.output->mutable_data<float16>();

  // ====================================================
  zynqmp::ConcatParam& concat_param = pe_.param();
  for (auto t : param.x) {
    concat_param.inputs.push_back(t->ZynqTensor());
  }
  concat_param.output = param.output->ZynqTensor();
  concat_param.axis = param.axis;
  pe_.init();
  pe_.apply();
}

void ConcatCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::ConcatParam& concat_param = pe_.param();
  Debugger::get_instance().registerOutput("concat", concat_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ConcatCompute,
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
