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

#include "lite/kernels/fpga/pooling_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void PoolCompute::PrepareForRun() {
  zynqmp::PoolingParam& pool_param = pe_.param();
  auto& param = Param<operators::PoolParam>();

  param.output->mutable_data<float16>();

  pool_param.input = param.x->ZynqTensor();
  pool_param.output = param.output->ZynqTensor();
  pool_param.relu.enabled = false;

  pool_param.type = param.pooling_type == "max" ? zynqmp::PoolingType::MAX
                                                : zynqmp::PoolingType::AVERAGE;
  pool_param.globalPooling = param.global_pooling;
  pool_param.kernelSize = param.ksize;
  pool_param.strides = param.strides;
  pool_param.paddings = param.paddings;
  pe_.init();
  pe_.apply();
}

void PoolCompute::Run() { 
  pe_.dispatch(); 
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pool2d, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::PoolCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
