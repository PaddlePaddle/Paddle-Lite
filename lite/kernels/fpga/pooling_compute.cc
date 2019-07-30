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

void PoolCompute::PrepareForRun() {
  zynqmp::PoolingParam& pool_param = pe_.param();
  auto& param = Param<operators::PoolParam>();

  input_.share_from_tensorlite(*param.x);
  output_.share_from_tensorlite(*param.output);
  pool_param.input = &input_;
  pool_param.output = &output_;
  pool_param.relu.enabled = false;

  LOG(ERROR) << input_;

  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  //   bool exclusive = param.exclusive;
  //   bool adaptive = param.adaptive;
  //   bool ceil_mode = param.ceil_mode;
  //   bool use_quantizer = param.use_quantizer;
  //   std::string& data_format = param.data_format;

  pool_param.type = param.pooling_type == "max" ? zynqmp::PoolingType::MAX
                                                : zynqmp::PoolingType::AVERAGE;
  pool_param.globalPooling = param.global_pooling;
  pool_param.kernelSize = param.ksize;
  pool_param.strides = param.strides;
  pool_param.paddings = param.paddings;

  pe_.init();
  pe_.apply();
  LOG(ERROR) << "init succes";
}

void PoolCompute::Run() {
  LOG(ERROR) << "Run";
  pe_.dispatch();
  LOG(ERROR) << output_;
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
