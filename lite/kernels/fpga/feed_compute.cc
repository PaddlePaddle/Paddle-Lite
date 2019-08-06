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

#include "lite/kernels/fpga/feed_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void FeedCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  // ====================================================
  zynqmp::InputParam& conv_param = pe_.param();
  Tensor& x = param.feed_list->at(param.col);

  param.out->mutable_data<float16>();
  conv_param.input = x.ZynqTensor();
  conv_param.output = param.out->ZynqTensor();
  pe_.init();
  pe_.apply();
}

void FeedCompute::Run() {
  auto& param = this->Param<param_t>();
  pe_.dispatch();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    feed, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::FeedCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
