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

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void FeedCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  Tensor& x = param.feed_list->at(param.col);
  param.out->Resize(x.dims());

  auto in_type = x.ZynqTensor()->dataType();

  switch (in_type) {
    case zynqmp::FP32:
    case zynqmp::FP16:
      param.out->mutable_data<float16>();
      break;
    case zynqmp::INT32:
      param.out->mutable_data<int32_t>();
      break;
    case zynqmp::INT64:
      param.out->mutable_data<int64_t>();
      break;
    default:
      throw "type not supported!";
  }

  // ====================================================
  zynqmp::InputParam& feed_param = pe_.param();
  feed_param.input = x.ZynqTensor();
  feed_param.output = param.out->ZynqTensor();
  pe_.init();
  pe_.apply();
}

void FeedCompute::Run() {
  auto& param = this->Param<param_t>();
  Tensor& x = param.feed_list->at(param.col);
  pe_.param().input = x.ZynqTensor();
  pe_.dispatch();
  auto out_lod = param.out->mutable_lod();
  *out_lod = x.lod();

#ifdef FPGA_PRINT_TENSOR
  zynqmp::InputParam& feed_param = pe_.param();
  Debugger::get_instance().registerOutput("feed", feed_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    feed, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::FeedCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(feed,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::FeedCompute,
                     feed_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
