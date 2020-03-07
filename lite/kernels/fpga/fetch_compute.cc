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
#include "lite/kernels/fpga/fetch_compute.h"
#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void FetchCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  // ====================================================
  zynqmp::OutputParam& fetch_param = pe_.param();
  auto fetch_list = param.fetch_list;
  if (fetch_list->size() <= static_cast<size_t>(param.col)) {
    fetch_list->resize(param.col + 1);
  }
  Tensor& out = param.fetch_list->at(param.col);
  out.Resize(param.input->dims());
  out.mutable_data<float>();

  fetch_param.input = param.input->ZynqTensor();
  fetch_param.output = out.ZynqTensor();

  pe_.init();
  pe_.apply();
}

void FetchCompute::Run() {
  auto& param = this->Param<param_t>();
  auto fetch_list = param.fetch_list;
  if (fetch_list->size() <= static_cast<size_t>(param.col)) {
    fetch_list->resize(param.col + 1);
  }
  Tensor& out = param.fetch_list->at(param.col);
  out.Resize(param.input->dims());
  pe_.dispatch();

#ifdef FPGA_PRINT_TENSOR
  zynqmp::OutputParam& fetch_param = pe_.param();
  Debugger::get_instance().registerOutput("fetch", fetch_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fetch,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::FetchCompute,
                     fpga_host)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(fetch,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::FetchCompute,
                     host_host)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
