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

#include "lite/kernels/nnadapter/io_copy_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

void IoCopyHostToDeviceCompute::Run() {
  auto& param = this->Param<param_t>();
  if (param.x != nullptr) {
    IoCopyHostToDevice(param.x, param.y);
  }
  if (param.x_array != nullptr) {
    for (size_t i = 0; i < param.x_array->size(); i++) {
      IoCopyHostToDevice(&(param.x_array->at(i)), &(param.y_array->at(i)));
    }
  }
}

void IoCopyDeviceToHostCompute::Run() {
  auto& param = this->Param<param_t>();
  if (param.x != nullptr) {
    IoCopyDeviceToHost(param.x, param.y);
  }
  if (param.x_array != nullptr) {
    for (size_t i = 0; i < param.x_array->size(); i++) {
      IoCopyDeviceToHost(&(param.x_array->at(i)), &(param.y_array->at(i)));
    }
  }
}

void IoCopyHostToDevice(const Tensor* x, Tensor* y) {}

void IoCopyDeviceToHost(const Tensor* x, Tensor* y) {}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    io_copy,
    kNNAdapter,
    kAny,
    kAny,
    paddle::lite::kernels::nnadapter::IoCopyHostToDeviceCompute,
    host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kHost),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kNNAdapter),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kNNAdapter),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    io_copy,
    kNNAdapter,
    kAny,
    kAny,
    paddle::lite::kernels::nnadapter::IoCopyDeviceToHostCompute,
    device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kNNAdapter),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kNNAdapter),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kHost),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    io_copy_once,
    kNNAdapter,
    kAny,
    kAny,
    paddle::lite::kernels::nnadapter::IoCopyHostToDeviceCompute,
    host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kHost),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kNNAdapter),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kNNAdapter),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    io_copy_once,
    kNNAdapter,
    kAny,
    kAny,
    paddle::lite::kernels::nnadapter::IoCopyDeviceToHostCompute,
    device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kNNAdapter),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("InputArray",
               {LiteType::GetTensorListTy(TARGET(kNNAdapter),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .BindOutput("OutArray",
                {LiteType::GetTensorListTy(TARGET(kHost),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();
