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

#include "lite/kernels/arm/write_to_array_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void WriteToArrayCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->template Param<operators::WriteToArrayParam>();
  CHECK_EQ(param.I->numel(), 1) << "input2 should have only one element";

  int id = param.I->data<int64_t>()[0];
  if (param.Out->size() < id + 1) {
    param.Out->resize(id + 1);
  }
  param.Out->at(id).CopyDataFrom(*param.X);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(write_to_array,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::WriteToArrayCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindInput("I", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
