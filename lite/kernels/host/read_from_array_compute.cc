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

#include "lite/kernels/host/read_from_array_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void ReadFromArrayCompute::Run() {
  auto& param = this->Param<param_t>();
  int num = param.X->size();
  CHECK_EQ(param.I->dims().production(), 1)
      << "The size of tensor I should be 1";
  auto idx = param.I->data<int64_t>()[0];
  CHECK_LT(idx, num)
      << "The value of tensor I should be less than the size of TensorArray X";
  param.Out->Resize((*param.X)[idx].dims());
  param.Out->CopyDataFrom((*param.X)[idx]);
  auto out_lod = param.Out->mutable_lod();
  *out_lod = (*param.X)[idx].lod();
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(read_from_array,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::ReadFromArrayCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorListTy(
                   TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindInput("I",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();
