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

#include "lite/kernels/host/flatten_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void FlattenContiguousRangeCompute::Run() {
  auto& param = Param<operators::FlattenContiguousRangeParam>();
  auto x = param.x;
  auto out = param.out;
  auto out_dims = out->dims();
  auto out_lod = out->lod();
  out->CopyDataFrom(*x);
  out->Resize(out_dims);
  out->set_lod(out_lod);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(flatten_contiguous_range,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::FlattenContiguousRangeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();
