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

#include "lite/kernels/host/unbind_compute.h"
#include <vector>
#include "lite/backends/host/math/unbind.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void UnbindCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::UnbindParam>();
  auto& dout = param.output;
  for (auto out : dout) {
    out->set_lod(param.x->lod());
  }
  lite::host::math::unbind<T>(param.x, dout, param.axis);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using unbind_float =
    paddle::lite::kernels::host::UnbindCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unbind, kHost, kFloat, kNCHW, unbind_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

using unbind_int64 =
    paddle::lite::kernels::host::UnbindCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unbind, kHost, kFloat, kNCHW, unbind_int64, def_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
