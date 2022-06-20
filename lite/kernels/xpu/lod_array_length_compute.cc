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

#include "lite/kernels/xpu/lod_array_length_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void LoDArrayLengthCompute::Run() {
  auto& param = this->template Param<operators::LoDArrayLengthParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int64_t array_length = param.x->size();
  int r =
      xdnn::constant<int64_t>(ctx.GetRawContext(),
                              param.out->mutable_data<int64_t>(TARGET(kXPU)),
                              1,
                              array_length);
  CHECK_EQ(r, 0);
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lod_array_length,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::LoDArrayLengthCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorListTy(TARGET(kXPU),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
