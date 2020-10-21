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

#include "lite/kernels/fpga/cast_compute.h"
#include <algorithm>
// #include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

template <class in_type, class out_type>
out_type TransOp(in_type in) {
  return static_cast<out_type>(in);
}

void CastCompute::PrepareForRun() {}

void CastCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::CastParam>();

  auto input_dims = param.X->dims();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cast, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::CastCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
