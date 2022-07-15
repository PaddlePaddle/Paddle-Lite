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

#include "lite/kernels/xpu/read_from_array_compute.h"
#include <algorithm>
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ReadFromArrayCompute::Run() {
  auto& param = this->template Param<operators::ReadFromArrayParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  CHECK_EQ(param.I->numel(), 1) << "I should have only one element";
  int64_t id;
  TargetWrapperXPU::MemcpySync(
      &id, param.I->raw_data(), sizeof(int64_t), IoDirection::DtoH);
  CHECK_LT(static_cast<size_t>(id), param.X->size()) << "id is not valid";
  const auto& elem = (*param.X)[id];
  param.Out->Resize(elem.dims());
  param.Out->set_lod(elem.lod());
  param.Out->set_precision(elem.precision());
  param.Out->mutable_data(TARGET(kXPU), elem.memory_size());
  int r = xdnn::copy<int8_t>(ctx.GetRawContext(),
                             elem.data<int8_t>(),
                             static_cast<int8_t*>(param.Out->raw_data()),
                             elem.memory_size());
  CHECK_EQ(r, 0) << " read from array failed";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(read_from_array,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::ReadFromArrayCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorListTy(TARGET(kXPU),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindInput("FakeAssociatedX",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("I",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
