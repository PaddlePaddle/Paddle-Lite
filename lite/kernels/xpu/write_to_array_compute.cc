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

#include "lite/kernels/xpu/write_to_array_compute.h"
#include <algorithm>
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void WriteToArrayCompute::Run() {
  auto& param = this->template Param<operators::WriteToArrayParam>();
  CHECK_EQ(param.I->numel(), 1) << "input2 should have only one element";
  auto& ctx = this->ctx_->template As<XPUContext>();

  int64_t id;
  TargetWrapperXPU::MemcpySync(
      &id, param.I->raw_data(), sizeof(int64_t), IoDirection::DtoH);
  if (param.Out->size() < id + 1) {
    param.Out->resize(id + 1);
  }

  auto& elem = param.Out->at(id);
  elem.Resize(param.X->dims());
  elem.set_lod(param.X->lod());
  elem.set_precision(param.X->precision());
  if (elem.numel() > 0) {
    elem.mutable_data(TARGET(kXPU), param.X->memory_size());
    int r = xdnn::copy<int8_t>(ctx.GetRawContext(),
                               param.X->data<int8_t>(),
                               static_cast<int8_t*>(elem.raw_data()),
                               param.X->memory_size());
    CHECK_EQ(r, 0) << " write to array failed";
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(write_to_array,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::WriteToArrayCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("I",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(TARGET(kXPU),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .BindOutput("FakeAssociatedOut",
                {LiteType::GetTensorListTy(TARGET(kXPU),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();
