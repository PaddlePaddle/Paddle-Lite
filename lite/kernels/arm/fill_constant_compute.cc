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

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T>
class FillConstantCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::FillConstantParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<ARMContext>();

    auto data = param.Out->template mutable_data<T>();

    for (int i = 0; i < param.Out->numel(); i++) {
      data[i] = param.value;
    }

    // paddle::operators::math::set_constant(
    //    *context.arm_device_context(), &param.Out->raw_tensor(), param.value);
  }

  virtual ~FillConstantCompute() = default;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(fill_constant,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::FillConstantCompute<float>,
                     def)
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
