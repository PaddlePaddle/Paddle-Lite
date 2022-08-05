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

#include "lite/kernels/arm/calib_inplace_compute.h"

#include <vector>

#include "lite/backends/arm/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#ifdef ENABLE_ARM_FP16
template <DataLayoutType DLType>
void CalibInplaceComputeFp16ToFp32<DLType>::Run() {
  std::cout << "in calib In 1" << std::endl;
  auto& param = this->template Param<operators::CalibInplaceParam>();

  std::cout << "in calib In 2" << std::endl;
  lite::Tensor tmp;
  tmp.CopyDataFrom(*param.input);

  std::cout << "in calib In 3" << std::endl;
  auto din = tmp.template data<float16_t>();

  std::cout << "in calib In 4" << std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << "in:" << din[i] << std::endl;
  }

  auto* dout = param.output->template mutable_data<float>();

  lite::arm::math::fp16::fp16_to_fp32(din, dout, param.input->numel());

  for (int i = 0; i < 10; i++) {
    std::cout << "out:" << dout[i] << std::endl;
  }
}
template <DataLayoutType DLType>
void CalibInplaceComputeFp32ToFp16<DLType>::Run() {
  std::cout << "in calib In 1" << std::endl;
  auto& param = this->template Param<operators::CalibInplaceParam>();

  std::cout << "in calib In 2" << std::endl;
  lite::Tensor tmp;
  tmp.CopyDataFrom(*param.input);

  std::cout << "in calib In 3" << std::endl;
  auto* din = tmp.template data<float>();
  std::cout << "in calib In 4" << std::endl;

  auto* dout = param.output->template mutable_data<float16_t>();
  std::cout << "in calib In 5" << std::endl;

  lite::arm::math::fp16::fp32_to_fp16(din, dout, param.input->numel());
  std::cout << "in calib In 6" << std::endl;
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(calib_inplace,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::CalibInplaceComputeFp16ToFp32<
                         DATALAYOUT(kNCHW)>,
                     fp16_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_inplace,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::CalibInplaceComputeFp32ToFp16<
                         DATALAYOUT(kNCHW)>,
                     fp32_to_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();

#endif  // ENABLE_ARM_FP16
