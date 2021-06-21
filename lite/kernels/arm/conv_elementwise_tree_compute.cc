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

#include "lite/kernels/arm/conv_elementwise_tree_compute.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/arm/conv_compute.h"
#include "lite/kernels/arm/elementwise_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <>
void ConvElementwiseTreeCompute<PRECISION(kFloat),
                                PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  conv_impl_ = new ConvCompute<PRECISION(kFloat), PRECISION(kFloat)>;
  elt_impl_ = new ElementwiseAddCompute<float, PRECISION(kFloat)>;
  Tensor* tmp_output;
  auto out_dims = param.output->dims();
  tmp_output->Resize(out_dims);
  param.conv_param.output = tmp_output;
  param.elt_param.X = tmp_output;
  auto& conv_param = param.conv_param;
  auto& elt_param = param.elt_param;

  conv_impl_->SetContext(std::move(this->ctx_));
  conv_impl_->SetParam(conv_param);
  conv_impl_->PrepareForRun();
  elt_impl_->SetContext(std::move(this->ctx_));
  elt_impl_->SetParam(elt_param);
  elt_impl_->PrepareForRun();
  is_first_epoch_ = false;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::
    ConvElementwiseTreeCompute<PRECISION(kFloat), PRECISION(kFloat)>
        ConvEltTreeFp32;

REGISTER_LITE_KERNEL(
    conv_elementwise_tree, kARM, kFloat, kNCHW, ConvEltTreeFp32, def)
    .BindInput("Input0",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Input1",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Prelu_alpha",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv_elementwise_tree", 1)
    .Finalize();
