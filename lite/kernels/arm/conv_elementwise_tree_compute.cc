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
#include "lite/kernels/arm/conv_gemmlike.h"
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
  auto has_elt_act = param.has_elt_act;
  conv_impl_ = new GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>;
  if (has_elt_act) {
    elt_impl_ = new ElementwiseAddActivationCompute<float, PRECISION(kFloat)>;
  } else {
    elt_impl_ = new ElementwiseAddCompute<float, PRECISION(kFloat)>;
  }
  // malloc space
  tmp_output_.set_precision(PRECISION(kFloat));
  auto out_dims = param.output->dims();
  tmp_output_.Resize(out_dims);
  param.conv_param.output = &tmp_output_;
  param.elt_param.X = &tmp_output_;
  operators::ConvParam& conv_param = param.conv_param;
  auto& elt_param = param.elt_param;

  conv_impl_->SetContext(std::move(this->ctx_));
  conv_impl_->SetParam(conv_param);
  conv_impl_->PrepareForRun();
  if (has_elt_act) {
    elt_impl_->SetParam(elt_param);
  } else {
    operators::ElementwiseParam elt_param1;
    elt_param1.X = elt_param.X;
    elt_param1.Y = elt_param.Y;
    elt_param1.Out = elt_param.Out;
    elt_param1.axis = elt_param.axis;
    elt_param1.fuse_scale = elt_param.fuse_scale;
    elt_param1.alpha = elt_param.alpha;
    elt_param1.bias = elt_param.bias;
    elt_impl_->SetParam(elt_param1);
  }
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
