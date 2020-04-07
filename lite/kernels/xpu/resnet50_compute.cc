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

#include "lite/kernels/xpu/resnet50_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {

namespace operators {

bool ResNet50Op::CheckShape() const { return true; }

bool ResNet50Op::InferShapeImpl() const {
  auto input_shape = param_.input->dims();
  input_shape[1] = 2048;
  input_shape[2] = 1;
  input_shape[3] = 1;
  param_.output->Resize(input_shape);
  return true;
}

bool ResNet50Op::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = const_cast<lite::Tensor*>(
      &scope->FindVar(op_desc.Input("Input").front())->Get<lite::Tensor>());
  param_.output = scope->FindVar(op_desc.Output("Output").front())
                      ->GetMutable<lite::Tensor>();

  param_.filter.clear();
  for (auto& name : op_desc.Input("Filter")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.filter.push_back(t);
  }
  param_.bias.clear();
  for (auto& name : op_desc.Input("Bias")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.bias.push_back(t);
  }
  param_.max_filter.clear();
  for (auto& name : op_desc.Input("MaxFilter")) {
    auto t =
        const_cast<lite::Tensor*>(&scope->FindVar(name)->Get<lite::Tensor>());
    param_.max_filter.push_back(t);
  }
  return true;
}

}  // namespace operators

namespace kernels {
namespace xpu {

ResNet50Compute::ResNet50Compute() {
  set_op_type("ResNet50");
  set_alias("def");
}

void ResNet50Compute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  for (auto* filter : param.filter) {
    arg_filter_.push_back(reinterpret_cast<int16_t*>(filter->data<float>()));
  }
  for (auto* bias : param.bias) {
    arg_bias_.push_back(reinterpret_cast<float*>(bias->data<float>()));
  }
  for (auto* max_filter : param.max_filter) {
    arg_max_filter_.push_back(
        reinterpret_cast<float*>(max_filter->data<float>()));
  }
}

void ResNet50Compute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int batch_size = param.input->dims()[0];
  int r = xdnn::conv2d_int16_resnet<float, int16_t>(
      ctx.GetRawContext(),                             /* context */
      batch_size,                                      /* num */
      param.input->data<float>(),                      /* bottom */
      &arg_filter_[0],                                 /* weight_list */
      param.output->mutable_data<float>(TARGET(kXPU)), /* top */
      &arg_bias_[0],                                   /* bias_list */
      &arg_max_filter_[0] /* max_filter_list */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(ResNet50, paddle::lite::operators::ResNet50Op);

REGISTER_LITE_KERNEL(ResNet50,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ResNet50Compute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("MaxFilter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
