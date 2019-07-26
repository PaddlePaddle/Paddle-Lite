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

#include "lite/kernels/fpga/conv_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void ConvCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  auto& ctx = this->ctx_->template As<ARMContext>();

  int win = x_dims[3];  // nchw
  int hin = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];
  int pad = param.paddings[0];
  int stride = param.strides[0];

  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  // ====================================================
  zynqmp::ConvParam& conv_param = pe_.param();
  // auto& param = Param<operators::ConvParam>();

  input_.share_from_tensorlite(*param.x);
  output_.share_from_tensorlite(*param.output);
  filter_.share_from_tensorlite(*param.filter);

  conv_param.input = &input_;
  conv_param.output = &output_;
  conv_param.filter = &filter_;
  conv_param.groups = param.groups;
  conv_param.strides = param.strides;
  conv_param.paddings = param.paddings;
  conv_param.dilations = param.dilations;

  // std::vector<int> kernelSize;
  pe_.init();
  pe_.apply();
}

void ConvCompute::Run() {
  auto& param = this->Param<param_t>();
  pe_.dispatch();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    conv2d, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::ConvCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .Finalize();
