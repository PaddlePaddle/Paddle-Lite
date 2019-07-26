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

#include "lite/kernels/fpga/elementwise_compute.h"
#include <string>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void ElementwiseAddCompute::PrepareForRun() {
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  auto& param = Param<operators::ElementwiseParam>();
  input_x_.share_from_tensorlite(*param.X);
  input_y_.share_from_tensorlite(*param.Y);
  output_.share_from_tensorlite(*param.Out);
  ew_param.inputs = {&input_x_, &input_y_};
  ew_param.output = &output_;
  ew_param.axis = param.axis;
  ew_param.relu.enabled = false;

  pe_.init();
  pe_.apply();
}
void ElementwiseAddCompute::Run() {
  input_x_.flush();
  input_y_.flush();

  LOG(ERROR) << "input x_";
  LOG(ERROR) << input_x_;
  LOG(ERROR) << input_y_;

  pe_.dispatch();
  LOG(ERROR) << "after dispatch";
  output_.invalidate();
}

void ElementwiseAddActivationCompute::PrepareForRun() {
  zynqmp::ElementwiseAddParam& ew_param = pe_.param();
  auto& param = Param<operators::FusionElementwiseActivationParam>();
  if (param.act_type != "relu") {
    LOG(FATAL) << "unsupported Activation type: " << param.act_type;
  }
  input_x_.share_from_tensorlite(*param.X);
  input_y_.share_from_tensorlite(*param.Y);
  output_.share_from_tensorlite(*param.Out);
  ew_param.inputs = {&input_x_, &input_y_};
  ew_param.output = &output_;
  ew_param.axis = param.axis;
  ew_param.relu.enabled = true;
  pe_.init();
  pe_.apply();
}
void ElementwiseAddActivationCompute::Run() {
  input_x_.flush();
  input_y_.flush();
  pe_.dispatch();
  output_.invalidate();
  auto& param = Param<operators::FusionElementwiseActivationParam>();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kFPGA,
    kFP16,
    kNHWC,
    paddle::lite::kernels::fpga::ElementwiseAddActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kFPGA))})
    .Finalize();
