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

#include "lite/kernels/fpga/prior_box_compute.h"
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void PriorBoxCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  param.boxes->mutable_data<float16>();
  param.variances->mutable_data<float16>();
  // ====================================================
  zynqmp::PriorBoxParam& priobox_param = pe_.param();
  priobox_param.input = param.input->ZynqTensor();
  priobox_param.image = param.image->ZynqTensor();
  priobox_param.outputBoxes = param.boxes->ZynqTensor();
  priobox_param.outputVariances = param.variances->ZynqTensor();
  priobox_param.minSizes = param.min_sizes;
  priobox_param.maxSizes = param.max_sizes;
  priobox_param.aspectRatios = param.aspect_ratios;
  priobox_param.variances = param.variances_;
  // priobox_param.minMaxAspectRatiosOrder = param->MinMaxAspectRatiosOrder();
  priobox_param.flip = param.flip;
  priobox_param.clip = param.clip;
  priobox_param.stepW = param.step_w;
  priobox_param.stepH = param.step_h;
  priobox_param.offset = param.offset;

  pe_.init();
  pe_.apply();
}

void PriorBoxCompute::Run() { pe_.dispatch(); }

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(prior_box,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::PriorBoxCompute,
                     def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Image",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Boxes",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("Variances",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
