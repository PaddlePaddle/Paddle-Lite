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

#include <string>
#include <vector>

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/kernels/fpga/prior_box_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

void PriorBoxCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  bool is_flip = param.flip;
  bool is_clip = param.clip;
  std::vector<float> min_size = param.min_sizes;
  std::vector<float> max_size = param.max_sizes;
  std::vector<float> aspect_ratio = param.aspect_ratios;
  std::vector<float> variance = param.variances_;
  int img_w = param.img_w;
  int img_h = param.img_h;
  float step_w = param.step_w;
  float step_h = param.step_h;
  float offset = param.offset;
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratio, is_flip, &aspect_ratios_vec);
  int prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();
  std::vector<std::string> order = param.order;
  bool min_max_aspect_ratios_order = param.min_max_aspect_ratios_order;

  int win1 = param.input->dims()[3];
  int hin1 = param.input->dims()[2];

  DDim shape_out({hin1, win1, prior_num, 4});
  param.boxes->Resize(shape_out);
  param.variances->Resize(shape_out);

  param.boxes->mutable_data<float>();
  param.variances->mutable_data<float>();

  zynqmp::PriorBoxParam& priobox_param = pe_.param();
  priobox_param.input = param.input->ZynqTensor();
  priobox_param.image = param.image->ZynqTensor();
  priobox_param.outputBoxes = param.boxes->ZynqTensor();
  priobox_param.outputVariances = param.variances->ZynqTensor();
  priobox_param.minSizes = param.min_sizes;
  priobox_param.maxSizes = param.max_sizes;
  priobox_param.aspectRatios = param.aspect_ratios;
  priobox_param.variances = param.variances_;
  priobox_param.minMaxAspectRatiosOrder = min_max_aspect_ratios_order;
  priobox_param.flip = param.flip;
  priobox_param.clip = param.clip;
  priobox_param.stepW = param.step_w;
  priobox_param.stepH = param.step_h;
  priobox_param.offset = param.offset;

  pe_.init();
  pe_.apply();
}

void PriorBoxCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::PriorBoxParam& priobox_param = pe_.param();
  Debugger::get_instance().registerOutput("pb_boxes",
                                          priobox_param.outputBoxes);
  Debugger::get_instance().registerOutput("pb_variances",
                                          priobox_param.outputVariances);
#endif
}

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
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Variances", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
