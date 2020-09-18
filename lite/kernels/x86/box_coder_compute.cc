// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/x86/box_coder_compute.h"
#include <string>
#include <vector>
#include "lite/backends/x86/math/box_coder.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

void BoxCoderCompute::Run() {
  auto& param = *param_.get_mutable<operators::BoxCoderParam>();
  // required inputs
  auto* prior_box = param.prior_box;    // M x 4 => M x [xmin, ymin, xmax, ymax]
  auto* target_box = param.target_box;  // encode_center_size => N x 4;
                                        // decode_center_size => N x M x 4
  // optional input
  auto* prior_box_var = param.prior_box_var;  // M x 4 or 4
  // output
  auto* output_box = param.proposals;  // N x M x 4
  // required attributes
  std::string code_type = param.code_type;
  bool normalized = param.box_normalized;
  // optional attributes
  std::vector<float> variance = param.variance;
  const int axis = param.axis;

  auto row = target_box->dims()[0];         // N
  auto col = prior_box->dims()[0];          // M
  if (code_type == "decode_center_size") {  // same as target_box
    col = target_box->dims()[1];
  }
  auto len = prior_box->dims()[1];      // 4
  output_box->Resize({row, col, len});  // N x M x 4
  auto* output = output_box->mutable_data<float>();

  const float* target_box_data = target_box->data<float>();
  const float* prior_box_data = prior_box->data<float>();
  const float* prior_box_var_data =
      prior_box_var ? prior_box_var->data<float>() : nullptr;

  if (code_type == "encode_center_size") {
    lite::x86::math::encode_center_size(row,
                                        col,
                                        len,
                                        target_box_data,
                                        prior_box_data,
                                        prior_box_var_data,
                                        normalized,
                                        variance,
                                        output);
  } else if (code_type == "decode_center_size") {
    int var_size = 0;
    if (prior_box_var) {
      var_size = 2;
    } else if (!(variance.empty())) {
      var_size = 1;
    }
    lite::x86::math::decode_center_size(axis,
                                        var_size,
                                        row,
                                        col,
                                        len,
                                        target_box_data,
                                        prior_box_data,
                                        prior_box_var_data,
                                        normalized,
                                        variance,
                                        output);
  } else {
    LOG(FATAL) << "box_coder don't support this code_type: " << code_type;
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_coder,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::BoxCoderCompute,
                     def)
    .BindInput("PriorBox", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("PriorBoxVar", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("TargetBox", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("OutputBox", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
