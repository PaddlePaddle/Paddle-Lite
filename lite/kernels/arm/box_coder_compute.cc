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

#include "lite/kernels/arm/box_coder_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void EncodeCenterSize(const Tensor* target_box,
                      const Tensor* prior_box,
                      const Tensor* prior_box_var,
                      const bool normalized,
                      const std::vector<float> variance,
                      float* output) {
  int64_t row = target_box->dims()[0];
  int64_t col = prior_box->dims()[0];
  int64_t len = prior_box->dims()[1];
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      auto* target_box_data = target_box->data<float>();
      auto* prior_box_data = prior_box->data<float>();
      int64_t offset = i * col * len + j * len;
      float prior_box_width = prior_box_data[j * len + 2] -
                              prior_box_data[j * len] + (normalized == false);
      float prior_box_height = prior_box_data[j * len + 3] -
                               prior_box_data[j * len + 1] +
                               (normalized == false);
      float prior_box_center_x = prior_box_data[j * len] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[j * len + 1] + prior_box_height / 2;

      float target_box_center_x =
          (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
      float target_box_center_y =
          (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
      float target_box_width = target_box_data[i * len + 2] -
                               target_box_data[i * len] + (normalized == false);
      float target_box_height = target_box_data[i * len + 3] -
                                target_box_data[i * len + 1] +
                                (normalized == false);

      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] =
          std::log(std::fabs(target_box_width / prior_box_width));
      output[offset + 3] =
          std::log(std::fabs(target_box_height / prior_box_height));
    }
  }

  if (prior_box_var) {
    const float* prior_box_var_data = prior_box_var->data<float>();
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int k = 0; k < 4; ++k) {
          int64_t offset = i * col * len + j * len;
          int64_t prior_var_offset = j * len;
          output[offset + k] /= prior_box_var_data[prior_var_offset + k];
        }
      }
    }
  } else if (!(variance.empty())) {
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int k = 0; k < 4; ++k) {
          int64_t offset = i * col * len + j * len;
          output[offset + k] /= static_cast<float>(variance[k]);
        }
      }
    }
  }
}

template <int axis, int var_size>
void DecodeCenterSize(const Tensor* target_box,
                      const Tensor* prior_box,
                      const Tensor* prior_box_var,
                      const bool normalized,
                      std::vector<float> variance,
                      float* output) {
  int64_t row = target_box->dims()[0];
  int64_t col = target_box->dims()[1];
  int64_t len = target_box->dims()[2];

  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      auto* target_box_data = target_box->data<float>();
      auto* prior_box_data = prior_box->data<float>();

      float var_data[4] = {1., 1., 1., 1.};
      float* var_ptr = var_data;
      int64_t offset = i * col * len + j * len;
      int64_t prior_box_offset = axis == 0 ? j * len : i * len;

      float prior_box_width = prior_box_data[prior_box_offset + 2] -
                              prior_box_data[prior_box_offset] +
                              (normalized == false);
      float prior_box_height = prior_box_data[prior_box_offset + 3] -
                               prior_box_data[prior_box_offset + 1] +
                               (normalized == false);
      float prior_box_center_x =
          prior_box_data[prior_box_offset] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[prior_box_offset + 1] + prior_box_height / 2;

      float target_box_center_x = 0, target_box_center_y = 0;
      float target_box_width = 0, target_box_height = 0;
      int64_t prior_var_offset = axis == 0 ? j * len : i * len;
      if (var_size == 2) {
        std::memcpy(var_ptr,
                    prior_box_var->data<float>() + prior_var_offset,
                    4 * sizeof(float));
      } else if (var_size == 1) {
        var_ptr = reinterpret_cast<float*>(variance.data());
      }
      float box_var_x = *var_ptr;
      float box_var_y = *(var_ptr + 1);
      float box_var_w = *(var_ptr + 2);
      float box_var_h = *(var_ptr + 3);

      target_box_center_x =
          box_var_x * target_box_data[offset] * prior_box_width +
          prior_box_center_x;
      target_box_center_y =
          box_var_y * target_box_data[offset + 1] * prior_box_height +
          prior_box_center_y;
      target_box_width =
          std::exp(box_var_w * target_box_data[offset + 2]) * prior_box_width;
      target_box_height =
          std::exp(box_var_h * target_box_data[offset + 3]) * prior_box_height;

      output[offset] = target_box_center_x - target_box_width / 2;
      output[offset + 1] = target_box_center_y - target_box_height / 2;
      output[offset + 2] =
          target_box_center_x + target_box_width / 2 - (normalized == false);
      output[offset + 3] =
          target_box_center_y + target_box_height / 2 - (normalized == false);
    }
  }
}

void BoxCoderCompute::Run() {
  auto& param = Param<operators::BoxCoderParam>();
  auto* prior_box = param.prior_box;
  auto* prior_box_var = param.prior_box_var;
  auto* target_box = param.target_box;
  auto* output_box = param.proposals;
  std::vector<float> variance = param.variance;
  const int axis = param.axis;
  std::string code_type = param.code_type;
  bool normalized = param.box_normalized;

  auto row = target_box->dims()[0];
  auto col = prior_box->dims()[0];
  if (code_type == "decode_center_size") {
    col = target_box->dims()[1];
  }
  auto len = prior_box->dims()[1];
  output_box->Resize({row, col, len});
  auto* output = output_box->mutable_data<float>();

  if (code_type == "encode_center_size") {
    EncodeCenterSize(
        target_box, prior_box, prior_box_var, normalized, variance, output);
  } else if (code_type == "decode_center_size") {
    if (prior_box_var) {
      if (axis == 0) {
        DecodeCenterSize<0, 2>(
            target_box, prior_box, prior_box_var, normalized, variance, output);
      } else {
        DecodeCenterSize<1, 2>(
            target_box, prior_box, prior_box_var, normalized, variance, output);
      }
    } else if (!(variance.empty())) {
      if (axis == 0) {
        DecodeCenterSize<0, 1>(
            target_box, prior_box, prior_box_var, normalized, variance, output);
      } else {
        DecodeCenterSize<1, 1>(
            target_box, prior_box, prior_box_var, normalized, variance, output);
      }
    } else {
      if (axis == 0) {
        DecodeCenterSize<0, 0>(
            target_box, prior_box, prior_box_var, normalized, variance, output);
      } else {
        DecodeCenterSize<1, 0>(
            target_box, prior_box, prior_box_var, normalized, variance, output);
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_coder,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::BoxCoderCompute,
                     def)
    .BindInput("PriorBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("PriorBoxVar", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("TargetBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("OutputBox", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
