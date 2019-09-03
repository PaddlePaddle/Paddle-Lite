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

#include "lite/kernels/arm/crop_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void CropCompute::crop_fun(const lite::Tensor* input, lite::Tensor* output) {
  auto input_dims = input->dims();
  int num = input_dims[0];
  int in_c = input_dims[1];
  int in_h = input_dims[2];
  int in_w = input_dims[3];
  const float* ptr_in = input->data<float>();
  float* ptr_out = output->mutable_data<float>();
  for (int i = 0; i < num; ++i) {
    int offset_n = i * in_c * in_h * in_w;
    for (int j = c_off; j < c_end; ++j) {
      int offset_c = offset_n + j * in_h * in_w;
      for (int k = h_off; k < h_end; ++k) {
        int offset_h = offset_c + k * in_w;
        for (int l = w_off; l < w_end; ++l) {
          ptr_out[0] = ptr_in[offset_h + l];
          ptr_out++;
        }
      }
    }
  }
}
void CropCompute::Run() {
  auto& param = Param<operators::CropParam>();
  const lite::Tensor* inputs = param.X;
  auto* out = param.Out;
  offsets_ = param.offsets;
  shape_ = param.shape;

  c_off = param.offsets[1];
  h_off = param.offsets[2];
  w_off = param.offsets[3];
  c_end = shape_[1] + c_off;
  h_end = shape_[2] + h_off;
  w_end = shape_[3] + w_off;
  crop_fun(inputs, out);

  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    crop, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::CropCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
