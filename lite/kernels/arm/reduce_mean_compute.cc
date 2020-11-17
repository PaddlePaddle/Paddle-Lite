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

#include "lite/kernels/arm/reduce_mean_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ReduceMeanCompute::Run() {
  auto& param = Param<operators::ReduceMeanParam>();
  const float* input = param.X->data<float>();
  auto x_dims = param.X->dims();
  int x_rank = x_dims.size();
  float* output = param.Out->mutable_data<float>();
  bool keep_dim = param.keep_dim;
  auto dim = param.dim;

  if (!dim.empty()) {
    for (int i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }

  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < x_dims.size(); ++j) {
    new_dims[j] = x_dims[j];
  }
  int n_in = new_dims[0];
  int c_in = new_dims[1];
  int h_in = new_dims[2];
  int w_in = new_dims[3];
  if (dim.size() == 0) {
    lite::arm::math::reduce_mean_all(input, output, n_in, c_in, h_in, w_in);
  } else if (dim.size() == 1) {
    switch (dim[0]) {
      case 0:
        lite::arm::math::reduce_mean_n(input, output, n_in, c_in, h_in, w_in);
        break;
      case 1:
        lite::arm::math::reduce_mean_c(input, output, n_in, c_in, h_in, w_in);
        break;
      case 2:
        lite::arm::math::reduce_mean_h(input, output, n_in, c_in, h_in, w_in);
        break;
      case 3:
        lite::arm::math::reduce_mean_w(input, output, n_in, c_in, h_in, w_in);
        break;
      default:
        LOG(FATAL) << "error!!!";
    }
  } else if (dim.size() == 2) {
    if (dim[0] == 0 && dim[1] == 1) {
      lite::arm::math::reduce_mean_nc(input, output, n_in, c_in, h_in, w_in);
    } else if (dim[0] == 1 && dim[1] == 2) {
      lite::arm::math::reduce_mean_ch(input, output, n_in, c_in, h_in, w_in);
    } else if (dim[0] == 2 && dim[1] == 3) {
      lite::arm::math::reduce_mean_hw(input, output, n_in, c_in, h_in, w_in);
    } else {
      LOG(FATAL) << "invalid dim!!";
    }
  } else {
    LOG(FATAL) << "dim's size over than 2, which is not supported now!!";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reduce_mean,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReduceMeanCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
