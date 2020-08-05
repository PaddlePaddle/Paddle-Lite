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

#include "lite/kernels/arm/reduce_max_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ReduceMaxCompute::Run() {
  auto& param = Param<operators::ReduceMaxParam>();
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

  if (x_dims.size() == 3) {
    if (dim.size() == 0 || dim.size() == 3) {
      lite::arm::math::reduce_all_of_three(
          input, output, x_dims[0], x_dims[1], x_dims[2]);
    } else if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_first_of_three(
              input, output, x_dims[0], x_dims[1], x_dims[2]);
          break;
        case 1:
          lite::arm::math::reduce_second_of_three(
              input, output, x_dims[0], x_dims[1], x_dims[2]);
          break;

        case 2:
          lite::arm::math::reduce_third_of_three(
              input, output, x_dims[0], x_dims[1], x_dims[2]);
          break;
        default:
          LOG(FATAL) << "error!!!";
      }
    } else if (dim.size() == 2) {
      LOG(FATAL) << "Will support later!!";
    } else {
      LOG(FATAL) << "dim size should not larger than 3!!!";
    }
  } else if (x_dims.size() == 4) {
    int n_in = x_dims[0];
    int c_in = x_dims[1];
    int h_in = x_dims[2];
    int w_in = x_dims[3];

    if (dim.size() == 0) {
      lite::arm::math::reduce_all(input, output, n_in, c_in, h_in, w_in);
    } else if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_n(input, output, n_in, c_in, h_in, w_in);
          break;
        case 1:
          lite::arm::math::reduce_c(input, output, n_in, c_in, h_in, w_in);
          break;
        case 2:
          lite::arm::math::reduce_h(input, output, n_in, c_in, h_in, w_in);
          break;
        case 3:
          lite::arm::math::reduce_w(input, output, n_in, c_in, h_in, w_in);
          break;
        default:
          LOG(FATAL) << "error!!!";
      }
    } else if (dim.size() == 2) {
      if (dim[0] == 0 && dim[1] == 1) {
        lite::arm::math::reduce_nc(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 1 && dim[1] == 2) {
        lite::arm::math::reduce_ch(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 2 && dim[1] == 3) {
        lite::arm::math::reduce_hw(input, output, n_in, c_in, h_in, w_in);
      } else {
        LOG(FATAL) << "invalid dim!!";
      }
    } else {
      LOG(FATAL) << "dim's size over than 2, which is not supported now!!";
    }
  } else {
    LOG(FATAL) << "only support input with 3&4 dimensions now!!";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reduce_max,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReduceMaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
