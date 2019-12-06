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

#include "lite/kernels/arm/reduce_prod_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <typename T, PrecisionType Ptype>
void ReduceProdCompute<T, Ptype>::Run() {
  auto& param = this->template Param<operators::ReduceParam>();
  auto* input = param.x->template data<T>();
  auto x_dims = param.x->dims();
  int x_rank = x_dims.size();
  auto* output = param.output->template mutable_data<T>();
  std::vector<int> dim = param.dim;
  bool keep_dim = param.keep_dim;
  bool reduce_all = param.reduce_all;

  if (!dim.empty()) {
    for (int i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }

  if (reduce_all) {
    lite::arm::math::reduce_prod_all(input, output, x_dims.production());
  } else {
    CHECK_EQ(x_rank, 4U);
    int n_in = x_dims[0];
    int c_in = x_dims[1];
    int h_in = x_dims[2];
    int w_in = x_dims[3];

    if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_prod_n(input, output, n_in, c_in, h_in, w_in);
          break;
        case 1:
          lite::arm::math::reduce_prod_c(input, output, n_in, c_in, h_in, w_in);
          break;
        case 2:
          lite::arm::math::reduce_prod_h(input, output, n_in, c_in, h_in, w_in);
          break;
        case 3:
          lite::arm::math::reduce_prod_w(input, output, n_in, c_in, h_in, w_in);
          break;
        default:
          LOG(FATAL) << "dim[0] should be less than 4.";
      }
    } else if (dim.size() == 2) {
      if (dim[0] == 0 && dim[1] == 1) {
        lite::arm::math::reduce_prod_nc(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 1 && dim[1] == 2) {
        lite::arm::math::reduce_prod_ch(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 2 && dim[1] == 3) {
        lite::arm::math::reduce_prod_hw(input, output, n_in, c_in, h_in, w_in);
      } else {
        LOG(FATAL)
            << "Only support the values of the dim are 0,1 1,2 or 2,3 for now.";
      }
    } else {
      LOG(FATAL) << "dim's size over than 2, which is not supported now!!";
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using reduce_prob_arm_int32 =
    paddle::lite::kernels::arm::ReduceProdCompute<int, PRECISION(kInt32)>;
using reduce_prob_arm_float =
    paddle::lite::kernels::arm::ReduceProdCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    reduce_prod, kARM, kInt32, kNCHW, reduce_prob_arm_int32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    reduce_prod, kARM, kFloat, kNCHW, reduce_prob_arm_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
