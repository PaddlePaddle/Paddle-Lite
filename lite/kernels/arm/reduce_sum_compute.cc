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

#include "lite/kernels/arm/reduce_sum_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ReduceSumCompute::Run() {
  auto& param = this->template Param<operators::ReduceParam>();
  auto* input = param.x->template data<float>();
  auto x_dims = param.x->dims();
  int x_rank = x_dims.size();
  auto* output = param.output->template mutable_data<float>();
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
    lite::arm::math::reduce_sum_all(input, output, x_dims.production());
  } else {
    int n_in = 1;
    int c_in = 1;
    int h_in = 1;
    int w_in = 1;
    switch (x_dims.size()) {
      case 4:
        w_in = x_dims[3];
      case 3:
        h_in = x_dims[2];
      case 2:
        c_in = x_dims[1];
      case 1:
        n_in = x_dims[0];
        break;
      default:
        LOG(FATAL) << "x_dims.size is " << x_dims.size()
                   << ", which should not be over than 4.";
    }

    if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_sum_n(input, output, n_in, c_in, h_in, w_in);
          break;
        case 1:
          lite::arm::math::reduce_sum_c(input, output, n_in, c_in, h_in, w_in);
          break;
        case 2:
          lite::arm::math::reduce_sum_h(input, output, n_in, c_in, h_in, w_in);
          break;
        case 3:
          lite::arm::math::reduce_sum_w(input, output, n_in, c_in, h_in, w_in);
          break;
        default:
          LOG(FATAL) << "dim[0] is " << dim[0]
                     << ", which should be less than 4.";
      }
    } else if (dim.size() == 2) {
      if (dim[0] == 0 && dim[1] == 1) {
        lite::arm::math::reduce_sum_nc(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 1 && dim[1] == 2) {
        lite::arm::math::reduce_sum_ch(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 2 && dim[1] == 3) {
        lite::arm::math::reduce_sum_hw(input, output, n_in, c_in, h_in, w_in);
      } else {
        LOG(FATAL)
            << "Only support the values of the dim are 0,1 1,2 or 2,3 for now.";
      }
    } else {
      LOG(FATAL) << "dim's size: " << dim.size()
                 << " over than 2, which is not supported now!!";
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reduce_sum,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReduceSumCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
