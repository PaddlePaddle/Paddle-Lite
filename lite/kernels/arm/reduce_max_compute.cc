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
#include "lite/backends/arm/math/reduce_max.h"
#include "lite/backends/arm/math/reduce_max_min.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T>
void ReduceMaxCompute<T>::Run() {
  auto& param = Param<operators::ReduceParam>();
  const T* input = param.X->template data<T>();
  auto x_dims = param.X->dims();

  int x_rank = x_dims.size();
  T* output = param.Out->template mutable_data<T>();
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
      lite::arm::math::reduce_all_of_three<T>(
          input, output, x_dims[0], x_dims[1], x_dims[2]);
    } else if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_first_of_three<T>(
              input, output, x_dims[0], x_dims[1], x_dims[2]);
          break;
        case 1:
          lite::arm::math::reduce_second_of_three<T>(
              input, output, x_dims[0], x_dims[1], x_dims[2]);
          break;

        case 2:
          lite::arm::math::reduce_third_of_three<T>(
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
      lite::arm::math::reduce_all<T>(input, output, n_in, c_in, h_in, w_in);
    } else if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_n<T>(input, output, n_in, c_in, h_in, w_in);
          break;
        case 1:
          lite::arm::math::reduce_c<T>(input, output, n_in, c_in, h_in, w_in);
          break;
        case 2:
          lite::arm::math::reduce_h<T>(input, output, n_in, c_in, h_in, w_in);
          break;
        case 3:
          lite::arm::math::reduce_w<T>(input, output, n_in, c_in, h_in, w_in);
          break;
        default:
          LOG(FATAL) << "error!!!";
      }
    } else if (dim.size() == 2) {
      if (dim[0] == 0 && dim[1] == 1) {
        lite::arm::math::reduce_nc<T>(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 1 && dim[1] == 2) {
        lite::arm::math::reduce_ch<T>(input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 2 && dim[1] == 3) {
        lite::arm::math::reduce_hw<T>(input, output, n_in, c_in, h_in, w_in);
      } else {
        LOG(FATAL) << "invalid dim!!";
      }
    } else {
      LOG(FATAL) << "dim's size over than 2, which is not supported now!!";
    }
  } else if (x_dims.size() == 2) {
    int first_in = x_dims[0];
    int second_in = x_dims[1];
    if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::arm::math::reduce_first_of_two<T>(
              input,
              output,
              first_in,
              second_in,
              lite::arm::math::MaxMinType::kMax);
          break;
        case 1:
          lite::arm::math::reduce_second_of_two<T>(
              input,
              output,
              first_in,
              second_in,
              lite::arm::math::MaxMinType::kMax);
          break;
        default:
          LOG(FATAL) << "error!!!";
      }
    } else {
      LOG(FATAL) << "dim's size over than 1, which is not supported now!!";
    }  // x_dims == 2 && dim.size() == 1
  } else if (x_dims.size() == 1) {
    lite::arm::math::reduce_one_line_max<T>(input, output, x_dims[0]);
  } else {
    LOG(FATAL) << "only support input with 1 to 4 dimensions now!!";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using float_reduce_max = paddle::lite::kernels::arm::ReduceMaxCompute<float>;
REGISTER_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, float_reduce_max, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using int64_reduce_max = paddle::lite::kernels::arm::ReduceMaxCompute<int64_t>;
REGISTER_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, int64_reduce_max, i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
