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

#include "lite/kernels/host/reduce_compute.h"
#include <set>
#include <string>
#include <vector>
#include "lite/backends/host/math/reduce.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, typename Functor>
void ReduceCompute<T, Functor>::Run() {
  auto& param = Param<operators::ReduceParam>();
  const T* input = param.X->template data<T>();
  auto x_dims = param.X->dims();
  int x_rank = x_dims.size();
  T* output = param.Out->template mutable_data<T>();

  std::vector<int> dim = param.dim;
  bool reduce_all = param.reduce_all;

  if (!dim.empty()) {
    for (size_t i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }

  std::set<int> dims_set(dim.begin(), dim.end());
  bool full_dim = true;
  for (int i = 0; i < static_cast<int>(x_dims.size()); i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  if (reduce_all) {
    lite::host::math::reduce_all<T, Functor>(
        input, output, x_dims.production());
  } else {
    // TODO(zhiqiang, juncai): update according to Paddle
    int new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < x_dims.size(); ++j) {
      new_dims[j] = static_cast<int>(x_dims[j]);
    }
    int n_in = new_dims[0];
    int c_in = new_dims[1];
    int h_in = new_dims[2];
    int w_in = new_dims[3];
    if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          lite::host::math::reduce_n<T, Functor>(
              input, output, n_in, c_in, h_in, w_in);
          break;
        case 1:
          lite::host::math::reduce_c<T, Functor>(
              input, output, n_in, c_in, h_in, w_in);
          break;
        case 2:
          lite::host::math::reduce_h<T, Functor>(
              input, output, n_in, c_in, h_in, w_in);
          break;
        case 3:
          lite::host::math::reduce_w<T, Functor>(
              input, output, n_in, c_in, h_in, w_in);
          break;
        default:
          LOG(FATAL) << "not support reduce dim == " << dim[0];
      }
    } else if (dim.size() == 2) {
      if (dim[0] == 0 && dim[1] == 1) {
        lite::host::math::reduce_nc<T, Functor>(
            input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 1 && dim[1] == 2) {
        lite::host::math::reduce_ch<T, Functor>(
            input, output, n_in, c_in, h_in, w_in);
      } else if (dim[0] == 2 && dim[1] == 3) {
        lite::host::math::reduce_hw<T, Functor>(
            input, output, n_in, c_in, h_in, w_in);
      } else {
        LOG(FATAL) << "invalid dim!!";
      }
    } else {
      LOG(FATAL) << "dim's size over than 2, which is not supported now!!";
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ReduceAll = paddle::lite::kernels::host::
    ReduceCompute<bool, paddle::lite::host::math::LogicalAnd>;
REGISTER_LITE_KERNEL(reduce_all, kHost, kFloat, kNCHW, ReduceAll, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using ReduceAny = paddle::lite::kernels::host::
    ReduceCompute<bool, paddle::lite::host::math::LogicalOr>;
REGISTER_LITE_KERNEL(reduce_any, kHost, kFloat, kNCHW, ReduceAny, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();
