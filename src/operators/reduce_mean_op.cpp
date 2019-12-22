/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef REDUCE_MEAN_OP

#include "operators/reduce_mean_op.h"
#include <algorithm>

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
void ReduceMeanOp<DeviceType, T>::InferShape() const {

  auto dims = this->param_.Dims();
  auto x_dims = this->param_.InputX()->dims();
  bool reduce_all = false;
  bool keep_dim = this->param_.KeepDim();
  auto x_rank = x_dims.size();
  if (dims.size() != 0) {
    for (int i = 0; i < dims.size(); i++) {
      if (dims[i] < 0) {
        dims[i] = x_rank + dims[i];
      }
    }
  }
  std::sort(dims.begin(), dims.end());
  if (dims.size() == 0) {
    reduce_all = true;
  }
  std::vector<int64_t> out_dims;
  if (reduce_all) {
    if (keep_dim) {
      out_dims.push_back(x_rank);
      out_dims.push_back(1);
    } else {
      out_dims.push_back(1);
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      out_dims.push_back(x_dims[i]);
    }
    if (keep_dim) {
      for (size_t i = 0; i < dims.size(); ++i) {
        out_dims[dims[i]] = 1;
      }
    } else {
      const int64_t kDelFlag = -2;
      for (size_t i = 0; i < dims.size(); ++i) {
        out_dims[dims[i]] = kDelFlag;
      }
      out_dims.erase(remove(out_dims.begin(), out_dims.end(), kDelFlag),
                     out_dims.end());
    }
    this->param_.Output()->Resize(framework::make_ddim(out_dims));
    if (dims[0] != 0) {
      // Only pass LoD when not reducing on the first dim.
      // *(this->param_).Output()->mutable_lod() = param_.X->lod();
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(reduce_mean, ops::ReduceMeanOp);
#endif

#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(reduce_mean, ops::ReduceMeanOp);
#endif

#endif  // REDUCE_MEAN_OP
