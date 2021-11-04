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

#pragma once
#include <algorithm>
#include <utility>
#include <vector>

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename DataType>
class ArgsortCompute
    : public KernelLite<TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny)> {
 public:
  using param_t = operators::ArgsortParam;

  void Run() {
    auto& param = Param<operators::ArgsortParam>();
    const DataType* x_data = param.X->template data<DataType>();
    DataType* out_val = param.Out->template mutable_data<DataType>();
    auto out_ind = param.Indices->template mutable_data<int64_t>();
    DDim x_dims = param.X->dims();
    int axis = param.axis;
    int dim_size = x_dims.size();
    bool descending = param.descending;
    if (axis < 0) {
      axis += dim_size;
    }

    int outer_size = x_dims.count(0, axis);
    int axis_size = x_dims[axis];
    int inner_size = x_dims.count(axis + 1, dim_size);
    int sort_size = axis_size * inner_size;
    LITE_PARALLEL_BEGIN(n, tid, outer_size) {
      const DataType* in_data = x_data + n * sort_size;
      DataType* out_data = out_val + n * sort_size;
      int64_t* out_ind_data = out_ind + n * sort_size;
      for (int i = 0; i < inner_size; i++) {
        std::vector<std::pair<DataType, int>> vec;
        vec.resize(axis_size);
        for (int j = 0; j < axis_size; j++) {
          vec[j] = std::make_pair(in_data[j * inner_size + i], j);
        }
        if (descending) {
          std::sort(vec.begin(),
                    vec.end(),
                    [](std::pair<DataType, int> a, std::pair<DataType, int> b) {
                      return a.first > b.first;
                    });
        } else {
          std::sort(vec.begin(),
                    vec.end(),
                    [](std::pair<DataType, int> a, std::pair<DataType, int> b) {
                      return a.first < b.first;
                    });
        }
        for (int j = 0; j < axis_size; j++) {
          out_data[j * inner_size + i] = vec[j].first;
          out_ind_data[j * inner_size + i] = vec[j].second;
        }
      }
    }
    LITE_PARALLEL_END()
  }

  virtual ~ArgsortCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
