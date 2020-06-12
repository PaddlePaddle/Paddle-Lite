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

#include "lite/kernels/host/where_index_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void where_index_compute_ref(lite::Tensor* condition, lite::Tensor* out) {
  auto dims = condition->dims();
  auto numel = condition->numel();
  const int64_t rank = static_cast<int64_t>(dims.size());
  const T* cond_data = condition->data<T>();
  std::vector<int64_t> true_index;
  for (auto i = 0; i < numel; i++) {
    if (static_cast<bool>(cond_data[i])) {
      true_index.push_back(i);
    }
  }
  int64_t true_num = static_cast<int64_t>(true_index.size());
  out->Resize({true_num, rank});
  int64_t* out_ptr = out->mutable_data<int64_t>();
  if (true_num == 0) {
    return;
  }

  std::vector<int64_t> stride(rank);
  stride[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * dims[i + 1];
  }
  for (int i = 0; i < true_num; ++i) {
    int64_t index = true_index[i];
    for (int j = 0; j < rank; ++j) {
      out_ptr[i * rank + j] = index / stride[j];
      index -= out_ptr[i * rank + j] * stride[j];
    }
  }
}

TEST(where_index, init) {
  WhereIndexCompute where_index;
  ASSERT_EQ(where_index.precision(), PRECISION(kAny));
  ASSERT_EQ(where_index.target(), TARGET(kHost));
}

TEST(where_index, retrive_op) {
  auto where_index =
      KernelRegistry::Global().Create<TARGET(kHost), PRECISION(kAny)>(
          "where_index");
  ASSERT_FALSE(where_index.empty());
  ASSERT_TRUE(where_index.front());
}

TEST(where_index, compute) {
  paddle::lite::DeviceInfo::Init();
  WhereIndexCompute where_index;
  operators::WhereIndexParam param;

  lite::Tensor input;
  lite::Tensor output;
  lite::Tensor output_ref;
  param.input = &input;
  param.output = &output;
  where_index.SetParam(param);
  for (auto& n : {1, 2, 4}) {
    for (auto& c : {1, 3, 21, 32}) {
      for (auto& h : {1, 5, 63}) {
        for (auto& w : {1, 5, 64}) {
          for (auto& dim_size : {1, 2, 3, 4}) {
            for (int i = 0; i < 5; ++i) {
              std::vector<int64_t> in_shape;
              in_shape.push_back(n);
              in_shape.push_back(c);
              in_shape.push_back(h);
              in_shape.push_back(w);
              int outer = 1;
              for (int i = dim_size - 1; i < in_shape.size(); ++i) {
                outer *= in_shape[i];
              }
              in_shape.resize(dim_size);
              in_shape[dim_size - 1] = outer;

              DDim indim(in_shape);
              LOG(INFO) << "in dims: ";
              for (int i = 0; i < dim_size; ++i) {
                LOG(INFO) << in_shape[i];
              }
              input.Resize(indim);
              std::default_random_engine engine;
              std::uniform_real_distribution<float> dist(-1, 1);
              if (i == 0) {
                int* indata = input.mutable_data<int32_t>();
                for (int i = 0; i < indim.production(); ++i) {
                  indata[i] = static_cast<int>(dist(engine) > 0);
                }
                where_index_compute_ref<int32_t>(&input, &output_ref);
              } else if (i == 1) {
                int64_t* indata = input.mutable_data<int64_t>();
                for (int i = 0; i < indim.production(); ++i) {
                  indata[i] = static_cast<int64_t>(dist(engine) > 0);
                }
                where_index_compute_ref<int64_t>(&input, &output_ref);
              } else if (i == 2) {
                int8_t* indata = input.mutable_data<int8_t>();
                for (int i = 0; i < indim.production(); ++i) {
                  indata[i] = static_cast<int8_t>(dist(engine) > 0);
                }
                where_index_compute_ref<int8_t>(&input, &output_ref);
              } else if (i == 3) {
                bool* indata = input.mutable_data<bool>();
                for (int i = 0; i < indim.production(); ++i) {
                  indata[i] = dist(engine) > 0;
                }
                where_index_compute_ref<bool>(&input, &output_ref);
              } else {
                float* indata = input.mutable_data<float>();
                for (int i = 0; i < indim.production(); ++i) {
                  indata[i] = dist(engine) > 0;
                }
                where_index_compute_ref<float>(&input, &output_ref);
              }
              where_index.Run();
              const int64_t* outdata = output.data<int64_t>();
              const int64_t* outdata_ref = output_ref.data<int64_t>();
              CHECK_EQ(output.dims(), output_ref.dims())
                  << "where_index out shape error! out_dim is not equal "
                     "to out_ref dim";
              for (int i = 0; i < output.numel(); i++) {
                if (std::abs(outdata[i] - outdata_ref[i]) > 0) {
                  LOG(FATAL) << "where_index cmp error, i: " << i
                             << ", output_data: " << outdata[i]
                             << ", output_ref_data: " << outdata_ref[i]
                             << "input precision: "
                             << static_cast<int>(input.precision());
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(where_index, kHost, kAny, kAny, def);
