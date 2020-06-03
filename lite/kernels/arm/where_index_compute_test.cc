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

#include "lite/kernels/arm/where_index_compute.h"
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
namespace arm {

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

TEST(where_index_int32, init) {
  WhereIndexCompute<int32_t, PRECISION(kInt32)> where_index_int32;
  ASSERT_EQ(where_index_int32.precision(), PRECISION(kInt32));
  ASSERT_EQ(where_index_int32.target(), TARGET(kARM));
}

TEST(where_index_int32, retrive_op) {
  auto where_index_int32 =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kInt32)>(
          "where_index");
  ASSERT_FALSE(where_index_int32.empty());
  ASSERT_TRUE(where_index_int32.front());
}

TEST(where_index_int32, compute) {
  paddle::lite::DeviceInfo::Init();
  WhereIndexCompute<int32_t, PRECISION(kInt32)> where_index_int32;
  operators::WhereIndexParam param;

  lite::Tensor input;
  lite::Tensor output;
  lite::Tensor output_ref;
  param.input = &input;
  param.output = &output;
  where_index_int32.SetParam(param);
  for (auto& n : {1, 2, 4}) {
    for (auto& c : {1, 3, 21, 32}) {
      for (auto& h : {1, 5, 63}) {
        for (auto& w : {1, 5, 64}) {
          for (auto& dim_size : {1, 2, 3, 4}) {
            for (auto& th : {1, 2, 4}) {
              std::unique_ptr<paddle::lite::KernelContext> ctx1(
                  new paddle::lite::KernelContext);
              auto& ctx = ctx1->As<paddle::lite::ARMContext>();
              ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(3), th);
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
              int* indata = input.mutable_data<int32_t>();
              std::default_random_engine engine;
              std::uniform_real_distribution<float> dist(-1, 1);
              for (int i = 0; i < indim.production(); ++i) {
                indata[i] = static_cast<int>(dist(engine) > 0);
              }
              where_index_int32.Run();
              where_index_compute_ref<int32_t>(&input, &output_ref);
              const int64_t* outdata = output.data<int64_t>();
              const int64_t* outdata_ref = output_ref.data<int64_t>();
              CHECK_EQ(output.dims(), output_ref.dims())
                  << "where_index int32 out shape error! out_dim is not equal "
                     "to out_ref dim";
              for (int i = 0; i < output.numel(); i++) {
                if (std::abs(outdata[i] - outdata_ref[i]) > 0) {
                  LOG(FATAL) << "where_index int32 cmp error, i: " << i
                             << ", output_data: " << outdata[i]
                             << ", output_ref_data: " << outdata_ref[i];
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(where_index_int64, init) {
  WhereIndexCompute<int64_t, PRECISION(kInt64)> where_index_int64;
  ASSERT_EQ(where_index_int64.precision(), PRECISION(kInt64));
  ASSERT_EQ(where_index_int64.target(), TARGET(kARM));
}

TEST(where_index_int64, retrive_op) {
  auto where_index_int64 =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kInt64)>(
          "where_index");
  ASSERT_FALSE(where_index_int64.empty());
  ASSERT_TRUE(where_index_int64.front());
}

TEST(where_index_int64, compute) {
  paddle::lite::DeviceInfo::Init();
  WhereIndexCompute<int64_t, PRECISION(kInt64)> where_index_int64;
  operators::WhereIndexParam param;

  lite::Tensor input;
  lite::Tensor output;
  lite::Tensor output_ref;
  param.input = &input;
  param.output = &output;
  where_index_int64.SetParam(param);
  for (auto& n : {1, 2, 4}) {
    for (auto& c : {1, 3, 21, 32}) {
      for (auto& h : {1, 5, 63}) {
        for (auto& w : {1, 5, 64}) {
          for (auto& dim_size : {1, 2, 3, 4}) {
            for (auto& th : {1, 2, 4}) {
              std::unique_ptr<paddle::lite::KernelContext> ctx1(
                  new paddle::lite::KernelContext);
              auto& ctx = ctx1->As<paddle::lite::ARMContext>();
              ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(3), th);
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
              int64_t* indata = input.mutable_data<int64_t>();
              std::default_random_engine engine;
              std::uniform_real_distribution<float> dist(-1, 1);
              for (int i = 0; i < indim.production(); ++i) {
                indata[i] = static_cast<int64_t>(dist(engine) > 0);
              }
              where_index_int64.Run();
              where_index_compute_ref<int64_t>(&input, &output_ref);
              const int64_t* outdata = output.data<int64_t>();
              const int64_t* outdata_ref = output_ref.data<int64_t>();
              CHECK_EQ(output.dims(), output_ref.dims())
                  << "where_index int64 out shape error! out_dim is not equal "
                     "to out_ref dim";
              for (int i = 0; i < output.numel(); i++) {
                if (std::abs(outdata[i] - outdata_ref[i]) > 0) {
                  LOG(FATAL) << "where_index int64 cmp error, i: " << i
                             << ", output_data: " << outdata[i]
                             << ", output_ref_data: " << outdata_ref[i];
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(where_index, kARM, kInt32, kNCHW, def);
USE_LITE_KERNEL(where_index, kARM, kInt64, kNCHW, def);
