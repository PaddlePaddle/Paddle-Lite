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

#include "lite/kernels/x86/fc_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(fc_x86, retrive_op) {
  auto fc =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("fc");
  ASSERT_FALSE(fc.empty());
  ASSERT_TRUE(fc.front());
}

TEST(fc_x86, init) {
  FcCompute<float> fc;
  ASSERT_EQ(fc.precision(), PRECISION(kFloat));
  ASSERT_EQ(fc.target(), TARGET(kX86));
}

template <typename T>
void SetupRandomTensor(lite::Tensor* tensor,
                       const std::vector<int64_t>& shape) {
  static unsigned int seed = 1;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  tensor->Resize(lite::DDim(shape));
  T* ptr = tensor->mutable_data<T>();
  for (int64_t i = 0; i < tensor->dims().production(); ++i) {
    ptr[i] = static_cast<T>(uniform_dist(rng)) - static_cast<T>(0.5);
  }
}

template <typename T>
void FCRef(const T* X,
           const T* W,
           const T* B,
           T* Y,
           int64_t M,
           int64_t N,
           int64_t K,
           bool with_bias,
           bool with_relu) {
  // y = x * w
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        sum += X[i * K + k] * W[k * N + j];
      }
      Y[i * N + j] = sum;
    }
  }

  if (with_bias) {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        Y[i * N + j] += B[j];
      }
    }
    if (with_relu) {
      for (int64_t i = 0; i < M * N; ++i) {
        if (Y[i] < 0) {
          Y[i] = 0;
        }
      }
    }
  }
}

void TestMain(int64_t batch_size,
              const std::vector<int64_t>& w_dims,
              bool with_bias,
              bool with_relu) {
  CHECK_EQ(w_dims.size(), 2U);

  lite::Tensor x;
  lite::Tensor w;
  lite::Tensor w_padding;
  lite::Tensor b;
  lite::Tensor out;

  SetupRandomTensor<float>(&x, {batch_size, w_dims[0]});
  SetupRandomTensor<float>(&w, w_dims);

  FcCompute<float> fc;

  // Set Param
  operators::FcParam param;
  param.in_num_col_dims = 1;
  param.input = &x;
  param.output = &out;
  param.in_mat_dims = x.dims();
  if (with_bias) {
    SetupRandomTensor<float>(&b, {1, w_dims[1]});
    param.bias = &b;

    if (with_relu) {
      param.activation_type = "relu";
    }
  }
  if (w_dims[0] % 128 == 0 && w_dims[1] % 128 == 0) {
    w_padding.Resize(lite::DDim({w_dims[0] + 4, w_dims[1] + 4}));
    float* w_padding_ptr = w_padding.mutable_data<float>();
    for (int64_t i = 0; i < w_dims[0]; ++i) {
      memcpy(w_padding_ptr + i * (w_dims[1] + 4),
             w.data<float>() + i * w_dims[1],
             w_dims[0] * sizeof(float));
    }
    param.w = &w_padding;
    param.padding_weights = true;
  } else {
    param.w = &w;
    param.padding_weights = false;
  }
  fc.SetParam(param);

  // Set Context to X86Context
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  fc.SetContext(std::move(ctx));

  fc.Run();

  lite::Tensor out_ref;
  out_ref.Resize(lite::DDim({batch_size, w_dims[1]}));
  float* out_ref_ptr = out_ref.mutable_data<float>();
  FCRef<float>(x.data<float>(),
               w.data<float>(),
               b.data<float>(),
               out_ref_ptr,
               batch_size,
               w_dims[1],
               w_dims[0],
               with_bias,
               with_relu);
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out.data<float>()[i], out_ref_ptr[i], 1e-5);
  }
}

TEST(fc_x86, base) {
  for (bool with_bias : {false, true}) {
    for (bool with_relu : {false, true}) {
      TestMain(/* batch_size= */ 2, /* w_dims= */ {3, 4}, with_bias, with_relu);
    }
  }
}

TEST(fc_x86, padding) {
  for (bool with_bias : {false, true}) {
    for (bool with_relu : {false, true}) {
      TestMain(
          /* batch_size= */ 32, /* w_dims= */ {128, 128}, with_bias, with_relu);
    }
  }
}

#ifdef PADDLE_WITH_MKLML
TEST(fc_x86, parallel) {
  omp_set_num_threads(3);
  for (bool with_bias : {false, true}) {
    for (bool with_relu : {false, true}) {
      TestMain(
          /* batch_size= */ 32, /* w_dims= */ {128, 128}, with_bias, with_relu);
    }
  }
}
#endif
}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
