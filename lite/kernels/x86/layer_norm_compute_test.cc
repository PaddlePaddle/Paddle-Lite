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

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "lite/backends/x86/jit/helper.h"
#include "lite/backends/x86/jit/kernel_base.h"
#include "lite/backends/x86/jit/kernels.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/x86/layer_norm_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

std::vector<float> ref(lite::Tensor* x,
                       lite::Tensor* Scale,
                       lite::Tensor* Bias,
                       lite::Tensor* y,
                       lite::Tensor* Mean,
                       lite::Tensor* Var,
                       int begin_norm_axis,
                       float epsilon) {
  auto x_dims = x->dims();

  y->mutable_data<float>();
  Mean->mutable_data<float>();
  Var->mutable_data<float>();

  auto matrix_dim = x_dims.Flatten2D(begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  lite::DDim matrix_shape({left, right});

  x->Resize(matrix_shape);
  Tensor out;
  out.ShareDataWith(*y);
  out.Resize(matrix_shape);

  auto ker = paddle::lite::jit::KernelFuncs<jit::LayerNormTuple<float>,
                                            lite::fluid::CPUPlace>::Cache()
                 .At(right);
  ker(x->mutable_data<float>(),
      out.mutable_data<float>(),
      Mean->mutable_data<float>(),
      Var->mutable_data<float>(),
      Scale->data<float>(),
      Bias->data<float>(),
      static_cast<int>(left),
      static_cast<const float>(epsilon),
      right);

  std::vector<float> ref_data;
  auto result = out.mutable_data<float>();
  for (int i = 0; i < y->dims().production(); ++i) {
    ref_data.emplace_back(result[i]);
  }
  return ref_data;
}

// layer_norm
TEST(layer_norm_x86, retrive_op) {
  auto layer_norm = KernelRegistry::Global().Create("layer_norm");
  ASSERT_FALSE(layer_norm.empty());
  ASSERT_TRUE(layer_norm.front());
}

TEST(layer_norm_x86, init) {
  lite::kernels::x86::LayerNormCompute<float> layer_norm;
  ASSERT_EQ(layer_norm.precision(), PRECISION(kFloat));
  ASSERT_EQ(layer_norm.target(), TARGET(kX86));
}

TEST(layer_norm_x86, run_test) {
  lite::Tensor x;
  lite::Tensor Scale;
  lite::Tensor Bias;

  lite::Tensor out;
  lite::Tensor Mean;
  lite::Tensor Var;

  std::vector<int64_t> x_shape({1, 2, 3, 1});
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape({1, 2, 3, 1});
  out.Resize(lite::DDim(out_shape));

  int begin_norm_axis = 0;
  float epsilon = 1e-5;
  int pre = 1;
  int post = 1;
  for (int i = 0; i < begin_norm_axis; ++i) {
    pre *= x_shape[i];
  }
  for (size_t i = begin_norm_axis; i < x_shape.size(); ++i) {
    post *= x_shape[i];
  }
  std::vector<int64_t> scale_shape({post});
  Scale.Resize(scale_shape);
  std::vector<int64_t> bias_shape({post});
  Bias.Resize(bias_shape);

  auto x_data = x.mutable_data<float>();
  auto scale_data = Scale.mutable_data<float>();
  auto bias_data = Bias.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto mean_data = Mean.mutable_data<float>();
  auto var_data = Var.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < Scale.dims().production(); ++i) {
    scale_data[i] = 1.5;
  }
  for (int64_t i = 0; i < Bias.dims().production(); ++i) {
    bias_data[i] = 0.25;
  }

  LayerNormCompute<float> layer_norm;
  operators::LayerNormParam param;

  param.X = &x;
  param.Y = &out;
  param.Scale = &Scale;
  param.Bias = &Bias;
  param.Mean = &Mean;
  param.Variance = &Var;
  param.begin_norm_axis = begin_norm_axis;
  param.epsilon = epsilon;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  layer_norm.SetContext(std::move(ctx));
  layer_norm.SetParam(param);
  layer_norm.Run();

  std::vector<float> ref_data =
      ref(&x, &Scale, &Bias, &out, &Mean, &Var, begin_norm_axis, epsilon);
  for (int j = 0; j < out.dims().production(); ++j) {
    EXPECT_NEAR(out_data[j], ref_data[j], 1e-5);
  }
  LOG(INFO) << *mean_data;
  LOG(INFO) << *var_data;
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(layer_norm, kX86, kFloat, kNCHW, def);
