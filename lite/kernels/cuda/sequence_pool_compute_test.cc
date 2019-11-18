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

#include "lite/kernels/cuda/sequence_pool_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/backends/x86/math/sequence_pooling.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

namespace {

static void sequence_pool_ref(const operators::SequencePoolParam& param, ) {
  auto* x = param.X;
  auto* out = param.Out;
  auto dims = x->dims();
  auto lod = x->lod();
  CHECK_EQ(lod.size(), 1UL);
  CHECK_GE(dims[0], static_cast<int64_t>(lod[0].size() - 1));

  dims[0] = lod[0].size() - 1;
  out->Resize({dims});
  out->mutable_data<float>();
  lite::Tensor* index = nullptr;

  const bool is_test = true;
  float pad_value = 0.0;

  lite::x86::math::SequencePoolFunctor<lite::TargetType::kX86, float> pool;
  pool(context, param.pool_type, pad_value, *x, out, is_test, index);
}

#define PREPARE_INPUT_DATA(name)                                 \
  name.Resize({name##_lod_len, feature_len});                    \
  name##_cpu.Resize({name##_lod_len, feature_len});              \
  name##_ref.Resize({name##_lod_len, feature_len});              \
  name.set_lod(lod_info_##name);                                 \
  name##_cpu.set_lod(lod_info_##name);                           \
  name##_ref.set_lod(lod_info_##name);                           \
  float* name##_cpu_data = name##_cpu.mutable_data<float>();     \
  float* name##_ref_data = name##_ref.mutable_data<float>();     \
  for (int i = 0; i < name##_cpu.numel(); ++i) {                 \
    name##_cpu_data[i] = (i - 2.0) * 1.0;                        \
    name##_ref_data[i] = (i - 2.0) * 1.0;                        \
  }                                                              \
  name.Assign<float, lite::DDim, TARGET(kCUDA)>(name##_cpu_data, \
                                                name##_cpu.dims());

#define PREPARE_OUTPUT_INFO(name)              \
  name##_cpu.Resize({y_lod_len, feature_len}); \
  name##_ref.Resize({y_lod_len, feature_len}); \
  name.Resize({y_lod_len, feature_len});       \
  float* name##_cpu_data = name##_cpu.mutable_data<float>();

}  // namespace

TEST(sequence_pool_cuda, normal) {
  SequencePoolCompute seq_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  std::unique_ptr<KernelContext> ctx_ref(new KernelContext);
  auto& context_ref = ctx_ref->As<X86Context>();

  operators::SequencePoolParam param;
  lite::Tensor x1, x2, x3, x1_cpu, x2_cpu, x3_cpu, x1_ref, x2_ref, x3_ref;
  lite::Tensor y, y_cpu, y_ref;

  int32_t x1_lod_len = 10, feature_len = 4;
  int32_t x2_lod_len = 4, x3_lod_len = 8;
  int32_t y_lod_len = x1_lod_len + x2_lod_len + x3_lod_len;
  LoD lod_info_x1{{0, 3, 5, 6, 10}};
  LoD lod_info_x2{{0, 1, 2, 3, 4}};
  LoD lod_info_x3{{0, 2, 4, 6, 8}};
  LoD lod_info_y{{0, 0, 0, 0, 0}};
  for (size_t i = 0; i < lod_info_x1[0].size(); ++i) {
    lod_info_y[0][i] =
        lod_info_x1[0][i] + lod_info_x2[0][i] + lod_info_x3[0][i];
  }

  PREPARE_INPUT_DATA(x1);
  PREPARE_INPUT_DATA(x2);
  PREPARE_INPUT_DATA(x3);
  PREPARE_OUTPUT_INFO(y);

  param.X = &x1;
  param.Out = &y;
  param.pool_type = "AVERAGE";
  seq_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  seq_kernel.SetContext(std::move(ctx));
  seq_kernel.Run();
  cudaDeviceSynchronize();

  auto* y_data = y.mutable_data<float>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(float) * y.numel(), IoDirection::DtoH);

  param.X = &x1_ref;
  param.Out = &y_ref;
  sequence_pool_ref(param);

  lite::x86::math::SequencePoolFunctor<lite::TargetType::kX86, float> pool;
  pool(context, param.pool_type, pad_value, *x, out, is_test, index);

  float* y_ref_data = y_ref.mutable_data<float>();
  for (int i = 0; i < y.numel(); i++) {
    EXPECT_NEAR(y_cpu_data[i], y_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
