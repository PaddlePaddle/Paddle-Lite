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

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/lookup_table_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

void LookupTableComputeRef(const operators::LookupTableParam& param) {
  auto* ids_t = param.Ids;
  auto* output_t = param.Out;
  int64_t padding_idx = param.padding_idx;
  auto* ids = ids_t->data<int64_t>();
  int64_t ids_numel = ids_t->dims().production();

  auto* table_t = param.W;
  int64_t row_number = table_t->dims()[0];
  int64_t row_width = table_t->dims()[1];

  auto* table = table_t->data<float>();
  auto* output = output_t->mutable_data<float>();
  memset(output, 0, output_t->dims().production() * sizeof(float));
  for (int64_t i = 0; i < ids_numel; ++i) {
    if (padding_idx != -1 && ids[i] == padding_idx) {
      memset(output + i * row_width, 0, row_width * sizeof(float));
    } else {
      CHECK_LT(ids[i], row_number);
      CHECK_GE(ids[i], 0);
      memcpy(output + i * row_width,
             table + ids[i] * row_width,
             row_width * sizeof(float));
    }
  }
}

TEST(lookup_table_cuda, retrieve_op) {
  auto lookup_table = KernelRegistry::Global().Create("lookup_table");
  ASSERT_FALSE(lookup_table.empty());
  ASSERT_TRUE(lookup_table.front());
}

TEST(lookup_table_cuda, init) {
  LookupTableCompute lookup_table;
  ASSERT_EQ(lookup_table.precision(), PRECISION(kFloat));
  ASSERT_EQ(lookup_table.target(), TARGET(kCUDA));
}

TEST(lookup_table_cuda, compute) {
  LookupTableCompute lookup_table;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::LookupTableParam param;

  Tensor w, ids, out;
  Tensor w_cpu, ids_cpu, out_cpu;
  Tensor w_ref, ids_ref, out_ref;

  int64_t padding_idx = 0;

  int vocab_size = 128;
  int emb_size = 64;
  int ids_h = 50;
  int ids_w = 30;

  auto w_dim = DDim({vocab_size, emb_size});
  auto ids_dim = DDim({ids_h, ids_w});
  auto out_dim = DDim({ids_h, ids_w, emb_size});

  int w_num = w_dim.production();
  int ids_num = ids_dim.production();
  int out_num = out_dim.production();

  w.Resize(w_dim);
  ids.Resize(ids_dim);
  out.Resize(out_dim);
  w_cpu.Resize(w_dim);
  ids_cpu.Resize(ids_dim);
  out_cpu.Resize(out_dim);
  w_ref.Resize(w_dim);
  ids_ref.Resize(ids_dim);
  out_ref.Resize(out_dim);

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));
  auto* w_cpu_data = w_cpu.mutable_data<float>();
  auto* ids_cpu_data = ids_cpu.mutable_data<int64_t>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();
  auto* w_ref_data = w_ref.mutable_data<float>();
  auto* ids_ref_data = ids_ref.mutable_data<int64_t>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  // generate test data
  for (int i = 0; i < w_num; i++) {
    w_cpu_data[i] = static_cast<float>(i + 1) / (w_num + 1);
    w_ref_data[i] = static_cast<float>(i + 1) / (w_num + 1);
  }
  for (int i = 0; i < ids_num; i++) {
    ids_cpu_data[i] = i % vocab_size;
    ids_ref_data[i] = i % vocab_size;
  }

  w.Assign<float, lite::DDim, TARGET(kCUDA)>(w_cpu_data, w_dim);
  ids.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(ids_cpu_data, ids_dim);

  param.W = &w;
  param.Ids = &ids;
  param.Out = &out;
  param.padding_idx = padding_idx;
  lookup_table.SetParam(param);

  // run cuda kernel
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  lookup_table.SetContext(std::move(ctx));
  lookup_table.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);

  // run ref kernel
  param.W = &w_ref;
  param.Ids = &ids_ref;
  param.Out = &out_ref;
  LookupTableComputeRef(param);

  for (int i = 0; i < out_num; i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(lookup_table, kCUDA, kFloat, kNCHW, def);
