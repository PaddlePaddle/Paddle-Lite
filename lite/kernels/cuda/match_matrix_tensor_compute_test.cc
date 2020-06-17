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

#include "lite/kernels/cuda/match_matrix_tensor_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

TEST(match_matrix_tensor, normal) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  MatchMatrixTensorCompute kernel;
  operators::MatchMatrixTensorParam param;

  // prepare ins and outs tensor in gpu, including size and lod
  int ix = 5, iy = 4, h = 2, dim_t = 2;
  Tensor x, w, y, out, tmp;
  x.Resize({ix, h});
  w.Resize({h, dim_t, h});
  y.Resize({iy, h});
  out.Resize({18, 1});
  tmp.Resize({20, 1});
  LoD x_lod{};
  x_lod.push_back({0, 2, 5});
  x.set_lod(x_lod);
  LoD y_lod{};
  y_lod.push_back({0, 3, 4});
  y.set_lod(y_lod);

  // init ins tensor in cpu
  Tensor x_cpu, w_cpu, y_cpu, out_cpu, tmp_cpu;
  x_cpu.Resize({ix, h});
  w_cpu.Resize({h, dim_t, h});
  y_cpu.Resize({iy, h});
  out_cpu.Resize({18, 1});
  tmp_cpu.Resize({20, 1});

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* w_cpu_data = w_cpu.mutable_data<float>();
  auto* y_cpu_data = y_cpu.mutable_data<float>();
  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = static_cast<float>(i);
  }
  for (int i = 0; i < w_cpu.numel(); ++i) {
    w_cpu_data[i] = static_cast<float>(i);
  }
  for (int i = 0; i < y_cpu.numel(); ++i) {
    y_cpu_data[i] = static_cast<float>(i);
  }

  // cpu tensor data assigin to gpu tensor
  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  w.Assign<float, lite::DDim, TARGET(kCUDA)>(w_cpu_data, w_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  param.x = &x;
  param.w = &w;
  param.y = &y;
  param.dim_t = dim_t;
  param.out = &out;
  param.tmp = &tmp;
  kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  kernel.SetContext(std::move(ctx));
  kernel.Launch();
  cudaDeviceSynchronize();

  auto* out_cpu_data = out_cpu.mutable_data<float>();
  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  std::vector<float> ref_results = {5,
                                    23,
                                    41,
                                    17,
                                    75,
                                    133,
                                    7,
                                    33,
                                    59,
                                    27,
                                    125,
                                    223,
                                    323,
                                    455,
                                    587,
                                    557,
                                    793,
                                    1029};
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], ref_results[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
