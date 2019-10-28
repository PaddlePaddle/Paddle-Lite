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

#include "lite/kernels/cuda/bilinear_interp_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

TEST(bilinear_interp, normal) {
  BilinearInterpCompute bilinear_interp_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::InterpolateParam param;

  Tensor x, osz, out;
  Tensor x_cpu, osz_cpu, out_cpu;
  Tensor x_ref, osz_ref, out_ref;

  int n = 1, c = 1, in_h = 3, in_w = 3;
  int out_h = 6, out_w = 6;
  float scale = 2.0;

  param.out_h = out_h;
  param.out_w = out_w;
  param.scale = scale;
  param.align_corners = false;
  param.align_mode = 0;

  x.Resize({n, c, in_h, in_w});
  osz.Resize({2});
  out.Resize({n, c, out_h, out_w});

  x_cpu.Resize({n, c, in_h, in_w});
  osz_cpu.Resize({2});
  out_cpu.Resize({n, c, out_h, out_w});

  x_ref.Resize({n, c, in_h, in_w});
  osz_ref.Resize({2});
  out_ref.Resize({n, c, out_h, out_w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* osz_cpu_data = osz_cpu.mutable_data<float>();
  float* out_cpu_data = out_cpu.mutable_data<float>();

  float* x_ref_data = x_ref.mutable_data<float>();
  float* osz_ref_data = osz_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  osz_cpu_data[0] = out_h;
  osz_cpu_data[1] = out_w;
  osz_ref_data[0] = out_h;
  osz_ref_data[1] = out_w;

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  osz.Assign<float, lite::DDim, TARGET(kCUDA)>(osz_cpu_data, osz_cpu.dims());

  param.X = &x;
  param.OutSize = &osz;
  param.Out = &out;
  bilinear_interp_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  bilinear_interp_kernel.SetContext(std::move(ctx));
  bilinear_interp_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  for (int i = 0; i < out.numel(); i++) {
    LOG(INFO) << out_cpu_data[i];
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
