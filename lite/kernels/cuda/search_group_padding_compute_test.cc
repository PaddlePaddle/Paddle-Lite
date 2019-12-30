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

#include "lite/kernels/cuda/search_group_padding_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

TEST(search_group_padding_cuda, run_test) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  lite::Tensor x, x_cpu, x_ref;
  lite::Tensor out_emb_padding, out_emb_padding_cpu, out_emb_padding_ref;
  lite::Tensor out_new, out_new_cpu, out_new_ref;
  lite::Tensor out_padding, out_padding_cpu, out_padding_ref;

  int x_dims0 = 2;
  int x_dims1 = 3;

  x.Resize({x_dims0, x_dims1});
  x_cpu.Resize({x_dims0, x_dims1});
  x_ref.Resize({x_dims0, x_dims1});
  out_emb_padding.Resize({1, x_dims1});
  out_emb_padding_cpu.Resize({1, x_dims1});
  out_emb_padding_ref.Resize({1, x_dims1});
  out_new.Resize({x_dims0, 1});
  out_new_cpu.Resize({x_dims0, 1});
  out_new_ref.Resize({x_dims0, 1});
  out_padding.Resize({1, 1});
  out_padding_cpu.Resize({1, 1});
  out_padding_ref.Resize({1, 1});

  LoD x_lod{};
  x_lod.push_back({0, 1});
  x.set_lod(x_lod);

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* x_ref_data = x_ref.mutable_data<float>();
  auto* out_emb_padding_data =
      out_emb_padding.mutable_data<float>(TARGET(kCUDA));
  auto* out_emb_padding_cpu_data = out_emb_padding_cpu.mutable_data<float>();
  auto* out_emb_padding_ref_data = out_emb_padding_ref.mutable_data<float>();
  auto* out_new_data = out_new.mutable_data<float>(TARGET(kCUDA));
  auto* out_new_cpu_data = out_new_cpu.mutable_data<float>();
  auto* out_new_ref_data = out_new_ref.mutable_data<float>();
  auto* out_padding_data = out_padding.mutable_data<float>(TARGET(kCUDA));
  auto* out_padding_cpu_data = out_padding_cpu.mutable_data<float>();
  auto* out_padding_ref_data = out_padding_ref.mutable_data<float>();

  for (int64_t i = 0; i < x_cpu.dims().production(); i++) {
    x_cpu_data[i] = static_cast<float>(i);
    x_ref_data[i] = static_cast<float>(i);
  }
  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  out_emb_padding_ref_data[0] = 0.f;
  out_emb_padding_ref_data[1] = 1.f;
  out_emb_padding_ref_data[2] = 2.f;
  out_new_ref_data[0] = 0.f;
  out_new_ref_data[1] = 0.f;
  out_padding_ref_data[0] = 0.f;

  SearchGroupPaddingCompute sgp_kernel;
  operators::SearchGroupPaddingParam param;

  param.x = &x;
  param.out_emb_padding = &out_emb_padding;
  param.out_new = &out_new;
  param.out_padding = &out_padding;

  sgp_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  sgp_kernel.SetContext(std::move(ctx));
  sgp_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(out_emb_padding_cpu_data,
                          out_emb_padding_data,
                          sizeof(float) * out_emb_padding.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(out_new_cpu_data,
                          out_new_data,
                          sizeof(float) * out_new.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(out_padding_cpu_data,
                          out_padding_data,
                          sizeof(float) * out_padding.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < out_emb_padding_cpu.dims().production(); i++) {
    EXPECT_NEAR(out_emb_padding_cpu_data[i], out_emb_padding_ref_data[i], 1e-5);
  }
  for (int i = 0; i < out_new_cpu.dims().production(); i++) {
    EXPECT_NEAR(out_new_cpu_data[i], out_new_ref_data[i], 1e-5);
  }
  for (int i = 0; i < out_padding_cpu.dims().production(); i++) {
    EXPECT_NEAR(out_padding_cpu_data[i], out_padding_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(search_group_padding, kCUDA, kFloat, kNCHW, def);
