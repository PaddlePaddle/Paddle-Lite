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

#include "lite/kernels/cuda/search_grnn_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test/test_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

TEST(search_grnn, normal) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  SearchGrnnCompute kernel;
  operators::SearchGrnnParam param;

  int num_input = 6;
  int num_hidden = 6;
  int num_batch = 3;
  Tensor x, wi, wh, out, idx_sorted_by_width, layout_input, tmp_buffer;
  x.Resize({num_batch, num_input});
  wi.Resize({3, num_hidden, num_input});
  wh.Resize({3, num_hidden, num_hidden});
  LoD x_lod{};
  x_lod.push_back({0, 1, 3});
  x.set_lod(x_lod);

  Tensor x_cpu, wi_cpu, wh_cpu, out_cpu, layout_input_cpu, tmp_buffer_cpu;
  x_cpu.Resize({num_batch, num_input});
  wi_cpu.Resize({3, num_hidden, num_input});
  wh_cpu.Resize({3, num_hidden, num_hidden});
  out_cpu.Resize({num_batch, num_hidden});
  layout_input_cpu.Resize({num_batch, num_input});
  tmp_buffer_cpu.Resize({20, num_batch, num_hidden});
  auto* x_cpu_data = x_cpu.mutable_data<float>();
  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = static_cast<float>(i);
  }
  auto* wi_cpu_data = wi_cpu.mutable_data<float>();
  for (int i = 0; i < wi_cpu.numel(); ++i) {
    wi_cpu_data[i] = static_cast<float>(i);
  }
  auto* wh_cpu_data = wh_cpu.mutable_data<float>();
  for (int i = 0; i < wh_cpu.numel(); ++i) {
    wh_cpu_data[i] = static_cast<float>(i);
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  wi.Assign<float, lite::DDim, TARGET(kCUDA)>(wi_cpu_data, wi_cpu.dims());
  wh.Assign<float, lite::DDim, TARGET(kCUDA)>(wh_cpu_data, wh_cpu.dims());

  param.x = &x;
  param.wi = &wi;
  param.wh = &wh;
  param.out = &out;
  param.idx_sorted_by_width = &idx_sorted_by_width;
  param.layout_input = &layout_input;
  param.tmp_buffer = &tmp_buffer;
  param.num_input = num_input;
  param.num_hidden = num_hidden;
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
  LOG(INFO) << "out_data:";
  for (int i = 0; i < out.numel(); i++) {
    // EXPECT_NEAR(out_cpu_data[i], ref_results[i], 1e-5);
    LOG(INFO) << out_cpu_data[i];
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
