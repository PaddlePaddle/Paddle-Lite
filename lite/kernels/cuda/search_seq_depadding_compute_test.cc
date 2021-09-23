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

#include "lite/kernels/cuda/search_seq_depadding_compute.h"
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

TEST(search_seq_depadding, normal) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  SearchSeqDepaddingCompute kernel;
  operators::SearchSeqDepaddingParam param;

  Tensor pad, src, out;
  pad.Resize({2 * 3, 4});
  src.Resize({3, 1});
  out.Resize({3, 4});
  LoD pad_lod{};
  pad_lod.push_back({0, 4, 6});
  pad.set_lod(pad_lod);
  LoD src_lod{};
  src_lod.push_back({0, 2, 3});
  src.set_lod(src_lod);

  Tensor pad_cpu, src_cpu, out_cpu;
  pad_cpu.Resize({2 * 3, 4});
  src_cpu.Resize({3, 1});
  out_cpu.Resize({3, 4});

  auto* pad_cpu_data = pad_cpu.mutable_data<float>();
  auto* src_cpu_data = src_cpu.mutable_data<float>();
  for (int i = 0; i < pad_cpu.numel(); ++i) {
    pad_cpu_data[i] = static_cast<float>(i);
  }

  pad.Assign<float, lite::DDim, TARGET(kCUDA)>(pad_cpu_data, pad_cpu.dims());
  src.Assign<float, lite::DDim, TARGET(kCUDA)>(src_cpu_data, src_cpu.dims());

  param.pad = &pad;
  param.src = &src;
  param.out = &out;
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

  std::vector<float> ref_results = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19};
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], ref_results[i], 1e-5);
    // LOG(INFO) << out_cpu_data[i];
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
