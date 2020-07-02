// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/kernels/cuda/sequence_mask_compute.h"
// #include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SequenceMaskTest : public ::testing::Test {
 protected:
  SequenceMaskTest()
      : maxlen(4),
        out_dtype(5),
        x_data({3, 2, 1, 0}),
        out_shape({static_cast<int64_t>(x_data.size()), maxlen}) {
    X_ref.Resize(lite::DDim({static_cast<int64_t>(x_data.size())}));
    X_gpu.Resize(X_ref.dims());

    auto* x_ref_data = X_ref.mutable_data<int64_t>();

    // prepare input
    for (size_t i = 0; i < x_data.size(); i++) {
      x_ref_data[i] = x_data[i];
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(Out_ref.dims());
    Out_cpu.Resize(Out_ref.dims());
    cpu_base(&X_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.X = &X_gpu;
    param.Y = &Out_gpu;
    param.maxlen = maxlen;
    param.out_dtype = out_dtype;
  }

  void float_data_init() {
    X_gpu.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(X_ref.data<int64_t>(),
                                                     X_gpu.dims());
  }

  void half_data_init() {}

  void cpu_base(const lite::Tensor* X, lite::Tensor* Out) {
    auto* out_data = Out->mutable_data<float>();

    for (size_t i = 0; i < x_data.size(); ++i) {
      for (int j = 0; j < maxlen; ++j) {
        out_data[i * maxlen + j] = j < x_data[i] ? 1 : 0;
      }
    }
  }

  int maxlen, out_dtype;
  std::vector<int64_t> x_data, out_shape;

  lite::Tensor X_ref, Out_ref;
  lite::Tensor X_gpu, Out_gpu;
  lite::Tensor Out_cpu;

  operators::SequenceMaskParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SequenceMaskTest, fp32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  SequenceMaskCompute<float, PRECISION(kFloat)> kernel;
  kernel.SetParam(param);
  kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp32, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  CopySync<TARGET(kCUDA)>(Out_cpu.mutable_data<float>(),
                          Out_gpu.data<float>(),
                          sizeof(float) * Out_gpu.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < Out_gpu.numel(); ++i) {
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
