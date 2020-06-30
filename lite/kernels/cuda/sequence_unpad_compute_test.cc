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

#include "lite/kernels/cuda/sequence_unpad_compute.h"

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
// #include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SequenceUnpadTest : public ::testing::Test {
 protected:
  SequenceUnpadTest()
      : batch(5),
        features(2),
        padded_length(3),
        out_lod({{0, 2, 5}}),
        x_shape({static_cast<int64_t>(out_lod[0].size() - 1),
                 padded_length,
                 features}),
        out_shape({batch, features}) {
    X_ref.Resize(lite::DDim(x_shape));
    X_gpu.Resize(X_ref.dims());

    Length_ref.Resize(
        lite::DDim({static_cast<int64_t>(out_lod[0].size() - 1)}));
    Length_gpu.Resize(Length_ref.dims());

    auto* x_ref_data = X_ref.mutable_data<float>();
    auto* length_ref_data = Length_ref.mutable_data<int64_t>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < out_lod[0].size() - 1; ++i) {
      length_ref_data[i] = out_lod[0][i + 1] - out_lod[0][i];
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_ref.set_lod(out_lod);
    Out_gpu.Resize(Out_ref.dims());
    Out_gpu.set_lod(Out_ref.lod());
    Out_cpu.Resize(Out_ref.dims());
    Out_cpu.set_lod(Out_ref.lod());

    cpu_base(&X_ref, &Length_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.X = &X_gpu;
    param.Length = &Length_gpu;
    param.Out = &Out_gpu;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    Length_gpu.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(
        Length_ref.data<int64_t>(), Length_gpu.dims());
  }

  void half_data_init() {}

  void cpu_base(const lite::Tensor* X,
                const lite::Tensor* Length,
                lite::Tensor* Out) {
    auto* out_data = Out->mutable_data<float>();

    for (size_t i = 0; i < 4; ++i) {
      out_data[i] = i;
    }
    for (size_t i = 6; i < 12; ++i) {
      out_data[i - 2] = i;
    }
  }

  int batch, features, padded_length;
  LoD out_lod;
  std::vector<int64_t> x_shape, out_shape;

  lite::Tensor X_ref, Out_ref, Length_ref;
  lite::Tensor X_gpu, Out_gpu, Length_gpu;
  lite::Tensor Out_cpu, Length_cpu;

  operators::SequencePadParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SequenceUnpadTest, fp32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  SequenceUnpadCompute<float, PRECISION(kFloat)> kernel;
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
