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

#include "lite/kernels/cuda/sequence_pad_compute.h"

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

class SequencePadTest : public ::testing::Test {
 protected:
  SequencePadTest()
      : batch(5),
        features(2),
        padded_length(3),
        x_lod({{0, 2, 5}}),
        x_shape({batch, features}),
        pad_value_shape({features}),
        out_shape({static_cast<int64_t>(x_lod[0].size() - 1),
                   padded_length,
                   features}) {
    X_ref.Resize(lite::DDim(x_shape));
    X_ref.set_lod(x_lod);
    X_gpu.Resize(X_ref.dims());

    PadValue_ref.Resize(lite::DDim(pad_value_shape));
    PadValue_gpu.Resize(PadValue_ref.dims());

    Length_ref.Resize(lite::DDim({static_cast<int64_t>(x_lod[0].size() - 1)}));
    Length_gpu.Resize(Length_ref.dims());

    auto x_ref_data = X_ref.mutable_data<float>();
    auto pad_value_ref_data = PadValue_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < PadValue_ref.numel(); i++) {
      pad_value_ref_data[i] = static_cast<float>(i);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(Out_ref.dims());
    Out_cpu.Resize(Out_ref.dims());
    cpu_base(&X_ref, &PadValue_ref, &Out_ref, &Length_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.X = &X_gpu;
    param.PadValue = &PadValue_gpu;
    param.Length = &Length_gpu;
    param.Out = &Out_gpu;
    param.padded_length = padded_length;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    PadValue_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(
        PadValue_ref.data<float>(), PadValue_gpu.dims());
  }

  void half_data_init() {}

  void cpu_base(const lite::Tensor* X,
                const lite::Tensor* PadValue,
                lite::Tensor* Out,
                lite::Tensor* Length) {
    auto* length_data = Length->mutable_data<int64_t>();
    auto* out_data = Out->mutable_data<float>();
    length_data[0] = 2;
    length_data[1] = 3;

    for (size_t i = 0; i < 4; ++i) {
      out_data[i] = i;
    }
    out_data[4] = 0;
    out_data[5] = 1;
    for (size_t i = 4; i < 10; ++i) {
      out_data[2 + i] = i;
    }
  }

  int batch, features, padded_length;
  LoD x_lod;
  std::vector<int64_t> x_shape, pad_value_shape, out_shape;

  lite::Tensor X_ref, PadValue_ref, Out_ref, Length_ref;
  lite::Tensor X_gpu, PadValue_gpu, Out_gpu, Length_gpu;
  lite::Tensor Out_cpu, Length_cpu;

  operators::SequencePadParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SequencePadTest, fp32) {
  float_data_init();
  SequencePadCompute<float, PRECISION(kFloat)> kernel;
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
  CopySync<TARGET(kCUDA)>(Length_cpu.mutable_data<int64_t>(),
                          Length_gpu.data<int64_t>(),
                          sizeof(int64_t) * Length_gpu.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < Out_gpu.numel(); ++i) {
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < Length_gpu.numel(); ++i) {
    EXPECT_NEAR(
        Length_cpu.data<int64_t>()[i], Length_ref.data<int64_t>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
