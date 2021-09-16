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

#include "lite/kernels/cuda/topk_pooling_compute.h"

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class TopkPooingTest : public ::testing::Test {
 protected:
  TopkPooingTest()
      : num(2),
        channels(4),
        height(4),
        width(4),
        top_k(2),
        feat_map_num(height * width),
        x_lod({{0, 4, 7}}),
        y_lod({{0, 4, 7}}),
        x_shape({num, channels, height, width}),
        out_shape({num, channels * top_k}) {
    CHECK_EQ(x_lod[0].size(), num + 1) << "invalid input.";
    for (size_t i = 1; i < x_lod[0].size(); ++i) {
      CHECK_LE(x_lod[0][i] - x_lod[0][i - 1], height) << "invalid input.";
    }

    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));
    X_ref.set_lod(x_lod);
    Y_gpu.Resize(lite::DDim(x_shape));
    Y_ref.Resize(lite::DDim(x_shape));
    Y_ref.set_lod(y_lod);
    auto x_ref_data = X_ref.mutable_data<float>();
    auto y_ref_data = Y_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 16);
    }
    for (int64_t i = 0; i < Y_ref.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i % 16);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.X = &X_gpu;
    param.Y = &Y_gpu;
    param.Out = &Out_gpu;
    param.top_k = top_k;
    param.feat_map_num = feat_map_num;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    Y_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(Y_ref.data<float>(),
                                                   Y_gpu.dims());
    Y_gpu.set_lod(Y_ref.lod());
  }

  void half_data_init() {}

  void cpu_base(const lite::Tensor* X,
                const lite::Tensor* Y,
                lite::Tensor* Out) {}

  int num, channels, height, width;
  int top_k, feat_map_num;
  std::vector<std::vector<uint64_t>> x_lod, y_lod;
  std::vector<int64_t> x_shape, out_shape;
  lite::Tensor X_ref, Y_ref, Out_ref;
  lite::Tensor X_gpu, Y_gpu;
  lite::Tensor Out_cpu, Out_gpu;

  operators::TopkPoolingParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(TopkPooingTest, fp32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  TopkPoolingCompute<float> kernel;
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
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
