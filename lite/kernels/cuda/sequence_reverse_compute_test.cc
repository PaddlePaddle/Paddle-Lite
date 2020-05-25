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

#include "lite/kernels/cuda/sequence_reverse_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

static void sequence_reverse_ref(const lite::Tensor* x, lite::Tensor* y) {
  const auto* x_data = x->data<float>();
  auto seq_offset = x->lod()[x->lod().size() - 1];
  int width = x->numel() / x->dims()[0];
  auto* y_data = y->mutable_data<float>();
  for (int i = 0; i < static_cast<int>(seq_offset.size()) - 1; ++i) {
    auto start_pos = seq_offset[i];
    auto end_pos = seq_offset[i + 1];
    for (auto pos = start_pos; pos < end_pos; ++pos) {
      auto cur_pos = end_pos - pos - 1 + start_pos;
      std::memcpy(y_data + pos * width,
                  x_data + cur_pos * width,
                  width * sizeof(float));
    }
  }
}

TEST(sequence_reverse_cuda, normal) {
  SequenceReverseCompute<float, PRECISION(kFloat)> seq_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::SequenceReverseParam param;
  lite::Tensor x, x_cpu, x_ref;
  lite::Tensor y, y_cpu, y_ref;

  int32_t lod_len = 10, feature_len = 4;
  LoD lod_info{{0, 2, 4}, {0, 3, 5, 6, 10}};

  x.Resize({lod_len, feature_len});
  x_cpu.Resize({lod_len, feature_len});
  x_ref.Resize({lod_len, feature_len});
  y.Resize({lod_len, feature_len});
  y_cpu.Resize({lod_len, feature_len});
  y_ref.Resize({lod_len, feature_len});
  x.set_lod(lod_info);
  x_cpu.set_lod(lod_info);
  x_ref.set_lod(lod_info);
  y.set_lod(lod_info);
  y_cpu.set_lod(lod_info);
  y_ref.set_lod(lod_info);

  auto* y_data = y.mutable_data<float>(TARGET(kCUDA));

  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* x_ref_data = x_ref.mutable_data<float>();
  float* y_cpu_data = y_cpu.mutable_data<float>();
  float* y_ref_data = y_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = (i - 2.0) * 1.0;
    x_ref_data[i] = (i - 2.0) * 1.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());

  param.X = &x;
  param.Out = &y;
  seq_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  seq_kernel.SetContext(std::move(ctx));
  seq_kernel.Run();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(float) * y.numel(), IoDirection::DtoH);

  sequence_reverse_ref(&x_ref, &y_ref);
  for (int i = 0; i < y.numel(); i++) {
    EXPECT_NEAR(y_cpu_data[i], y_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
