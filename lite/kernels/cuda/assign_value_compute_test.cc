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

#include "lite/kernels/cuda/assign_value_compute.h"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class AssignValueTest : public ::testing::Test {
 protected:
  AssignValueTest() : dtype(5), shape({1}) {
    int num =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    fp32_values.resize(num);
    int32_values.resize(num);
    int64_values.resize(num);
    bool_values.resize(num);
    for (int i = 0; i < num; ++i) {
      fp32_values[i] = i + 5;
      int32_values[i] = i;
      int64_values[i] = i;
      bool_values[i] = i;
    }
    std::vector<int64_t> out_shape(shape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) out_shape[i] = shape[i];
    Out_ref.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(Out_ref.dims());
    Out_cpu.Resize(Out_ref.dims());

    cpu_base(&Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.shape = shape;
    param.dtype = dtype;
    param.fp32_values = fp32_values;
    param.int32_values = int32_values;
    param.int64_values = int64_values;
    param.bool_values = bool_values;
    param.Out = &Out_gpu;
  }

  void float_data_init() {}

  void half_data_init() {}

  void cpu_base(lite::Tensor* Out) {
    if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
      for (size_t i = 0; i < int32_values.size(); ++i) {
        Out->mutable_data<int>()[i] = int32_values[i];
      }
    } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
      for (size_t i = 0; i < fp32_values.size(); ++i) {
        Out->mutable_data<float>()[i] = fp32_values[i];
      }
    } else if (dtype == static_cast<int>(lite::core::FluidType::INT64)) {
      for (size_t i = 0; i < int64_values.size(); ++i) {
        Out->mutable_data<int64_t>()[i] = int64_values[i];
      }
    } else if (dtype == static_cast<bool>(lite::core::FluidType::BOOL)) {
      for (size_t i = 0; i < bool_values.size(); ++i) {
        Out->mutable_data<bool>()[i] = bool_values[i];
      }
    } else {
      LOG(FATAL) << "Unsupported dtype for assign_value_op:" << dtype;
    }
  }

  int dtype;
  std::vector<int> shape;
  std::vector<float> fp32_values;
  std::vector<int> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<int> bool_values;

  lite::Tensor Out_ref;
  lite::Tensor Out_gpu;
  lite::Tensor Out_cpu;

  operators::AssignValueParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(AssignValueTest, fp32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  AssignValueCompute kernel;
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
