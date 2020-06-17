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

#include "lite/kernels/cuda/sequence_pool_compute.h"
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

TEST(sequence_pool_cuda, normal) {
  SequencePoolCompute seq_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  lite::Tensor x, x_cpu, out, out_cpu;
  lite::LoD lod;
  lod.push_back(std::vector<uint64_t>{0, 10});

  x.set_lod(lod);
  x_cpu.set_lod(lod);
  const size_t second_dim = 8u;
  std::vector<int64_t> input_shape{static_cast<int64_t>(lod[0].back()),
                                   static_cast<int64_t>(second_dim)};
  lite::DDim in_dims(input_shape);
  x.Resize(in_dims);
  x_cpu.Resize(in_dims);

  const size_t out_first_dim = lod[0].size() - 1;
  std::vector<int64_t> output_shape{static_cast<int64_t>(out_first_dim),
                                    static_cast<int64_t>(second_dim)};
  lite::DDim out_dims(output_shape);
  out.Resize(out_dims);
  out_cpu.Resize(out_dims);

  auto x_cpu_data = x_cpu.mutable_data<float>();
  auto out_data = out.mutable_data<float>(TARGET(kCUDA));
  auto out_cpu_data = out_cpu.mutable_data<float>();

  for (int64_t i = 0; i < x_cpu.dims().production(); i++) {
    x_cpu_data[i] = 1.1f * i;
  }
  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());

  operators::SequencePoolParam param;
  param.X = &x;
  param.Out = &out;
  std::vector<std::string> pool_types(
      {"MAX", "AVERAGE", "SUM", "SQRT", "FIRST", "LAST"});
  std::map<std::string, std::vector<float>> type_map;
  type_map["MAX"] = {79.2, 80.3, 81.4, 82.5, 83.6, 84.7, 85.8, 86.9};
  type_map["AVERAGE"] = {39.6, 40.7, 41.8, 42.9, 44, 45.1, 46.2, 47.3};
  type_map["SUM"] = {396, 407, 418, 429, 440, 451, 462, 473};
  type_map["SQRT"] = {
      125.226, 128.705, 132.183, 135.662, 139.14, 142.619, 146.097, 149.576};
  type_map["FIRST"] = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  type_map["LAST"] = {79.2, 80.3, 81.4, 82.5, 83.6, 84.7, 85.8, 86.9};

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  seq_kernel.SetContext(std::move(ctx));
  for (std::string pool_type : pool_types) {
    param.pool_type = pool_type;
    seq_kernel.SetParam(param);

    seq_kernel.Run();
    cudaDeviceSynchronize();

    CopySync<TARGET(kCUDA)>(out_cpu_data,
                            out_data,
                            sizeof(float) * out_cpu.numel(),
                            IoDirection::DtoH);

    std::vector<float> ref_results = type_map[pool_type];

    for (int i = 0; i < out_cpu.numel(); i++) {
      EXPECT_NEAR(out_cpu_data[i], ref_results[i], 1e-3);
    }
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
