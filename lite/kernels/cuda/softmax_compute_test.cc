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

#include "lite/kernels/cuda/softmax_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;
using DDim = lite::DDim;

template <typename dtype>
static void softmax_compute_ref(const operators::SoftmaxParam& param) {
  const dtype* x_data = param.x->mutable_data<const dtype>();
  dtype* output_data = param.output->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  ASSERT_EQ(x_dims, param.output->dims());
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] = exp(x_data[offset] - max_data);
      sum_data += output_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

TEST(softmax_cuda, compute) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  SoftmaxCompute softmax;
  operators::SoftmaxParam param;
  softmax.SetContext(std::move(ctx));
  lite::Tensor x;
  lite::Tensor x_cpu;
  lite::Tensor output;
  lite::Tensor output_cpu;
  lite::Tensor output_ref;
  for (auto n : {1, 3}) {
    for (auto c : {1, 4}) {
      for (auto h : {5, 1, 112}) {
        for (auto w : {1, 6, 112}) {
          for (auto axis : {-2, -1, 0, 1, 2}) {
            x.Resize({n, c, h, w});
            x_cpu.Resize({n, c, h, w});
            output.Resize({n, c, h, w});
            output_cpu.Resize({n, c, h, w});
            output_ref.Resize({n, c, h, w});
            auto* x_cpu_data = x_cpu.mutable_data<float>();
            auto* output_data = output.mutable_data<float>(TARGET(kCUDA));
            auto* output_cpu_data = output_ref.mutable_data<float>();
            auto* output_ref_data = output_ref.mutable_data<float>();
            for (int i = 0; i < x.dims().production(); i++) {
              x_cpu_data[i] = i;
            }
            x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data,
                                                       x_cpu.dims());
            param.x = &x;
            param.axis = axis;
            param.output = &output;
            softmax.SetParam(param);
            softmax.Launch();
            param.x = &x_cpu;
            param.output = &output_ref;
            softmax_compute_ref<float>(param);
            cudaDeviceSynchronize();
            CopySync<TARGET(kCUDA)>(output_cpu_data,
                                    output_data,
                                    sizeof(float) * output.numel(),
                                    IoDirection::DtoH);
            for (int i = 0; i < output.dims().production(); i++) {
              EXPECT_NEAR(output_cpu_data[i], output_ref_data[i], 1e-5);
            }
          }
        }
      }
    }
  }
}
}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
