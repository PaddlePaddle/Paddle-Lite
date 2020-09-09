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

#include "lite/kernels/cuda/concat_compute.h"
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

bool infer_shape(const operators::ConcatParam& param) {
  std::vector<lite::DDim> input_dims;
  for (auto p : param.x) {
    input_dims.push_back(p->dims());
  }
  size_t axis = static_cast<size_t>(param.axis);
  const size_t n = input_dims.size();
  CHECK_GT_OR_FALSE(n, 0);
  auto& out_dims = input_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += input_dims[i][j];
      } else {
        CHECK_EQ_OR_FALSE(out_dims[j], input_dims[i][j]);
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }
  // Set output dims
  param.output->Resize(lite::DDim(out_dims));
  return true;
}

void concat_compute_ref(const operators::ConcatParam& param) {
  std::vector<lite::Tensor*> input = param.x;
  int axis = param.axis;
  infer_shape(param);

  lite::Tensor* output = param.output;
  int num = input.size();
  int rows = 1;
  auto dim_0 = input[0]->dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int out_rows = rows, out_cols = 0;

  std::vector<int> input_cols(input.size());
  for (int i = 0; i < num; ++i) {
    int input_i_numel = input[i]->dims().size() == 0 ? 0 : 1;
    for (size_t didx = 0; didx < input[i]->dims().size(); ++didx) {
      input_i_numel *= input[i]->dims()[didx];
    }
    int t_cols = input_i_numel / rows;
    out_cols += t_cols;
    input_cols[i] = t_cols;
  }

  auto output_data = output->mutable_data<float>();
  int col_idx = 0;
  for (int j = 0; j < num; ++j) {
    int col_len = input_cols[j];
    auto input_data = input[j]->data<float>();
    for (int k = 0; k < out_rows; ++k) {
      memcpy(output_data + k * out_cols + col_idx,
             input_data + k * col_len,
             sizeof(float) * col_len);
    }
    col_idx += col_len;
  }
}

TEST(concat, init) {
  ConcatCompute<float> concat;
  ASSERT_EQ(concat.precision(), PRECISION(kFloat));
  ASSERT_EQ(concat.target(), TARGET(kCUDA));
}

TEST(concat, compute_input_multi) {
  ConcatCompute<float> concat_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ConcatParam param;
  operators::ConcatParam param_ref;

  LOG(INFO) << "test concat start";
  // init param
  std::vector<lite::Tensor*> x;
  std::vector<lite::Tensor*> x_cpu;
  std::vector<lite::Tensor*> x_ref;
  lite::Tensor out;
  lite::Tensor out_cpu;
  lite::Tensor out_ref;
  lite::Tensor tensorA;
  lite::Tensor tensorB;
  lite::Tensor tensorC;
  lite::Tensor tensorD;
  lite::Tensor tensorA_cpu;
  lite::Tensor tensorB_cpu;
  lite::Tensor tensorC_cpu;
  lite::Tensor tensorD_cpu;
  lite::Tensor tensorA_ref;
  lite::Tensor tensorB_ref;
  lite::Tensor tensorC_ref;
  lite::Tensor tensorD_ref;

  DDimLite ddimA({1, 3, 38, 38});
  DDimLite ddimB({1, 4, 38, 38});
  DDimLite ddimC({1, 5, 38, 38});
  DDimLite ddimD({1, 6, 38, 38});

  tensorA.Resize(ddimA);
  tensorB.Resize(ddimB);
  tensorC.Resize(ddimC);
  tensorD.Resize(ddimD);
  tensorA_cpu.Resize(ddimA);
  tensorB_cpu.Resize(ddimB);
  tensorC_cpu.Resize(ddimC);
  tensorD_cpu.Resize(ddimD);
  tensorA_ref.Resize(ddimA);
  tensorB_ref.Resize(ddimB);
  tensorC_ref.Resize(ddimC);
  tensorD_ref.Resize(ddimD);

  out.Resize({1, 18, 38, 38});
  out_cpu.Resize({1, 18, 38, 38});
  out_ref.Resize({1, 18, 38, 38});
  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));
  auto* out_cpu_data = out_cpu.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();
  for (int i = 0; i < tensorA_cpu.numel(); i++) {
    tensorA_cpu.mutable_data<float>()[i] = i;
    tensorA_ref.mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < tensorB_cpu.numel(); i++) {
    tensorB_cpu.mutable_data<float>()[i] = i + 3;
    tensorB_ref.mutable_data<float>()[i] = i + 3;
  }
  for (int i = 0; i < tensorC_cpu.numel(); i++) {
    tensorC_cpu.mutable_data<float>()[i] = i + 6;
    tensorC_ref.mutable_data<float>()[i] = i + 6;
  }
  for (int i = 0; i < tensorD_cpu.numel(); i++) {
    tensorD_cpu.mutable_data<float>()[i] = i + 9;
    tensorD_ref.mutable_data<float>()[i] = i + 9;
  }
  tensorA.Assign<float, lite::DDim, TARGET(kCUDA)>(
      tensorA_cpu.mutable_data<float>(), tensorA_cpu.dims());
  tensorB.Assign<float, lite::DDim, TARGET(kCUDA)>(
      tensorB_cpu.mutable_data<float>(), tensorB_cpu.dims());
  tensorC.Assign<float, lite::DDim, TARGET(kCUDA)>(
      tensorC_cpu.mutable_data<float>(), tensorC_cpu.dims());
  tensorD.Assign<float, lite::DDim, TARGET(kCUDA)>(
      tensorD_cpu.mutable_data<float>(), tensorD_cpu.dims());

  x.push_back(&tensorA);
  x.push_back(&tensorB);
  x.push_back(&tensorC);
  x.push_back(&tensorD);
  x_cpu.push_back(&tensorA_cpu);
  x_cpu.push_back(&tensorB_cpu);
  x_cpu.push_back(&tensorC_cpu);
  x_cpu.push_back(&tensorD_cpu);
  x_ref.push_back(&tensorA_ref);
  x_ref.push_back(&tensorB_ref);
  x_ref.push_back(&tensorC_ref);
  x_ref.push_back(&tensorD_ref);

  for (int cur_axis : {1}) {
    param.x = x;
    param.axis = cur_axis;
    param.output = &out;

    concat_kernel.SetParam(param);
    LOG(INFO) << "test concat start cur_axis:" << cur_axis;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context.SetExecStream(stream);

    concat_kernel.SetContext(std::move(ctx));
    concat_kernel.Launch();
    cudaDeviceSynchronize();
    LOG(INFO) << "sync end";
    CHECK(cudaSuccess == cudaMemcpy(out_cpu_data,
                                    out_data,
                                    sizeof(float) * out.numel(),
                                    cudaMemcpyDeviceToHost));
    LOG(INFO) << "concat.Run end";

    param_ref.x = x_ref;
    param_ref.axis = cur_axis;
    param_ref.output = &out_ref;

    LOG(INFO) << "concat_compute_ref start";
    concat_compute_ref(param_ref);
    LOG(INFO) << "concat_compute_ref end";

    for (int i = 0; i < out_ref.numel(); i++) {
      EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
    }
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
