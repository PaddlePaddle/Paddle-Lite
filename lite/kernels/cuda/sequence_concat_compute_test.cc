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

#include "lite/kernels/cuda/sequence_concat_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

namespace {
inline LoD ConcatLoD(const std::vector<lite::Tensor*>& xs,
                     std::vector<lite::Tensor>* xs_in_order) {
  std::vector<size_t> result;
  result.resize(xs[0]->lod()[0].size());

  for (size_t i = 1; i < result.size(); ++i) {
    size_t sum = 0;
    for (size_t j = 0; j < xs.size(); ++j) {
      auto& x_lod = xs[j]->lod()[0];
      if (x_lod[i - 1] < x_lod[i]) {
        xs_in_order->emplace_back(xs[j]->Slice<float>(x_lod[i - 1], x_lod[i]));
      }
      sum += x_lod[i];
    }
    result[i] = sum;
  }
  LoD lod;
  lod.emplace_back(result);
  return lod;
}

static void sequence_concat_ref(const std::vector<lite::Tensor*>& xs,
                                lite::Tensor* out) {
  std::vector<int64_t> out_dims;
  int64_t batch_size = 0;
  int64_t feature_size = 0;
  for (const auto& tensor : xs) {
    const auto x_dims = tensor->dims();
    if (out_dims.empty()) {
      out_dims = x_dims.Vectorize();
    }
    batch_size += x_dims[0];
    if (feature_size == 0) {
      feature_size = x_dims.production() / x_dims[0];
    } else {
      CHECK_EQ(feature_size, x_dims.production() / x_dims[0])
          << "Inputs of sequence concat must have same feature size";
    }
  }
  out_dims[0] = batch_size;
  out->Resize(out_dims);
  std::vector<lite::Tensor> x_in_order;
  out->set_lod(ConcatLoD(xs, &x_in_order));

  int num = x_in_order.size();
  std::vector<int64_t> input_cols(num);
  for (int i = 0; i < num; ++i) {
    input_cols[i] = x_in_order[i].numel();
  }
  float* out_data = out->mutable_data<float>();
  int col_idx = 0;
  for (int j = 0; j < num; ++j) {
    int col_len = input_cols[j];
    auto input_data = x_in_order[j].data<float>();
    memcpy(out_data + col_idx, input_data, sizeof(float) * col_len);
    col_idx += col_len;
  }
}

#define PREPARE_INPUT_DATA(name)                                 \
  name.Resize({name##_lod_len, feature_len});                    \
  name##_cpu.Resize({name##_lod_len, feature_len});              \
  name##_ref.Resize({name##_lod_len, feature_len});              \
  name.set_lod(lod_info_##name);                                 \
  name##_cpu.set_lod(lod_info_##name);                           \
  name##_ref.set_lod(lod_info_##name);                           \
  float* name##_cpu_data = name##_cpu.mutable_data<float>();     \
  float* name##_ref_data = name##_ref.mutable_data<float>();     \
  for (int i = 0; i < name##_cpu.numel(); ++i) {                 \
    name##_cpu_data[i] = (i - 2.0) * 1.0;                        \
    name##_ref_data[i] = (i - 2.0) * 1.0;                        \
  }                                                              \
  name.Assign<float, lite::DDim, TARGET(kCUDA)>(name##_cpu_data, \
                                                name##_cpu.dims());

#define PREPARE_OUTPUT_INFO(name)              \
  name##_cpu.Resize({y_lod_len, feature_len}); \
  name##_ref.Resize({y_lod_len, feature_len}); \
  name.Resize({y_lod_len, feature_len});       \
  float* name##_cpu_data = name##_cpu.mutable_data<float>();

}  // namespace

TEST(sequence_concat_cuda, normal) {
  SequenceConcatCompute seq_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::SequenceConcatParam param;
  lite::Tensor x1, x2, x3, x1_cpu, x2_cpu, x3_cpu, x1_ref, x2_ref, x3_ref;
  lite::Tensor y, y_cpu, y_ref;

  int32_t x1_lod_len = 10, feature_len = 4;
  int32_t x2_lod_len = 4, x3_lod_len = 8;
  int32_t y_lod_len = x1_lod_len + x2_lod_len + x3_lod_len;
  LoD lod_info_x1{{0, 3, 5, 6, 10}};
  LoD lod_info_x2{{0, 1, 2, 3, 4}};
  LoD lod_info_x3{{0, 2, 4, 6, 8}};
  LoD lod_info_y{{0, 0, 0, 0, 0}};
  for (size_t i = 0; i < lod_info_x1[0].size(); ++i) {
    lod_info_y[0][i] =
        lod_info_x1[0][i] + lod_info_x2[0][i] + lod_info_x3[0][i];
  }

  PREPARE_INPUT_DATA(x1);
  PREPARE_INPUT_DATA(x2);
  PREPARE_INPUT_DATA(x3);
  PREPARE_OUTPUT_INFO(y);

  param.X = std::vector<lite::Tensor*>({&x1, &x2, &x3});
  param.Out = &y;
  seq_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  seq_kernel.SetContext(std::move(ctx));
  seq_kernel.Run();
  cudaDeviceSynchronize();

  auto* y_data = y.mutable_data<float>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      y_cpu_data, y_data, sizeof(float) * y.numel(), IoDirection::DtoH);

  std::vector<lite::Tensor*> input_ref({&x1_ref, &x2_ref, &x3_ref});
  sequence_concat_ref(input_ref, &y_ref);
  float* y_ref_data = y_ref.mutable_data<float>();
  for (int i = 0; i < y.numel(); i++) {
    EXPECT_NEAR(y_cpu_data[i], y_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
