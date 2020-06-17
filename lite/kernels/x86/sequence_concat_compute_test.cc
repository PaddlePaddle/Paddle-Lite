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

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/sequence_concat_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

namespace {
inline LoD ConcatLoD(const std::vector<lite::Tensor*>& xs,
                     std::vector<lite::Tensor>* xs_in_order) {
  std::vector<uint64_t> result;
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

#define PREPARE_INPUT(name)                        \
  name.Resize({name##_lod_len, feature_len});      \
  name.set_lod(lod_info_##name);                   \
  float* name##_data = name.mutable_data<float>(); \
  for (int i = 0; i < name.numel(); ++i) {         \
    name##_data[i] = (i - 2.0) * 1.0;              \
  }

}  // namespace

TEST(sequence_concat_x86, retrive_op) {
  auto sequence_concat = KernelRegistry::Global().Create("sequence_concat");
  ASSERT_FALSE(sequence_concat.empty());
  ASSERT_TRUE(sequence_concat.front());
}

TEST(sequence_concat_x86, init) {
  SequenceConcatCompute<float> sequence_concat;
  ASSERT_EQ(sequence_concat.precision(), PRECISION(kFloat));
  ASSERT_EQ(sequence_concat.target(), TARGET(kX86));
}

TEST(sequence_concat_x86, run_test) {
  SequenceConcatCompute<float> seq_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();

  operators::SequenceConcatParam param;
  lite::Tensor x1, x2, x3;
  lite::Tensor y, y_ref;

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

  PREPARE_INPUT(x1);
  PREPARE_INPUT(x2);
  PREPARE_INPUT(x3);

  y_ref.Resize({y_lod_len, feature_len});
  y.Resize({y_lod_len, feature_len});
  y_ref.set_lod(lod_info_y);
  y.set_lod(lod_info_y);

  std::vector<lite::Tensor*> xs{&x1, &x2, &x3};

  param.X = xs;
  param.Out = &y;
  seq_kernel.SetParam(param);

  seq_kernel.SetContext(std::move(ctx));
  seq_kernel.Run();

  auto* y_data = y.mutable_data<float>();
  sequence_concat_ref(xs, &y_ref);
  float* y_ref_data = y_ref.mutable_data<float>();

  for (int i = 0; i < y.numel(); i++) {
    EXPECT_NEAR(y_data[i], y_ref_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sequence_concat, kX86, kFloat, kNCHW, def);
