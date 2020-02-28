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

#include "lite/kernels/arm/lookup_table_compute.h"
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void lookup_table_compute_ref(const operators::LookupTableParam &param) {
  auto *ids_t = param.Ids;
  auto *output_t = param.Out;
  int64_t padding_idx = param.padding_idx;
  auto *ids = ids_t->data<int64_t>();
  int64_t ids_numel = ids_t->dims().production();

  auto *table_t = param.W;
  int64_t row_number = table_t->dims()[0];
  int64_t row_width = table_t->dims()[1];

  auto *table = table_t->data<float>();
  auto *output = output_t->mutable_data<float>();
  memset(output, 0, output_t->dims().production() * sizeof(float));
  for (int64_t i = 0; i < ids_numel; ++i) {
    if (padding_idx != -1 && ids[i] == padding_idx) {
      memset(output + i * row_width, 0, row_width * sizeof(float));
    } else {
      CHECK_LT(ids[i], row_number);
      CHECK_GE(ids[i], 0);
      memcpy(output + i * row_width,
             table + ids[i] * row_width,
             row_width * sizeof(float));
    }
  }
}

TEST(lookup_table_arm, retrieve_op) {
  auto lookup_table =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kAny)>(
          "lookup_table");
  ASSERT_FALSE(lookup_table.empty());
  ASSERT_TRUE(lookup_table.front());
}

TEST(lookup_table_arm, init) {
  LookupTableCompute lookup_table;
  ASSERT_EQ(lookup_table.precision(), PRECISION(kAny));
  ASSERT_EQ(lookup_table.target(), TARGET(kARM));
}

TEST(lookup_table_arm, compute) {
  LookupTableCompute lookup_table;
  operators::LookupTableParam param;
  lite::Tensor w, ids, out, out_ref;
  int64_t padding_idx = -1;

  auto w_dim = DDim(std::vector<int64_t>({4, 5}));
  auto ids_dim = DDim(std::vector<int64_t>({3, 2}));
  auto out_dim = DDim(std::vector<int64_t>({3, 2, 5}));

  w.Resize(w_dim);
  ids.Resize(ids_dim);
  out.Resize(out_dim);
  out_ref.Resize(out_dim);

  auto *w_data = w.mutable_data<float>();
  auto *ids_data = ids.mutable_data<int64_t>();
  auto *out_data = out.mutable_data<float>();
  auto *out_ref_data = out_ref.mutable_data<float>();

  int w_num = w_dim.production();
  for (int i = 0; i < w_num; i++) {
    w_data[i] = static_cast<float>(i + 1) / (w_num + 1);
  }
  int ids_num = ids_dim.production();
  for (int i = 0; i < ids_num; i++) {
    ids_data[i] = i % 4;
  }
  int out_num = out_dim.production();

  param.W = &w;
  param.Ids = &ids;
  param.Out = &out;
  lookup_table.SetParam(param);
  lookup_table.Run();
  param.Out = &out_ref;
  lookup_table_compute_ref(param);
  for (int i = 0; i < out_num; i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(lookup_table, kARM, kAny, kNCHW, def);
