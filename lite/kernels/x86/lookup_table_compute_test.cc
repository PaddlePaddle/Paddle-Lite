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

#include "lite/kernels/x86/lookup_table_compute.h"
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(lookup_table_x86, compute) {
  LookupTableCompute<float> lookup_table;
  operators::LookupTableParam param;
  lite::Tensor w, ids, out, out_ref;
  int64_t padding_idx = -1;

  int vocab_size = 40;
  int emb_size = 50;
  int ids_h = 30;
  int ids_w = 20;

  auto w_dim = DDim({vocab_size, emb_size});
  auto ids_dim = DDim({ids_h, ids_w});
  auto out_dim = DDim({ids_h, ids_w, emb_size});

  w.Resize(w_dim);
  ids.Resize(ids_dim);
  out.Resize(out_dim);
  out_ref.Resize(out_dim);

  auto* w_data = w.mutable_data<float>();
  auto* ids_data = ids.mutable_data<int64_t>();
  auto* out_data = out.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  int w_num = w_dim.production();
  for (int i = 0; i < w_num; i++) {
    w_data[i] = static_cast<float>(i + 1) / (w_num + 1);
  }
  int ids_num = ids_dim.production();
  for (int i = 0; i < ids_num; i++) {
    ids_data[i] = i % vocab_size;
  }
  int out_num = out_dim.production();
  for (int i = 0; i < out_num; i++) {
    out_ref_data[i] =
        static_cast<float>((i % (vocab_size * emb_size)) + 1) / (w_num + 1);
  }

  param.W = &w;
  param.Ids = &ids;
  param.Out = &out;
  param.padding_idx = padding_idx;
  lookup_table.SetParam(param);
  lookup_table.Run();
  for (int i = 0; i < out_num; i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(lookup_table, kX86, kFloat, kNCHW, def);
