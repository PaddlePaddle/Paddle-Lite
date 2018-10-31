/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef LOOKUP_OP
#pragma once

#include <vector>
#include "framework/ddim.h"
#include "operators/op_param.h"

constexpr int64_t kNoPadding = -1;

namespace paddle_mobile {
namespace operators {

template <typename P>
void LookupCompute(const LookupParam<CPU> &param) {
  auto *ids_t = param.InputIds();
  auto *table_t = param.InputW();
  auto *output_t = param.Out();
  int64_t padding_idx = param.PaddingIdx();
  const framework::DDim &table_dim = table_t->dims();
  int64_t ids_numel;
  const auto *ids = ids_t->data<int64_t>();
  ids_numel = ids_t->numel();
  int64_t row_number = table_t->dims()[0];
  int64_t row_width = table_t->dims()[1];
  auto *table = table_t->data<float>();
  auto *output = output_t->mutable_data<float>();
  for (int64_t i = 0; i < ids_numel; ++i) {
    if (padding_idx != kNoPadding && ids[i] == padding_idx) {
      memset(output + i * row_width, 0, row_width * sizeof(float));
    } else {
      PADDLE_MOBILE_ENFORCE(ids[i] < row_number,
                            "look uptable ids[i] <row_number check failed");
      PADDLE_MOBILE_ENFORCE(ids[i] >= 0,
                            "lookuptable ids[i] >= 0 check failed");

      memcpy(output + i * row_width, table + ids[i] * row_width,
             row_width * sizeof(float));
    }
  }
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
