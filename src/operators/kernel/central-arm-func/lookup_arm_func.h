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
  DLOG << "LookupCompute in";
  auto *ids_t = param.InputIds();
  auto *table_t = param.InputW();
  DLOG << "LookupCompute in2";

  Tensor *output_t = param.Out();
  int64_t padding_idx = param.PaddingIdx();
  DLOG << "LookupCompute in3";

  DLOG << "padding_idx: " << padding_idx;

  const framework::DDim &table_dim = table_t->dims();

  //  auto *table_var = context.InputVar("W");
  //  auto *ids_t = context.InputVar("Ids");
  //  Tensor *output_t = context.Output<Tensor>("Out");
  //  int64_t padding_idx = context.Attr<int64_t>("padding_idx");

  //  if (table_var->IsType<LoDTensor>()) {
  //    table_dim = context.Input<LoDTensor>("W")->dims();
  //  } else if (table_var->IsType<SelectedRows>()) {
  //    auto *table_t = context.Input<SelectedRows>("W");
  //    table_dim = table_t->value().dims();
  //  } else {
  //    PADDLE_THROW(
  //            "The parameter W of a LookupTable "
  //                    "must be either LoDTensor or SelectedRows");
  //  }

  //   int64_t *ids;
  int64_t ids_numel;
  DLOG << "LookupCompute in4";

  // The type of Ids(Input) is SelectedRows or LoDTensor, when Ids's type
  // is LoDTensor, this tensor contains the ids to be looked up in W;
  // when Ids's type is SelectedRows, the rows of Ids contains the
  // ids to be looked up in W.
  //  if (ids_var->IsType<LoDTensor>()) {
  //    auto *ids_t = context.Input<LoDTensor>("Ids");

  const auto *ids = ids_t->data<int64_t>();
  DLOG << "LookupCompute in5";

  //    float d = ids[0];
  //  DLOG<<"(int64_t)d:  "<<(int64_t)d;
  ids_numel = ids_t->numel();
  DLOG << "ids_numel:  " << ids_numel;
  //
  //  } else if (ids_var->IsType<SelectedRows>()) {
  //    auto *ids_t = context.Input<SelectedRows>("Ids");
  //    ids = const_cast<int64_t *>(ids_t->rows().data());
  //    ids_numel = ids_t->rows().size();
  //    output_t->Resize({ids_numel, table_dim[1]});
  //  } else {
  //    PADDLE_THROW("Unsupported Variable Type of Ids");
  //  }

  //  if (table_var->IsType<LoDTensor>()) {
  //    auto *table_t = context.Input<LoDTensor>("W");
  int64_t row_number = table_t->dims()[0];
  int64_t row_width = table_t->dims()[1];
  DLOG << "LookupCompute in6";

  DLOG << "row_number: " << row_number;
  DLOG << "row_width: " << row_width;

  auto *table = table_t->data<float>();

  auto *output = output_t->mutable_data<float>();

  for (int64_t i = 0; i < ids_numel; ++i) {
    //   DLOG << "ids[" << i << "]" << ids[i];

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
    //  } else if (table_var->IsType<SelectedRows>()) {
    //    const auto &table_t = table_var->Get<SelectedRows>();
    //    int64_t row_width = table_t.value().dims()[1];
    //    const auto *table = table_t.value().data<T>();
    //    auto *output = output_t->mutable_data<T>(context.GetPlace());
    //
    //    for (int64_t i = 0; i < ids_numel; ++i) {
    //      if (padding_idx != kNoPadding && ids[i] == padding_idx) {
    //        memset(output + i * row_width, 0, row_width * sizeof(T));
    //      } else {
    //        PADDLE_ENFORCE_GE(ids[i], 0);
    //        auto id_index = table_t.Index(ids[i]);
    //        PADDLE_ENFORCE_GE(id_index, 0, "the input key should be exists.");
    //        memcpy(output + i * row_width, table + id_index * row_width,
    //               row_width * sizeof(T));
    //      }
    //    }
    //  }
    //}
  }
}
}  // namespace operators

}  // namespace paddle_mobile

#endif
