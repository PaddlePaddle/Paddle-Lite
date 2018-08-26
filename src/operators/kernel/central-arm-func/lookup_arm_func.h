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
#include "operators/op_param.h"

constexpr int64_t kNoPadding = -1;

namespace paddle_mobile {
namespace operators {

// vector<int> pos;
// template <typename T>
// void TransposeFunc(const int numel, const T* input, const vector<int> axis,
//                    const vector<int> old_strides, const vector<int>
//                    new_strides, T* output) {
//   for (int i = 0; i < numel; ++i) {
//     int old_idx = 0;
//     int idx = i;
//     for (int j = 0; j < axis.size(); ++j) {
//       int order = axis[j];
//       old_idx += (idx / new_strides[j]) * old_strides[order];
//       idx %= new_strides[j];
//     }
//     output[i] = input[old_idx];
//   }
// }

template <typename P>
void LookupCompute(const LookupParam<CPU> &param) {
  //  const auto* input_x = param.InputX();
  //  const auto input_x_dims = input_x->dims();
  //  auto* out = param.Out();
  //  const auto axis = param.Axis();
  //  const auto* input_x_data = input_x->data<float>();
  //  auto* out_data = out->mutable_data<float>();
  //
  //  size_t ndim = axis.size();
  //  std::vector<int> xdim(ndim);
  //  std::vector<int> xstride(ndim);
  //  std::vector<int> xout(ndim);
  //  for (int i = 0; i < ndim; i++) {
  //    int j = ndim - 1 - i;
  //    xdim[j] = input_x_dims[axis[i]];
  //    xstride[j] = 1;
  //    for (int k = axis[i] + 1; k < ndim; k++) {
  //      xstride[j] *= input_x_dims[k];
  //    }
  //    xout[j] = xstride[j] * xdim[j];
  //  }
  //
  //  auto numel = input_x->numel();
  //  size_t pind = 0;
  //  std::vector<int> ind(ndim);
  //  for (int i = 0; i < numel; i++) {
  //    out_data[i] = input_x_data[pind];
  //    ind[0]++;
  //    pind += xstride[0];
  //    for (int j = 0; j < ndim - 1; j++) {
  //      if (ind[j] == xdim[j]) {
  //        ind[j + 1]++;
  //        ind[j] = 0;
  //        pind += xstride[j + 1];
  //        pind -= xout[j];
  //      } else {
  //        break;
  //      }
  //    }
  //  }
  /*  auto *ids_t = param.InputIds();
    auto *output_t = param.Out();
    auto *table_var = param.InputW();
    auto padding_idx = param.PaddingIdx();

    //    auto *ids_t = context.Input<LoDTensor>("Ids");      // int tensor
    //    auto *output_t = context.Output<LoDTensor>("Out");  // float tensor
    //    auto *table_var = context.InputVar("W");
    //
    //    int64_t padding_idx = context.Attr<int64_t>("padding_idx");



    int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
    int64_t ids_numel = ids_t->numel();
    DLOG << "table_var->type()" << table_var->type();
    if (true) {
      int64_t row_number = table_var->dims()[0];
      int64_t row_width = table_var->dims()[1];

      auto *table = table_var->data<LoDTensor>();
      auto *output = output_t->mutable_data<float>();

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(float));
        } else {
          PADDLE_MOBILE_ENFORCE(ids[i] < row_number,
                                "LookupCompute ,ids[i]<row_number");
          PADDLE_MOBILE_ENFORCE(ids[i] >= 0, "LookupCompute ,iids[i]>=0");
          memcpy(output + i * row_width, table + ids[i] * row_width,
                 row_width * sizeof(float));
        }
      }
    } else {
      PADDLE_MOBILE_ENFORCE(false,
                            "LookupCompute got unsupported table_var type!")
    }


      */

  //    } else if (table_var->IsType<SelectedRows>()) {
  //        const auto &table_t = table_var->Get<SelectedRows>();
  //        int64_t row_width = table_t.value().dims()[1];
  //        const auto *table = table_t.value().data<T>();
  //        auto *output = output_t->mutable_data<T>(context.GetPlace());
  //
  //        for (int64_t i = 0; i < ids_numel; ++i) {
  //            if (padding_idx != kNoPadding && ids[i] == padding_idx) {
  //                memset(output + i * row_width, 0, row_width * sizeof(T));
  //            } else {
  //                PADDLE_ENFORCE_GE(ids[i], 0);
  //                auto id_index = table_t.Index(ids[i]);
  //                PADDLE_ENFORCE_GE(id_index, 0, "the input key should be
  //                exists."); memcpy(output + i * row_width, table + id_index *
  //                row_width,
  //                       row_width * sizeof(T));
  //            }
  //        }
  //    }
}

}  // namespace operators

}  // namespace paddle_mobile

#endif
