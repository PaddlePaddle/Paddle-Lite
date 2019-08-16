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

#ifdef SUM_OP
#pragma once

#include <vector>
#include "operators/math/selected_rows_functor.h"

namespace paddle_mobile {
namespace operators {

using LoDTensorArray = std::vector<LoDTensor>;

template <typename P>
void SumCompute(const SumParam<CPU> &param) {
  auto inputsvars = param.InputsVars();
  int N = inputsvars.size();
  auto *outvar = param.OutVar();

  bool in_place = outvar == inputsvars[0];
  if (outvar->IsType<framework::LoDTensor>()) {
    auto *out = outvar->GetMutable<LoDTensor>();
    if (!in_place) {
      out->mutable_data<float>();
    }
    auto *outptr = out->data<float>();
    // auto result = Flatten(*out);

    if (!in_place) {
      std::fill(out->data<float>(), out->data<float>() + out->numel(), 0);
    }
    math::SelectedRowsAddToTensor<float> functor;
    for (int i = in_place ? 1 : 0; i < N; i++) {
      if (inputsvars[i]->IsType<framework::LoDTensor>()) {
        auto *in_t = inputsvars[i]->Get<framework::LoDTensor>();
        auto *inptr = in_t->data<float>();
        if (in_t->numel() == 0) {
          continue;
        }
        for (int j = 0; j < out->numel(); ++j) {
          outptr[j] = outptr[j] + inptr[j];
        }

      } else if (inputsvars[i]->IsType<framework::SelectedRows>()) {
        auto *in_t = inputsvars[i]->Get<framework::SelectedRows>();
        functor(*in_t, out);
      } else {
        PADDLE_MOBILE_THROW_EXCEPTION(
            "Variable type must be LoDTensor/SelectedRows.");
      }
    }

  } else if (outvar->IsType<framework::SelectedRows>()) {
    std::unique_ptr<framework::SelectedRows> in0;
    if (in_place) {
      // If is in_place, we store the input[0] to in0
      auto *in_sel0 = inputsvars[0]->Get<framework::SelectedRows>();
      auto &rows = in_sel0->rows();
      in0.reset(new framework::SelectedRows(rows, in_sel0->height()));
      in0->mutable_value()->ShareDataWith(in_sel0->value());
    }

    auto get_selected_row = [&](size_t i) -> const framework::SelectedRows & {
      if (i == 0 && in0) {
        return *in0.get();
      } else {
        return *(inputsvars[i]->Get<framework::SelectedRows>());
      }
    };

    auto *out = outvar->GetMutable<framework::SelectedRows>();
    out->mutable_rows()->clear();
    auto *out_value = out->mutable_value();

    // Runtime InferShape
    size_t first_dim = 0;
    for (int i = 0; i < N; i++) {
      auto &sel_row = get_selected_row(i);
      first_dim += sel_row.rows().size();
    }
    auto in_dim = framework::vectorize(get_selected_row(N - 1).value().dims());
    in_dim[0] = static_cast<int64_t>(first_dim);

    out_value->Resize(framework::make_ddim(in_dim));

    // if all the input sparse vars are empty, no need to
    // merge these vars.
    if (first_dim == 0UL) {
      return;
    }
    out_value->mutable_data<float>();
    math::SelectedRowsAddTo<float> functor;

    int64_t offset = 0;
    for (int i = 0; i < N; i++) {
      auto &sel_row = get_selected_row(i);
      if (sel_row.rows().size() == 0) {
        continue;
      }
      PADDLE_MOBILE_ENFORCE(out->height() == sel_row.height(),
                            "seletrows height != outheight");
      functor(sel_row, offset, out);
      offset += sel_row.value().numel();
    }
  } else if (outvar->IsType<LoDTensorArray>()) {
    auto &out_array = *outvar->GetMutable<LoDTensorArray>();
    for (size_t i = in_place ? 1 : 0; i < inputsvars.size(); ++i) {
      PADDLE_MOBILE_ENFORCE(inputsvars[i]->IsType<LoDTensorArray>(),
                            "Only support all inputs are TensorArray");
      auto *in_array = inputsvars[i]->Get<LoDTensorArray>();

      for (size_t i = 0; i < in_array->size(); ++i) {
        if ((*in_array)[i].numel() != 0) {
          if (i >= out_array.size()) {
            out_array.resize(i + 1);
          }
          if (out_array[i].numel() == 0) {
            framework::TensorCopy((*in_array)[i], &out_array[i]);
            out_array[i].set_lod((*in_array)[i].lod());
          } else {
            PADDLE_MOBILE_ENFORCE(out_array[i].lod() == (*in_array)[i].lod(),
                                  "outLod != inLod");
            auto *inptr = (*in_array)[i].data<float>();
            auto *outptr = out_array[i].data<float>();

            for (int j = 0; j < (*in_array)[i].numel(); ++j) {
              outptr[j] = inptr[j] + outptr[j];
            }
          }
        }
      }
    }
  } else {
    PADDLE_MOBILE_THROW_EXCEPTION(
        "Unexpected branch, output variable type is %d", outvar->Type());
  }
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
