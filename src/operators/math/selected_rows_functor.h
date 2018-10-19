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

#pragma once
#include "framework/selected_rows.h"

#define INLINE_FOR2(sizei, sizej)     \
  for (int64_t i = 0; i < sizei; i++) \
    for (int64_t j = 0; j < sizej; j++)

namespace paddle_mobile {
namespace operators {
namespace math {

// SelectedRows + SelectedRows will simplely concat value and rows.
// The real computation happens in dealing with LoDTensor.
// template <typename T>
// struct SelectedRowsAdd {
//  void operator()(
//                  const framework::SelectedRows& input1,
//                  const framework::SelectedRows& input2,
//                  framework::SelectedRows* output);
//};
//
// template <typename T>
// struct SelectedRowsAddTensor {
//  void operator()(
//                  const framework::SelectedRows& input1,
//                  const framework::Tensor& input2, framework::Tensor* output);
//};

// input2 = input1 + input2
template <typename T>
struct SelectedRowsAddTo {
  void operator()(const framework::SelectedRows& input1,
                  const int64_t input2_offset,
                  framework::SelectedRows* input2) {
    auto in1_height = input1.height();
    PADDLE_MOBILE_ENFORCE(in1_height == input2->height(), "height error");

    auto& in1_rows = input1.rows();
    auto& in2_rows = *(input2->mutable_rows());

    auto& in1_value = input1.value();
    auto* in2_value = input2->mutable_value();

    // concat rows
    in2_rows.Extend(in1_rows.begin(), in1_rows.end());

    //    auto in1_place = input1.place();
    //    PADDLE_ENFORCE(platform::is_cpu_place(in1_place));
    //    auto in2_place = input2->place();
    //    PADDLE_ENFORCE(platform::is_cpu_place(in2_place));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = in2_value->data<T>();
    memory::Copy(in2_data + input2_offset, in1_data,
                 in1_value.numel() * sizeof(T));
  }
};

// input2 = input1 + input2
template <typename T>
struct SelectedRowsAddToTensor {
  void operator()(const framework::SelectedRows& input1,
                  framework::Tensor* input2) {
    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    PADDLE_MOBILE_ENFORCE(in1_height == in2_dims[0], "height != dims[0]");

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_MOBILE_ENFORCE(in1_row_numel == input2->numel() / in1_height,
                          "row_numel error");

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }
  }
};

// namespace scatter {
//// functors for manuplating SelectedRows data
// template <typename T>
// struct MergeAdd {
//  // unary functor, merge by adding duplicated rows in
//  // the input SelectedRows object.
//  framework::SelectedRows operator()(
//                                     const framework::SelectedRows& input);
//};

// template <typename T>
// struct Add {
//  framework::SelectedRows operator()(
//                                     const framework::SelectedRows& input1,
//                                     const framework::SelectedRows& input2) {
//    framework::SelectedRows out;
//    out.set_rows(input1.rows());
//    out.set_height(input1.height());
//    out.mutable_value()->mutable_data<T>(input1.value().dims(),
//                                         );
//    auto e_out = framework::EigenVector<T>::Flatten(*(out.mutable_value()));
//    auto e_in1 = framework::EigenVector<T>::Flatten(input1.value());
//    auto e_in2 = framework::EigenVector<T>::Flatten(input2.value());
//    e_out.device(*context.eigen_device()) = e_in1 + e_in2;
//    return out;
//  }
//};

// template <typename T>
// struct Mul {
//  // multiply two SelectedRows
//  framework::SelectedRows operator()(
//                                     const framework::SelectedRows& input1,
//                                     const framework::SelectedRows& input2) {
//    framework::SelectedRows out;
//    out.set_rows(input1.rows());
//    out.set_height(input1.height());
//    out.mutable_value()->mutable_data<T>(input1.value().dims()
//                                         );
//    auto e_out = framework::EigenVector<T>::Flatten(*(out.mutable_value()));
//    auto e_in1 = framework::EigenVector<T>::Flatten(input1.value());
//    auto e_in2 = framework::EigenVector<T>::Flatten(input2.value());
//    e_out.device(*context.eigen_device()) = e_in1 * e_in2;
//    return out;
//  }
//  // multiply scalar to SelectedRows
//  framework::SelectedRows operator()(
//                                     const framework::SelectedRows& input1,
//                                     const T input2) {
//    framework::SelectedRows out;
//    out.set_rows(input1.rows());
//    out.set_height(input1.height());
//    out.mutable_value()->mutable_data<T>(input1.value().dims(),
//                                         );
//    auto e_out = framework::EigenVector<T>::Flatten(*(out.mutable_value()));
//    auto e_in1 = framework::EigenVector<T>::Flatten(input1.value());
//    e_out.device(*context.eigen_device()) = input2 * e_in1;
//    return out;
//  }
//};

enum class ScatterOps { ASSIGN, ADD, SUB, SUBBY, MUL, DIV, DIVBY };

// out = seleted_rows_in / tensor
template <typename T>
struct UpdateToTensor {
  void operator()(const ScatterOps& op, const framework::SelectedRows& input1,
                  framework::Tensor* input2);
};

// namespace scatter
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
