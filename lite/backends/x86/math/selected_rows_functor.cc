/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <map>
#include <set>

#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/selected_rows_functor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <typename T>
struct SelectedRowsAdd<lite::TargetType::kX86, T> {
  void operator()(const lite::X86Context& context,
                  const fluid::SelectedRows& input1,
                  const fluid::SelectedRows& input2,
                  fluid::SelectedRows* output) {
    auto in1_height = input1.height();
    CHECK_EQ(in1_height, input2.height());
    output->set_height(in1_height);

    auto& in1_rows = input1.rows();
    auto& in2_rows = input2.rows();
    std::vector<int64_t> out_rows;
    out_rows.reserve(in1_rows.size() + in2_rows.size());

    // concat rows
    out_rows.insert(out_rows.end(), in1_rows.begin(), in1_rows.end());
    out_rows.insert(out_rows.end(), in2_rows.begin(), in2_rows.end());
    output->set_rows(out_rows);

    auto* out_value = output->mutable_value();
    auto& in1_value = input1.value();
    auto& in2_value = input2.value();

    auto in1_row_numel = in1_value.numel() / in1_rows.size();
    CHECK_EQ(in1_row_numel, in2_value.numel() / in2_rows.size());
    CHECK_EQ(in1_row_numel, out_value->numel() / out_rows.size());

    auto* out_data = out_value->template mutable_data<T>();
    auto* in1_data = in1_value.data<T>();
    std::copy_n(in1_data, in1_value.numel(), out_data);

    auto* in2_data = in2_value.data<T>();
    std::copy_n(in2_data, in2_value.numel(), out_data + in1_value.numel());
  }
};

template struct SelectedRowsAdd<lite::TargetType::kX86, float>;
template struct SelectedRowsAdd<lite::TargetType::kX86, double>;

template <typename T>
struct SelectedRowsAddTensor<lite::TargetType::kX86, T> {
  void operator()(const lite::X86Context& context,
                  const fluid::SelectedRows& input1,
                  const lite::Tensor& input2,
                  lite::Tensor* output) {
    auto in1_height = input1.height();
    auto in2_dims = input2.dims();
    auto out_dims = output->dims();
    CHECK_EQ(in1_height, in2_dims[0]);
    CHECK_EQ(in1_height, out_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    CHECK_EQ(in1_row_numel, input2.numel() / in1_height);
    CHECK_EQ(in1_row_numel, output->numel() / in1_height);

    SetConstant<lite::TargetType::kX86, T> functor;
    functor(context, output, 0.0);

    auto* in1_data = in1_value.data<T>();
    auto* out_data = output->template mutable_data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        out_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }

    auto out_eigen = fluid::EigenVector<T>::Flatten(*output);
    auto in2_eigen = fluid::EigenVector<T>::Flatten(input2);
    out_eigen.device(lite::fluid::EigenDeviceType<TARGET(kX86)>()) =
        out_eigen + in2_eigen;
  }
};

template struct SelectedRowsAddTensor<lite::TargetType::kX86, float>;
template struct SelectedRowsAddTensor<lite::TargetType::kX86, double>;

template <typename T>
struct SelectedRowsAddTo<lite::TargetType::kX86, T> {
  void operator()(const lite::X86Context& context,
                  const fluid::SelectedRows& input1,
                  const int64_t input2_offset,
                  fluid::SelectedRows* input2) {
    auto in1_height = input1.height();
    CHECK_EQ(in1_height, input2->height());

    auto& in1_rows = input1.rows();
    auto& in2_rows = *(input2->mutable_rows());

    auto& in1_value = input1.value();
    auto* in2_value = input2->mutable_value();

    // concat rows
    in2_rows.reserve(in2_rows.size() +
                     size_t(in1_rows.end() - in1_rows.begin()));
    in2_rows.insert(in2_rows.end(), in1_rows.begin(), in1_rows.end());

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = in2_value->template mutable_data<T>();
    std::copy_n(in1_data, in1_value.numel(), in2_data + input2_offset);
  }
};

template struct SelectedRowsAddTo<lite::TargetType::kX86, float>;
template struct SelectedRowsAddTo<lite::TargetType::kX86, double>;
template struct SelectedRowsAddTo<lite::TargetType::kX86, int>;
template struct SelectedRowsAddTo<lite::TargetType::kX86, int64_t>;

template <typename T>
struct SelectedRowsSumTo<lite::TargetType::kX86, T> {
  void operator()(const lite::X86Context& context,
                  const std::vector<fluid::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  fluid::SelectedRows* input2) {
    // Ensure all selected rows have the same height
    size_t size = 0u;
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      auto& in_rows = (*iter)->rows();
      size += in_rows.end() - in_rows.begin();
      auto in1_height = (*iter)->height();
      CHECK_EQ(in1_height, input2->height());
    }
    // concat rows
    std::vector<int64_t> in2_rows;
    in2_rows.reserve(in2_rows.size() + size);
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      const std::vector<int64_t>& in_rows = (*iter)->rows();
      in2_rows.insert(in2_rows.end(), in_rows.begin(), in_rows.end());
    }
    input2->set_rows(in2_rows);

    auto* in2_value = input2->mutable_value();
    T* in2_data = in2_value->template mutable_data<T>();
    auto blas = math::GetBlas<lite::TargetType::kX86, T>(context);
    size_t offset = 0u;
    for (size_t i = 0u; i != input1.size(); ++i) {
      auto& in_value = input1[i]->value();
      const T* in_data = in_value.data<T>();
      offset += input2_offsets[i];
      blas.VCOPY(in_value.numel(), in_data, in2_data + offset);
    }
  }
};

template struct SelectedRowsSumTo<lite::TargetType::kX86, float>;
template struct SelectedRowsSumTo<lite::TargetType::kX86, double>;

template <typename T>
struct SelectedRowsAddToTensor<lite::TargetType::kX86, T> {
  void operator()(const lite::X86Context& context,
                  const fluid::SelectedRows& input1,
                  lite::Tensor* input2) {
    CHECK(input1.rows().size() != 0) << "input selected rows is empty!";

    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    CHECK_EQ(in1_height, in2_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    CHECK_EQ(in1_row_numel, input2->numel() / in1_height);

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->template mutable_data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }
  }
};

template struct SelectedRowsAddToTensor<lite::TargetType::kX86, float>;
template struct SelectedRowsAddToTensor<lite::TargetType::kX86, double>;
template struct SelectedRowsAddToTensor<lite::TargetType::kX86, int>;
template struct SelectedRowsAddToTensor<lite::TargetType::kX86, int64_t>;

// This is a separated namespace for manipulate SelectedRows typed
// data. Like merge duplicated rows, adding two SelectedRows etc.
//
// Another group of functors is called "scatter updates", which means
// use SelectedRows to update a dense tensor with different Ops, like
// add or mul.
namespace scatter {

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, lite::X86Context>::value>::type
elementwise_add_to(const DeviceContext& ctx,
                   BlasT<lite::TargetType::kX86, T>* blas,
                   size_t data_len,
                   const T* in,
                   T* out) {
  blas->AXPY(data_len, 1., in, out);
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, lite::X86Context>::value>::type
elementwise_add_to(const DeviceContext& ctx,
                   BlasT<lite::TargetType::kX86, T>* blas,
                   size_t data_len,
                   const T* in,
                   T* out) {
  for (size_t i = 0; i < data_len; i++) {
    out[i] += in[i];
  }
}

template <typename T>
struct MergeAdd<lite::TargetType::kX86, T> {
  fluid::SelectedRows operator()(const lite::X86Context& context,
                                 const fluid::SelectedRows& input,
                                 const bool sorted_result = false) {
    fluid::SelectedRows out;
    (*this)(context, input, &out, sorted_result);
    return out;
  }

  void operator()(const lite::X86Context& context,
                  const fluid::SelectedRows& input,
                  fluid::SelectedRows* output,
                  const bool sorted_result = false) {
    std::vector<const fluid::SelectedRows*> inputs;
    inputs.push_back(&input);
    (*this)(context, inputs, output, sorted_result);
  }

  void operator()(const lite::X86Context& context,
                  const std::vector<const fluid::SelectedRows*>& inputs,
                  fluid::SelectedRows* output,
                  const bool sorted_result = false) {
    if (inputs.size() == 0) {
      VLOG(3) << "no input! return";
      return;
    }
    const fluid::SelectedRows* has_value_input = nullptr;
    for (auto* in : inputs) {
      if (in->rows().size() > 0) {
        has_value_input = in;
        break;
      }
    }
    if (has_value_input == nullptr) {
      VLOG(3) << "no input has value! just return";
      return;
    }
    auto input_width = has_value_input->value().dims()[1];
    auto input_height = has_value_input->height();
    fluid::SelectedRows& out = *output;
    std::set<int64_t> merged_row_set;
    size_t row_num = 0;
    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      CHECK_EQ(input_width, input->value().dims()[1])
          << "all input should have same "
             "dimension except for the first one";
      CHECK_EQ(input_height, input->height())
          << "all input should have same height";
      row_num += input->rows().size();
      merged_row_set.insert(input->rows().begin(), input->rows().end());
    }

    out.set_height(input_height);
    lite::DDim dims(std::vector<int64_t>(
        {static_cast<int64_t>(merged_row_set.size()), input_width}));
    out.mutable_value()->Resize(dims);
    auto* out_data = out.mutable_value()->template mutable_data<T>();

    if (merged_row_set.size() == row_num && !sorted_result) {
      // no duplicated ids, just concat the result together
      std::vector<int64_t> merge_rows;
      merge_rows.reserve(row_num);
      // concat rows
      for (auto* in : inputs) {
        merge_rows.insert(
            merge_rows.end(), in->rows().begin(), in->rows().end());
      }
      out.set_rows(merge_rows);
      int64_t copied_numel = 0;
      for (auto* in : inputs) {
        auto* in_data = in->value().data<T>();
        auto in_numel = in->value().numel();
        std::copy_n(in_data, in_numel, out_data + copied_numel);
        copied_numel += in_numel;
      }
    } else {
      std::vector<int64_t> merge_rows(merged_row_set.begin(),
                                      merged_row_set.end());

      if (sorted_result) {
        std::stable_sort(merge_rows.begin(), merge_rows.end());
      }

      out.set_rows(merge_rows);
      math::SetConstant<lite::TargetType::kX86, T> constant_functor;
      constant_functor(context, out.mutable_value(), 0.0);

      std::map<int64_t, size_t> rows_to_id;
      for (size_t i = 0; i < merge_rows.size(); ++i) {
        rows_to_id[merge_rows[i]] = i;
      }

      auto blas = math::GetBlas<lite::TargetType::kX86, T>(context);
      for (auto* input : inputs) {
        if (input->rows().size() == 0) {
          continue;
        }
        auto* input_data = input->value().data<T>();
        auto& input_rows = input->rows();

        for (size_t i = 0; i < input_rows.size(); i++) {
          size_t out_i = rows_to_id[input_rows[i]];
          elementwise_add_to<lite::X86Context, T>(
              context,
              &blas,
              static_cast<size_t>(input_width),
              &input_data[i * input_width],
              &out_data[out_i * input_width]);
        }
      }
    }
  }
};

template struct MergeAdd<lite::TargetType::kX86, int>;
template struct MergeAdd<lite::TargetType::kX86, int64_t>;
template struct MergeAdd<lite::TargetType::kX86, float>;
template struct MergeAdd<lite::TargetType::kX86, double>;

template <typename T>
struct UpdateToTensor<lite::TargetType::kX86, T> {
  void operator()(const lite::X86Context& context,
                  const ScatterOps& op,
                  const fluid::SelectedRows& input1,
                  lite::Tensor* input2) {
    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    CHECK_EQ(in1_height, in2_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    CHECK_EQ(in1_row_numel, input2->numel() / in1_height);

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->template data<T>();

    // FIXME(typhoonzero): use macro fix the below messy code.
    switch (op) {
      case ScatterOps::ASSIGN:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::ADD:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::SUB:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] -=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::SUBBY:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j] -
            input2_data[in1_rows[i] * in1_row_numel + j];
        break;
      case ScatterOps::MUL:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] *=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::DIV:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] /=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::DIVBY:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j] /
            input2_data[in1_rows[i] * in1_row_numel + j];
        break;
    }
  }
};

}  // namespace scatter
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
