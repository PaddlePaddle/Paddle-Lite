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

#include "lite/backends/x86/math/math_function.h"

#ifdef PADDLE_WITH_MKLML
#include "lite/backends/x86/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <vector>
#include "lite/backends/x86/math/math_function_impl.h"
#include "lite/fluid/data_type.h"
#include "lite/fluid/float16.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template struct SetConstant<lite::TargetType::kX86, lite::fluid::float16>;
template struct SetConstant<lite::TargetType::kX86, float>;
template struct SetConstant<lite::TargetType::kX86, double>;
template struct SetConstant<lite::TargetType::kX86, int>;
template struct SetConstant<lite::TargetType::kX86, int64_t>;
template struct SetConstant<lite::TargetType::kX86, bool>;
template struct SetConstant<lite::TargetType::kX86, uint8_t>;

#define DEFINE_CPU_TRANS(RANK)                                      \
  template struct Transpose<lite::TargetType::kX86,                 \
                            lite::fluid::float16,                   \
                            RANK>;                                  \
  template struct Transpose<lite::TargetType::kX86, float, RANK>;   \
  template struct Transpose<lite::TargetType::kX86, double, RANK>;  \
  template struct Transpose<lite::TargetType::kX86, int, RANK>;     \
  template struct Transpose<lite::TargetType::kX86, int64_t, RANK>; \
  template struct Transpose<lite::TargetType::kX86, bool, RANK>;    \
  template struct Transpose<lite::TargetType::kX86, int16_t, RANK>; \
  template struct Transpose<lite::TargetType::kX86, uint8_t, RANK>; \
  template struct Transpose<lite::TargetType::kX86, int8_t, RANK>;

DEFINE_CPU_TRANS(1);
DEFINE_CPU_TRANS(2);
DEFINE_CPU_TRANS(3);
DEFINE_CPU_TRANS(4);
DEFINE_CPU_TRANS(5);
DEFINE_CPU_TRANS(6);

struct TensorSetConstantCPU {
  TensorSetConstantCPU(lite::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    auto* begin = tensor_->template mutable_data<T>(lite::TargetType::kX86);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  lite::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<lite::TargetType::kX86>(
    const lite::Context<lite::TargetType::kX86>& context,
    lite::Tensor* tensor,
    float value) {
  // lite::VisitDataType(tensor->type(), TensorSetConstantCPU(tensor, value));
  TensorSetConstantCPU(tensor, value).apply<float>();
}

// template <>
// void set_constant_with_place<platform::CUDAPinnedPlace>(
//    const platform::DeviceContext& context, framework::Tensor* tensor,
//    float value) {
//  framework::VisitDataType(tensor->type(), TensorSetConstantCPU(tensor,
//  value));
//}

template <lite::TargetType Target>
struct TensorSetConstantWithTarget /*: public boost::static_visitor<void>*/ {
  TensorSetConstantWithTarget(const lite::Context<Target>& context,
                              lite::Tensor* tensor,
                              float value)
      : context_(context), tensor_(tensor), value_(value) {}

  void operator()() const {
    set_constant_with_place<Target>(context_, tensor_, value_);
  }

  const lite::Context<Target>& context_;
  lite::Tensor* tensor_;
  float value_;
};

template <lite::TargetType Target>
void set_constant(const lite::Context<Target>& context,
                  lite::Tensor* tensor,
                  float value) {
  TensorSetConstantWithTarget<Target> func(context, tensor, value);
  func();
}

template <typename T>
struct RowwiseAdd<lite::TargetType::kX86, T> {
  void operator()(const lite::Context<lite::TargetType::kX86>& context,
                  const lite::Tensor& input,
                  const lite::Tensor& vector,
                  lite::Tensor* output) {
    const auto& in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(vector.numel(), size);
    PADDLE_ENFORCE_EQ(output->dims(), in_dims);

    const T* input_data = input.data<T>();
    const T* vector_data = vector.data<T>();
    T* output_data = output->template mutable_data<T>();
    for (int64_t i = 0; i < in_dims[0]; ++i) {
      for (int64_t j = 0; j < size; ++j) {
        output_data[i * size + j] = input_data[i * size + j] + vector_data[j];
      }
    }
  }
};

template struct RowwiseAdd<lite::TargetType::kX86, float>;
template struct RowwiseAdd<lite::TargetType::kX86, double>;

template struct ColwiseSum<lite::TargetType::kX86, float>;
template struct ColwiseSum<lite::TargetType::kX86, double>;
template struct ColwiseSum<lite::TargetType::kX86, int>;
template struct ColwiseSum<lite::TargetType::kX86, int64_t>;

template struct RowwiseSum<lite::TargetType::kX86, float>;
template struct RowwiseSum<lite::TargetType::kX86, double>;

template struct RowwiseMean<lite::TargetType::kX86, float>;
template struct RowwiseMean<lite::TargetType::kX86, double>;

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
