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

#include "lite/backends/loongarch/math/math_function.h"

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <vector>
#include "lite/backends/loongarch/fluid/data_type.h"
#include "lite/backends/loongarch/fluid/float16.h"
#include "lite/backends/loongarch/math/math_function_impl.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template struct SetConstant<lite::TargetType::kLoongArch, lite::fluid::float16>;
template struct SetConstant<lite::TargetType::kLoongArch, float>;
template struct SetConstant<lite::TargetType::kLoongArch, double>;
template struct SetConstant<lite::TargetType::kLoongArch, int>;
template struct SetConstant<lite::TargetType::kLoongArch, int64_t>;
template struct SetConstant<lite::TargetType::kLoongArch, bool>;
template struct SetConstant<lite::TargetType::kLoongArch, uint8_t>;

#define DEFINE_CPU_TRANS(RANK)                                      \
  template struct Transpose<lite::TargetType::kLoongArch,                 \
                            lite::fluid::float16,                   \
                            RANK>;                                  \
  template struct Transpose<lite::TargetType::kLoongArch, float, RANK>;   \
  template struct Transpose<lite::TargetType::kLoongArch, double, RANK>;  \
  template struct Transpose<lite::TargetType::kLoongArch, int, RANK>;     \
  template struct Transpose<lite::TargetType::kLoongArch, int64_t, RANK>; \
  template struct Transpose<lite::TargetType::kLoongArch, bool, RANK>;    \
  template struct Transpose<lite::TargetType::kLoongArch, int16_t, RANK>; \
  template struct Transpose<lite::TargetType::kLoongArch, uint8_t, RANK>; \
  template struct Transpose<lite::TargetType::kLoongArch, int8_t, RANK>;

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
    auto* begin = tensor_->template mutable_data<T>(lite::TargetType::kLoongArch);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  lite::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<lite::TargetType::kLoongArch>(
    const lite::Context<lite::TargetType::kLoongArch>& context,
    lite::Tensor* tensor,
    float value) {
  // lite::VisitDataType(tensor->type(), TensorSetConstantCPU(tensor, value));
  TensorSetConstantCPU(tensor, value).apply<float>();
}

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
struct RowwiseAdd<lite::TargetType::kLoongArch, T> {
  void operator()(const lite::Context<lite::TargetType::kLoongArch>& context,
                  const lite::Tensor& input,
                  const lite::Tensor& vector,
                  lite::Tensor* output) {
    const auto& in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    CHECK_EQ(vector.numel(), size);
    CHECK_EQ(output->dims(), in_dims);

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

template struct RowwiseAdd<lite::TargetType::kLoongArch, float>;
template struct RowwiseAdd<lite::TargetType::kLoongArch, double>;

template struct ColwiseSum<lite::TargetType::kLoongArch, float>;
template struct ColwiseSum<lite::TargetType::kLoongArch, double>;
template struct ColwiseSum<lite::TargetType::kLoongArch, int>;
template struct ColwiseSum<lite::TargetType::kLoongArch, int64_t>;

template struct RowwiseSum<lite::TargetType::kLoongArch, float>;
template struct RowwiseSum<lite::TargetType::kLoongArch, double>;

template struct RowwiseMean<lite::TargetType::kLoongArch, float>;
template struct RowwiseMean<lite::TargetType::kLoongArch, double>;

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
