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

#include "lite/backends/x86/math/unpooling.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {
template <typename T>
class Unpool2dMaxFunctor<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& input,
                  const lite::Tensor& indices,
                  lite::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = output->template mutable_data<T>(lite::TargetType::kX86);
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];
          CHECK(index < output_feasize) << "err index in unpooling!";
          output_data[index] = input_data[i];
        }
        input_data += input_feasize;
        indices_data += input_feasize;
        output_data += output_feasize;
      }
    }
  }
};
template <class T>
class Unpool2dMaxGradFunctor<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& input,
                  const lite::Tensor& indices,
                  const lite::Tensor& output,
                  const lite::Tensor& output_grad,
                  lite::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    int input_feasize = input_height * input_width;
    int output_feasize = output_height * output_width;
    const int* indices_data = indices.data<int>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data =
        input_grad->template mutable_data<T>(lite::TargetType::kX86);

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < output_channels; ++c) {
        for (int i = 0; i < input_feasize; ++i) {
          int index = indices_data[i];
          CHECK(index < output_feasize) << "err index in unpooling!";
          input_grad_data[i] = output_grad_data[index];
        }
        input_grad_data += input_feasize;
        indices_data += input_feasize;
        output_grad_data += output_feasize;
      }
    }
  }
};
template class Unpool2dMaxGradFunctor<lite::TargetType::kX86, float>;
template class Unpool2dMaxGradFunctor<lite::TargetType::kX86, double>;
template class Unpool2dMaxFunctor<lite::TargetType::kX86, float>;
template class Unpool2dMaxFunctor<lite::TargetType::kX86, double>;
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
