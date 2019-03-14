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

#ifdef SEQUENCE_SOFTMAX_OP

#include "framework/lod_tensor.h"
#include "operators/kernel/sequence_kernels.h"
#include "operators/math/softmax.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
class SequenceSoftmaxKernel<CPU, T>
    : public framework::OpKernelBase<CPU, SoftmaxParam<CPU>> {
 public:
  bool Init(SoftmaxParam<CPU> *param) { return true; }

  void Compute(const SoftmaxParam<CPU> &param) {
    param.Out()->mutable_data<float>();
    const framework::LoDTensor *input = param.InputX();
    framework::LoDTensor *output = param.Out();
    math::SequenceSoftmaxFuntor<CPU, T> sequence_softmax;
    sequence_softmax(input, output);
  }
};

template class SequenceSoftmaxKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif  // SEQUENCE_SOFTMAX_OP
