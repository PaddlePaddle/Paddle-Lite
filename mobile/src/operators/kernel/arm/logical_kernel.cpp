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

#include "operators/kernel/logical_kernel.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
struct LogicalAndFunctor {
  bool operator()(const T& a, const T& b) const { return a && b; }
};

template <typename T>
struct LogicalOrFunctor {
  bool operator()(const T& a, const T& b) const { return a || b; }
};

template <typename T>
struct LogicalNotFunctor {
  bool operator()(const T& a) const { return !a; }
};

template <typename T>
struct LogicalXorFunctor {
  bool operator()(const T& a, const T& b) const {
    return (a || b) && !(a && b);
  }
};

template <typename T, typename Functor>
void UnaryLogicalCompute(const Tensor* inputX, Tensor* output) {
  Functor func;
  std::transform(inputX->data<T>(), inputX->data<T>() + inputX->numel(),
                 output->data<T>(), func);
}

template <typename T, typename Functor>
void BinaryLogicalCompute(const Tensor* inputX, const Tensor* inputY,
                          Tensor* output) {
  Functor func;
  std::transform(inputX->data<T>(), inputX->data<T>() + inputX->numel(),
                 inputY->data<T>(), output->data<T>(), func);
}

#ifdef LOGICAL_AND_OP
template <>
bool LogicalAndKernel<CPU, float>::Init(LogicalBinaryParam<CPU>* param) {
  return true;
}

template <>
void LogicalAndKernel<CPU, float>::Compute(
    const LogicalBinaryParam<CPU>& param) {
  auto* inputX = param.InputX();
  auto* inputY = param.InputY();
  auto* out = param.Out();
  out->mutable_data<bool>();
  BinaryLogicalCompute<bool, LogicalAndFunctor<bool>>(inputX, inputY, out);
}
#endif

#ifdef LOGICAL_OR_OP
template <>
bool LogicalOrKernel<CPU, float>::Init(LogicalBinaryParam<CPU>* param) {
  return true;
}

template <>
void LogicalOrKernel<CPU, float>::Compute(
    const LogicalBinaryParam<CPU>& param) {
  auto* inputX = param.InputX();
  auto* inputY = param.InputY();
  auto* out = param.Out();
  out->mutable_data<bool>();
  BinaryLogicalCompute<bool, LogicalOrFunctor<bool>>(inputX, inputY, out);
}
#endif

#ifdef LOGICAL_NOT_OP
template <>
bool LogicalNotKernel<CPU, float>::Init(LogicalUnaryParam<CPU>* param) {
  return true;
}

template <>
void LogicalNotKernel<CPU, float>::Compute(
    const LogicalUnaryParam<CPU>& param) {
  auto* inputX = param.InputX();
  auto* out = param.Out();
  out->mutable_data<bool>();
  UnaryLogicalCompute<bool, LogicalNotFunctor<bool>>(inputX, out);
}
#endif

#ifdef LOGICAL_XOR_OP
template <>
bool LogicalXorKernel<CPU, float>::Init(LogicalBinaryParam<CPU>* param) {
  return true;
}

template <>
void LogicalXorKernel<CPU, float>::Compute(
    const LogicalBinaryParam<CPU>& param) {
  auto* inputX = param.InputX();
  auto* inputY = param.InputY();
  auto* out = param.Out();
  out->mutable_data<bool>();
  BinaryLogicalCompute<bool, LogicalXorFunctor<bool>>(inputX, inputY, out);
}
#endif

}  // namespace operators
}  // namespace paddle_mobile
