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

#pragma once

#include <algorithm>
#include <type_traits>

#include "lite/backends/x86/fluid/hostdevice.h"
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace fluid {

// Transform applys a unary or a binary functor on each element in a
// range defined by a pair of iterators.
//
// - The specialization for CPU calls std::transform.
// - The specialization for CUDA calls thrust::tranform.
//
// NOTE: We need to define InputIter and OutputIter defined as
//       different types, because the InputIter points op's inputs and
//       OutputIter pints to op's outputs.
//
// NOTE: We don't assume that InputIter to be const InputType* and
//       OutputIter to be OutputType*, because we might use a iterator
//       class, paddle::fluid::operators::RowwiseTRansformIterator.
template <lite::TargetType Target>
struct Transform {
  // The unary version.
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const lite::Context<Target>& context,
                  InputIter first,
                  InputIter last,
                  OutputIter result,
                  UnaryOperation op);

  // The binary version.
  template <typename InputIter1,
            typename InputIter2,
            typename OutputIter,
            typename BinaryOperation>
  void operator()(const lite::Context<Target>& context,
                  InputIter1 first1,
                  InputIter1 last1,
                  InputIter2 first2,
                  OutputIter result,
                  BinaryOperation op);
};

template <>
struct Transform<lite::TargetType::kX86> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const lite::X86Context& context,
                  InputIter first,
                  InputIter last,
                  OutputIter result,
                  UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename InputIter1,
            typename InputIter2,
            typename OutputIter,
            typename BinaryOperation>
  void operator()(const lite::X86Context& context,
                  InputIter1 first1,
                  InputIter1 last1,
                  InputIter2 first2,
                  OutputIter result,
                  BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};

}  // namespace fluid
}  // namespace lite
}  // namespace paddle
