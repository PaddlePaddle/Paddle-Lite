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

#include <algorithm>

namespace paddle_mobile {
namespace operators {
namespace math {

// Transform applys a unary or a binary functor on each element in a
// range defined by a pair of iterators.
//
// - The specialization for CPU calls std::transform.
// - The specialization for CUDA calls thrust::tranform.
//
// NOTE: We need to define InputIter and OutputIter defined as
//       different types, because the InputIter points op's inputs
//       and
//       OutputIter pints to op's outputs.
//
// NOTE: We don't assume that InputIter to be const InputType* and
//       OutputIter to be OutputType*, because we might use a
//       iterator
//       class, paddle::fluid::operators::RowwiseTRansformIterator.

struct Transform {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(InputIter first, InputIter last, OutputIter result,
                  UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(InputIter1 first1, InputIter1 last1, InputIter2 first2,
                  OutputIter result, BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
