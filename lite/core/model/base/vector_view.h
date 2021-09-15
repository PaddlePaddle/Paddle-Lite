// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <type_traits>
#include <vector>
#include "lite/core/model/base/traits.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace vector_view {

template <typename T, typename U = void>
struct ElementTraits {
  typedef T element_type;
};

template <typename T, typename U>
struct VectorTraits;

template <typename T>
struct VectorTraits<T, Standard> {
  typedef std::vector<T> vector_type;
  typedef typename vector_type::const_iterator const_iterator;
  typedef typename vector_type::const_reference const_reference;
  typedef const_reference subscript_return_type;
};

}  // namespace vector_view

// In the process of optimizing the performance of model loading, we found
// that it was necessary to reduce the copying and construction of STL
// containers. So use VectorView to simulate the operation of STL containers
// without copying, such as iteration and subscripting.
//
// Currently, VectorView is applicable to STL vector and Flatbuffers Vector.
// We used the template Traits to unify the behavior of the two, and provided
// an implicit conversion operator from VectorView to STL vector. Please use
// implicit conversion with caution because it will bring significant overhead.

template <typename T, typename U = Flatbuffers>
class VectorView {
 public:
  typedef vector_view::VectorTraits<T, U> Traits;
  explicit VectorView(typename Traits::vector_type const* cvec) {
    cvec_ = cvec;
  }
  typename Traits::subscript_return_type operator[](size_t i) const {
    return cvec_->operator[](i);
  }
  typename Traits::const_iterator begin() const {
    if (!cvec_) {
      return typename Traits::const_iterator();
    }
    return cvec_->begin();
  }
  typename Traits::const_iterator end() const {
    if (!cvec_) {
      return typename Traits::const_iterator();
    }
    return cvec_->end();
  }
  size_t size() const {
    if (!cvec_) {
      return 0;
    }
    return cvec_->size();
  }
  operator std::vector<T>() const {
    VLOG(5) << "Copying elements out of VectorView will damage performance.";
    std::vector<T> tmp;
    tmp.resize(size());
    for (size_t i = 0; i < size(); ++i) {
      tmp[i] = cvec_->operator[](i);
    }
    return tmp;
  }
  ~VectorView() = default;

 private:
  typename Traits::vector_type const* cvec_;
};

}  // namespace lite
}  // namespace paddle
