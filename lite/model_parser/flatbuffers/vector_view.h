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

#include <type_traits>
#include <vector>
#include "flatbuffers/flatbuffers.h"

namespace paddle {
namespace lite {
namespace fbs {

struct Flatbuffers {};
struct Standand {};

template <typename T, typename U, typename K = void>
struct VectorTraits;

template <typename T, typename K>
struct VectorTraits<T, Flatbuffers, K> {
  typedef flatbuffers::Vector<T> vector_type;
  typedef typename vector_type::const_iterator const_iterator;
  typedef typename const_iterator::value_type value_type;
  typedef const typename const_iterator::reference const_reference;
  typedef value_type subscript_return_type;
};

template <typename T>
struct VectorTraits<T,
                    Flatbuffers,
                    typename std::enable_if<std::is_class<T>::value>::type> {
  typedef flatbuffers::Vector<flatbuffers::Offset<T>> vector_type;
  typedef typename vector_type::const_iterator const_iterator;
  typedef typename const_iterator::value_type value_type;
  typedef const typename const_iterator::reference const_reference;
  typedef value_type subscript_return_type;
};

template <typename T>
struct VectorTraits<T, Standand> {
  typedef std::vector<T> vector_type;
  typedef typename vector_type::const_iterator const_iterator;
  typedef typename vector_type::const_reference const_reference;
  typedef const_reference subscript_return_type;
};

template <typename T, typename U = Flatbuffers>
class VectorView {
 public:
  typedef VectorTraits<T, U> Traits;
  explicit VectorView(typename Traits::vector_type const* cvec) {
    cvec_ = cvec;
  }
  typename Traits::subscript_return_type operator[](size_t i) const {
    return cvec_->operator[](i);
  }
  typename Traits::const_iterator begin() const { return cvec_->begin(); }
  typename Traits::const_iterator end() const { return cvec_->end(); }
  size_t size() const { return cvec_->size(); }
  ~VectorView() = default;

 private:
  typename Traits::vector_type const* cvec_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
