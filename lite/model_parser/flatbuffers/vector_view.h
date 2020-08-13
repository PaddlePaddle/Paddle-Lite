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
#include <vector>
#include "flatbuffers/flatbuffers.h"
#include "lite/model_parser/base/vector_view.h"

namespace paddle {
namespace lite {
namespace vector_view {

template <typename T>
struct ElementTraits<T*,
                     typename std::enable_if<std::is_class<T>::value>::type> {
  typedef flatbuffers::Offset<T> element_type;
};

template <>
struct ElementTraits<std::string, void> {
  typedef flatbuffers::Offset<flatbuffers::String> element_type;
};

template <typename T>
struct VectorTraits<T, Flatbuffers> {
  typedef flatbuffers::Vector<typename ElementTraits<T>::element_type>
      vector_type;
  typedef typename vector_type::const_iterator const_iterator;
  typedef typename const_iterator::value_type value_type;
  typedef const typename const_iterator::reference const_reference;
  typedef value_type subscript_return_type;
};

struct FBSStrIterator {
  typedef flatbuffers::VectorIterator<
      flatbuffers::Offset<flatbuffers::String>,
      typename flatbuffers::IndirectHelper<
          flatbuffers::Offset<flatbuffers::String>>::return_type>
      VI;

  FBSStrIterator() = default;
  explicit FBSStrIterator(const VI& iter) { iter_ = iter; }
  const VI& raw_iter() const { return iter_; }

  bool operator==(const FBSStrIterator& other) const {
    return iter_ == other.raw_iter();
  }

  bool operator<(const FBSStrIterator& other) const {
    return iter_ < other.raw_iter();
  }

  bool operator!=(const FBSStrIterator& other) const {
    return iter_ != other.raw_iter();
  }

  ptrdiff_t operator-(const FBSStrIterator& other) const {
    return iter_ - other.raw_iter();
  }

  std::string operator*() const { return iter_.operator*()->str(); }
  std::string operator->() const { return iter_.operator->()->str(); }

  FBSStrIterator& operator++() {
    iter_++;
    return *this;
  }

  FBSStrIterator& operator--() {
    iter_--;
    return *this;
  }

  FBSStrIterator operator+(const size_t& offset) {
    return FBSStrIterator(iter_ + offset);
  }

  FBSStrIterator operator-(const size_t& offset) {
    return FBSStrIterator(iter_ - offset);
  }

 private:
  VI iter_;
};

}  // namespace vector_view

template <>
class VectorView<std::string, Flatbuffers> {
 public:
  typedef vector_view::VectorTraits<std::string, Flatbuffers> Traits;
  explicit VectorView(typename Traits::vector_type const* cvec) {
    cvec_ = cvec;
  }
  std::string operator[](size_t i) const { return cvec_->operator[](i)->str(); }
  vector_view::FBSStrIterator begin() const {
    if (!cvec_) {
      return vector_view::FBSStrIterator();
    }
    return vector_view::FBSStrIterator(cvec_->begin());
  }
  vector_view::FBSStrIterator end() const {
    if (!cvec_) {
      return vector_view::FBSStrIterator();
    }
    return vector_view::FBSStrIterator(cvec_->end());
  }
  size_t size() const {
    if (!cvec_) {
      return 0;
    }
    return cvec_->size();
  }
  operator std::vector<std::string>() const {
    VLOG(5) << "Copying elements out of VectorView will damage performance.";
    std::vector<std::string> tmp;
    tmp.resize(size());
    for (size_t i = 0; i < size(); ++i) {
      tmp[i] = cvec_->operator[](i)->str();
    }
    return tmp;
  }
  ~VectorView() = default;

 private:
  typename Traits::vector_type const* cvec_;
};

}  // namespace lite
}  // namespace paddle
