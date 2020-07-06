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
#include "flatbuffers/flatbuffers.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace fbs {

struct Flatbuffers {};
struct Standand {};

template <typename T, typename U = void>
struct ElementTraits {
  typedef T element_type;
};

template <typename T>
struct ElementTraits<T*,
                     typename std::enable_if<std::is_class<T>::value>::type> {
  typedef flatbuffers::Offset<T> element_type;
};

template <>
struct ElementTraits<std::string, void> {
  typedef flatbuffers::Offset<flatbuffers::String> element_type;
};

template <typename T, typename U>
struct VectorTraits;

template <typename T>
struct VectorTraits<T, Flatbuffers> {
  typedef flatbuffers::Vector<typename ElementTraits<T>::element_type>
      vector_type;
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
  operator std::vector<T>() {
    VLOG(10) << "Copying elements out of VectorView will damage performance.";
    std::vector<T> tmp;
    tmp.reserve(cvec_->size());
    for (auto val : *cvec_) {
      tmp.push_back(val);
    }
    return tmp;
  }
  ~VectorView() = default;

 private:
  typename Traits::vector_type const* cvec_;
};

struct FBSStrIterator {
  typedef flatbuffers::VectorIterator<
      flatbuffers::Offset<flatbuffers::String>,
      typename flatbuffers::IndirectHelper<
          flatbuffers::Offset<flatbuffers::String>>::return_type>
      VI;

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

template <>
class VectorView<std::string, Flatbuffers> {
 public:
  typedef VectorTraits<std::string, Flatbuffers> Traits;
  explicit VectorView(typename Traits::vector_type const* cvec) {
    cvec_ = cvec;
  }
  std::string operator[](size_t i) const { return cvec_->operator[](i)->str(); }
  FBSStrIterator begin() const { return FBSStrIterator(cvec_->begin()); }
  FBSStrIterator end() const { return FBSStrIterator(cvec_->end()); }
  size_t size() const { return cvec_->size(); }
  operator std::vector<std::string>() {
    VLOG(10) << "Copying elements out of VectorView will damage performance.";
    std::vector<std::string> tmp;
    tmp.reserve(cvec_->size());
    for (auto val : *cvec_) {
      tmp.push_back(val->str());
    }
    return tmp;
  }
  ~VectorView() = default;

 private:
  typename Traits::vector_type const* cvec_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
