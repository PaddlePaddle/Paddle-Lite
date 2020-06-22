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

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "lite/utils/cp_logging.h"

#define span_REQUIRES_T(VA) , typename std::enable_if<(VA), int>::type = 0

namespace paddle {
namespace lite {

template <typename T>
class Span;

namespace span_detail {

// ---------- size ----------
template <typename T, size_t N>
inline constexpr auto size(const T (&)[N]) noexcept -> size_t {
  return N;
}

template <typename C>
inline constexpr auto size(C const& cont) -> decltype(cont.size()) {
  return cont.size();
}

// ---------- data ----------
template <typename T, size_t N>
inline constexpr auto data(T (&arr)[N]) noexcept -> T* {
  return &arr[0];
}

template <typename C>
inline constexpr auto data(C& cont) -> decltype(cont.data()) {  // NOLINT
  return cont.data();
}

template <typename C>
inline constexpr auto data(C const& cont) -> decltype(cont.data()) {
  return cont.data();
}

template <typename E>
inline constexpr auto data(std::initializer_list<E> il) noexcept -> E const* {
  return il.begin();
}

// ---------- is_span ----------
template <typename Q>
struct is_span_oracle : std::false_type {};

template <typename T>
struct is_span_oracle<Span<T>> : std::true_type {};

template <typename Q>
struct is_span : is_span_oracle<typename std::remove_cv<Q>::type> {};

// ---------- is_compatible_container ----------
template <typename C,
          typename E span_REQUIRES_T(
              (!is_span<C>::value &&
               (std::is_convertible<
                   typename std::remove_pointer<decltype(span_detail::data(
                       std::declval<C&>()))>::type (*)[],  // NOLINT
                   E (*)[]>::value))),                     // NOLINT
          typename = decltype(span_detail::size(std::declval<C>())),
          typename = decltype(span_detail::data(std::declval<C>()))>
struct is_compatible_container : std::true_type {};

// ---------- to_size ----------
template <typename T>
inline constexpr size_t to_size(T size) {
  return static_cast<size_t>(size);
}

}  // namespace span_detail

template <typename T>
class Span {
 public:
  typedef T element_type;
  typedef typename std::remove_cv<T>::type value_type;

  typedef T& reference;
  typedef T* pointer;
  typedef T const* const_pointer;
  typedef T const& const_reference;
  typedef size_t size_type;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  Span() noexcept : data_(nullptr), size_(0) {}

  Span(pointer ptr, size_type count) : data_(ptr), size_(count) {}

  Span(pointer firstElem, pointer lastElem)
      : data_(firstElem),
        size_(span_detail::to_size(std::distance(firstElem, lastElem))) {
    CHECK_GE(std::distance(firstElem, lastElem), 0);
  }

  template <size_t N>
  explicit Span(element_type (&arr)[N]) noexcept : data_(&arr[0]), size_(N) {}

  template <typename Container span_REQUIRES_T(
      (span_detail::is_compatible_container<Container, element_type>::value))>
  explicit Span(Container& cont)  // NOLINT
      : data_(span_detail::data(cont)),
        size_(span_detail::to_size(span_detail::size(cont))) {}

  template <typename Container span_REQUIRES_T(
      (std::is_const<element_type>::value &&
       span_detail::is_compatible_container<Container, element_type>::value))>
  explicit Span(Container const& cont)
      : data_(span_detail::data(cont)),
        size_(span_detail::to_size(span_detail::size(cont))) {}

  Span(Span const& other) noexcept = default;
  ~Span() noexcept = default;
  Span& operator=(Span const& other) noexcept = default;

  size_type size() const noexcept { return size_; }

  std::ptrdiff_t ssize() const noexcept {
    return static_cast<std::ptrdiff_t>(size_);
  }

  size_type size_bytes() const noexcept {
    return size() * span_detail::to_size(sizeof(element_type));
  }

  bool empty() const noexcept { return size() == 0; }

  reference operator[](size_type idx) const {
    CHECK(0 <= idx && idx < size());
    return *(data() + idx);
  }

  reference at(size_type idx) const { return this->operator[](idx); }

  pointer data() const noexcept { return data_; }

  reference front() const noexcept {
    CHECK(!empty());
    return *data();
  }

  reference back() const noexcept {
    CHECK(!empty());
    return *(data() + size() - 1);
  }

  void swap(Span& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  iterator begin() const noexcept { return data(); }

  iterator end() const noexcept { return (data() + size()); }

  const_iterator cbegin() const noexcept { return data(); }

  const_iterator cend() const noexcept { return (data() + size()); }

  const_iterator rbegin() const noexcept { return reverse_iterator(end()); }

  const_iterator rend() const noexcept { return reverse_iterator(begin()); }

  const_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  const_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

 private:
  pointer data_;
  size_type size_;
};

template <typename T1, typename T2>
inline bool operator==(Span<T1> const& l, Span<T2> const& r) {
  return (l.size() == r.size() && std::equal(l.begin(), l.end(), r.begin()));
}

}  // namespace lite
}  // namespace paddle
