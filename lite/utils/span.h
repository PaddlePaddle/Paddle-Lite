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

// ------- void pointer -------
template <typename T>
struct byte_pointer {
  typedef uint8_t* type;
  typedef void* void_type;
};

template <typename T>
struct byte_pointer<const T> {
  typedef uint8_t const* type;
  typedef void const* void_type;
};

}  // namespace span_detail

template <typename T>
struct IndirectHelper {
  typedef T return_type;
  typedef T const const_return_type;
  typedef T* pointer;
  typedef T& reference;
  static const size_t element_stride = sizeof(T);
  static T& Read(void* p, uint32_t i) { return *(reinterpret_cast<T*>(p) + i); }
  static T const& Read(void const* p, uint32_t i) {
    return *(reinterpret_cast<T const*>(p) + i);
  }
  static T* Address(void* p, uint32_t i) {
    return (reinterpret_cast<T*>(p) + i);
  }
  static T const* Address(void const* p, uint32_t i) {
    return (reinterpret_cast<T const*>(p) + i);
  }
};

template <typename T, typename IT>
struct VectorIterator {
  typedef std::random_access_iterator_tag iterator_category;
  typedef IT value_type;
  typedef std::ptrdiff_t difference_type;
  typedef IT* pointer;
  typedef IT& reference;
  typedef typename span_detail::byte_pointer<IT>::type byte_pointer;
  typedef typename span_detail::byte_pointer<IT>::void_type void_pointer;

  VectorIterator(void_pointer data, uint32_t i)
      : data_(reinterpret_cast<byte_pointer>(data) +
              IndirectHelper<T>::element_stride * i) {}
  VectorIterator(VectorIterator const& other) : data_(other.data_) {}
  VectorIterator() : data_(nullptr) {}

  VectorIterator& operator=(VectorIterator const& other) {
    data_ = other.data_;
    return *this;
  }

  VectorIterator& operator=(VectorIterator&& other) {
    data_ = other.data_;
    return *this;
  }

  bool operator==(VectorIterator const& other) const {
    return data_ == other.data_;
  }

  bool operator<(VectorIterator const& other) const {
    return data_ < other.data_;
  }

  bool operator!=(VectorIterator const& other) const {
    return data_ != other.data_;
  }

  difference_type operator-(VectorIterator const& other) const {
    return (data_ - other.data_) / IndirectHelper<T>::element_stride;
  }

  reference operator*() const { return IndirectHelper<T>::Read(data_, 0); }

  pointer operator->() const { return IndirectHelper<T>::Address(data_, 0); }

  VectorIterator& operator++() {
    data_ += IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator operator++(int) {
    VectorIterator temp(data_, 0);
    data_ += IndirectHelper<T>::element_stride;
    return temp;
  }

  VectorIterator operator+(uint32_t const& offset) const {
    return VectorIterator(data_, offset * IndirectHelper<T>::element_stride, 0);
  }

  VectorIterator& operator+=(uint32_t const& offset) {
    data_ += offset * IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator& operator--() {
    data_ -= IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator operator--(int) {
    VectorIterator temp(data_, 0);
    data_ -= IndirectHelper<T>::element_stride;
    return temp;
  }

  VectorIterator operator-(uint32_t const& offset) const {
    return VectorIterator(data_ - offset * IndirectHelper<T>::element_stride,
                          0);
  }

  VectorIterator& operator-=(uint32_t const& offset) {
    data_ -= offset * IndirectHelper<T>::element_stride;
    return *this;
  }

 private:
  byte_pointer data_;
};

template <typename Iterator>
struct VectorReverseIterator : public std::reverse_iterator<Iterator> {
  explicit VectorReverseIterator(Iterator iter)
      : std::reverse_iterator<Iterator>(iter) {}

  typename Iterator::value_type operator*() const {
    return *(std::reverse_iterator<Iterator>::current);
  }

  typename Iterator::value_type operator->() const {
    return *(std::reverse_iterator<Iterator>::current);
  }
};

template <typename T>
class Span {
 public:
  typedef T element_type;
  typedef typename IndirectHelper<T>::pointer pointer;
  typedef typename IndirectHelper<T>::reference reference;

  typedef typename std::remove_cv<T>::type value_type;

  typedef VectorIterator<T, typename IndirectHelper<T>::return_type> iterator;
  typedef VectorIterator<T, typename IndirectHelper<T>::const_return_type>
      const_iterator;
  typedef VectorReverseIterator<iterator> reverse_iterator;
  typedef VectorReverseIterator<const_iterator> const_reverse_iterator;

  typedef size_t size_type;
  typedef std::ptrdiff_t difference_type;

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
    return IndirectHelper<T>::Read(data(), idx);
  }

  reference at(size_type idx) const { return this->operator[](idx); }

  pointer data() const noexcept { return data_; }

  reference front() const noexcept {
    CHECK(!empty());
    return IndirectHelper<T>::Read(data(), 0);
  }

  reference back() const noexcept {
    CHECK(!empty());
    return IndirectHelper<T>::Read(data(), size() - 1);
  }

  void swap(Span& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  iterator begin() const noexcept { return iterator(data(), 0); }
  iterator end() const noexcept { return iterator(data(), size()); }
  reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end() - 1);
  }
  reverse_iterator rend() const noexcept {
    return reverse_iterator(begin() - 1);
  }

  const_iterator cbegin() const noexcept { return const_iterator(data(), 0); }
  const_iterator cend() const noexcept {
    return const_iterator(data(), size());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end() - 1);
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin() - 1);
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
