// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include "arm_dnn_library/core/macros.h"

namespace armdnnlibrary {

// Refer from
// https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/bfloat16.h.
// Assume that most of the bf16 operator kernels are accelerated by NEON, so we
// only keep the naive implementation of the arithmetic operators.
struct ARM_DNN_LIBRARY_ALIGN(2) bfloat16 {
 public:
  uint16_t x;

  // Constructors
  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;

  inline explicit bfloat16(float val) {
    std::memcpy(&x, reinterpret_cast<char*>(&val) + 2, 2);
  }

  template <class T>
  inline explicit bfloat16(const T& val)
      : x(bfloat16(static_cast<float>(val)).x) {}

  // Assignment operators
  inline bfloat16& operator=(bool b) {
    x = b ? 0x3f80 : 0;
    return *this;
  }

  inline bfloat16& operator=(int8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint8_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(int16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint16_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(int32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint32_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(int64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(uint64_t val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(float val) {
    x = bfloat16(val).x;
    return *this;
  }

  inline bfloat16& operator=(double val) {
    x = bfloat16(val).x;
    return *this;
  }

  // Conversion opertors
  inline operator float() const {
    float val = 0.f;
    uint16_t temp = x;
    std::memcpy(
        reinterpret_cast<char*>(&val) + 2, reinterpret_cast<char*>(&temp), 2);
    return val;
  }

  inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  inline explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  inline explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  inline explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  inline explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  inline operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
};

inline bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}

inline bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}

inline bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}

inline bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}

inline bfloat16 operator-(const bfloat16& a) {
  bfloat16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

inline bfloat16& operator+=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) + static_cast<float>(b));
  return a;
}

inline bfloat16& operator-=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) - static_cast<float>(b));
  return a;
}

inline bfloat16& operator*=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) * static_cast<float>(b));
  return a;
}

inline bfloat16& operator/=(bfloat16& a, const bfloat16& b) {  // NOLINT
  a = bfloat16(static_cast<float>(a) / static_cast<float>(b));
  return a;
}

inline bfloat16 raw_uint16_to_bfloat16(uint16_t a) {
  bfloat16 res;
  res.x = a;
  return res;
}

// Comparison operators
inline bool operator==(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

inline bool operator!=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

inline bool operator<(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

inline bool operator>(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

inline bool operator>=(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

inline bool(isnan)(const bfloat16& a) { return (a.x & 0x7FFF) > 0x7F80; }

inline bool(isinf)(const bfloat16& a) { return (a.x & 0x7F80) == 0x7F80; }

inline bool(isfinite)(const bfloat16& a) {
  return !((isnan)(a)) && !((isinf)(a));
}

inline bfloat16(abs)(const bfloat16& a) {
  return bfloat16(std::abs(static_cast<float>(a)));
}

inline std::ostream& operator<<(std::ostream& os, const bfloat16& a) {
  os << static_cast<float>(a);
  return os;
}

}  // namespace armdnnlibrary

namespace std {

template <>
struct is_pod<armdnnlibrary::bfloat16> {
  static const bool value = is_trivial<armdnnlibrary::bfloat16>::value &&
                            is_standard_layout<armdnnlibrary::bfloat16>::value;
};

template <>
struct is_floating_point<armdnnlibrary::bfloat16>
    : std::integral_constant<
          bool,
          std::is_same<
              armdnnlibrary::bfloat16,
              typename std::remove_cv<armdnnlibrary::bfloat16>::type>::value> {
};
template <>
struct is_signed<armdnnlibrary::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<armdnnlibrary::bfloat16> {
  static const bool value = false;
};

inline bool isnan(const armdnnlibrary::bfloat16& a) {
  return armdnnlibrary::isnan(a);
}

inline bool isinf(const armdnnlibrary::bfloat16& a) {
  return armdnnlibrary::isinf(a);
}

template <>
struct numeric_limits<armdnnlibrary::bfloat16> {
  static const bool is_specialized = true;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const bool has_infinity = true;
  static const bool has_quiet_NaN = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_present;
  static const bool has_denorm_loss = false;
  static const std::float_round_style round_style = std::round_to_nearest;
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  static const int digits = 8;
  static const int digits10 = 2;
  static const int max_digits10 = 9;
  static const int radix = 2;
  static const int min_exponent = -125;
  static const int min_exponent10 = -37;
  static const int max_exponent = 128;
  static const int max_exponent10 = 38;
  static const bool traps = true;
  static const bool tinyness_before = false;

  static armdnnlibrary::bfloat16(min)() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0x0080);
  }
  static armdnnlibrary::bfloat16 lowest() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0xff7f);
  }
  static armdnnlibrary::bfloat16(max)() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0x7f7f);
  }
  static armdnnlibrary::bfloat16 epsilon() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0x3C00);
  }
  static armdnnlibrary::bfloat16 round_error() {
    return armdnnlibrary::bfloat16(0.5);
  }
  static armdnnlibrary::bfloat16 infinity() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0x7f80);
  }
  static armdnnlibrary::bfloat16 quiet_NaN() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0xffc1);
  }
  static armdnnlibrary::bfloat16 signaling_NaN() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0xff81);
  }
  static armdnnlibrary::bfloat16 denorm_min() {
    return armdnnlibrary::raw_uint16_to_bfloat16(0x0001);
  }
};

}  // namespace std
