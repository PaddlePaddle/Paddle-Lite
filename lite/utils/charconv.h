// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdlib.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <system_error>  // NOLINT

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

namespace paddle {
namespace lite {
namespace utils {

/*
 * The `std::string` handle can improve the encapsulation, but
 * `const char*` is still needed to improve efficiency in the
 * processing of small strings. This source code gives a simple
 * implementation of the std::from_chars helper function in the
 * C++ 17 standard.
 */
struct from_chars_result {
  const char* ptr{nullptr};
  std::errc ec{};
};

/*
 * Most C++ environments use ASCII as the development character
 * set. Array `kAsciiToInt` is the look-up table implementation
 * of the following functions:
 *
 * static inline uint8_t get_char_val(char c) {
 *   if (likely(c >= '0' && c <= '9'))
 *     return c - '0';
 *   if (c >= 'A' && c <= 'Z')
 *     return 10 + (c - 'A');
 *   if (c >= 'a' && c <= 'z')
 *     return 10 + (c - 'a');
 *   return std::numeric_limits<uint8_t>::max();
 * }
 */
constexpr uint8_t kAsciiToInt[256] = {
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    0 /* 0 */,  1,         2,         3,         4,         5,
    6,          7,         8,         9,         UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, 10 /* A */,
    12,         13,        14,        15,        16,        17,
    18,         19,        20,        21,        22,        23,
    24,         25,        26,        27,        28,        29,
    30,         31,        32,        33,        34,        35,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    10 /* a */, 11,        12,        13,        14,        15,
    16,         17,        18,        19,        20,        21,
    22,         23,        24,        25,        26,        27,
    28,         29,        30,        31,        32,        33,
    34,         35,        UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX, UCHAR_MAX,
    UCHAR_MAX,  UCHAR_MAX, UCHAR_MAX};

/* function: aton_unsigned
 * brief: Convert a string constant to an unsigned integer.
 * parameters:
 *   str(const char*): input string constant.
 *   len(int): bytes of the string.
 *   value(T): result value.
 *   base(int): integer base to use, default to 10.
 * return(from_chars_result):
 *   the error value and the current character pointer.
 */
template <typename T>
from_chars_result aton_unsigned(const char* str,
                                int len,
                                T& value,  // NOLINT
                                int base = 10) {
  from_chars_result result;
  result.ptr = str;

  if (unlikely(!str || len <= 0)) {
    result.ec = std::errc::invalid_argument;
    return result;
  }
  uint64_t val = 0;
  if (unlikely(*str == '-')) {
    result.ec = std::errc::result_out_of_range;
    return result;
  }
  if (unlikely(*str == '+')) {
    ++str;
    --len;
  }
  int i = 0;
  for (; i < len; ++i) {
    uint8_t cv = kAsciiToInt[reinterpret_cast<const uint8_t&>(str[i])];
    if (unlikely(cv >= base)) {
      value = static_cast<T>(val);
      result.ptr = str + i;
      return result;
    }
    // Handling integer values that may exceed the range represented by the
    // basic type.
    if (unlikely(i > std::numeric_limits<uint32_t>::digits10 + 1) &&
        i == std::numeric_limits<uint64_t>::digits10) {
      uint64_t mx = static_cast<uint64_t>(std::numeric_limits<T>::max());
      if (val > mx / 10 || mx - (val * base) < cv) {
        value = static_cast<T>(std::numeric_limits<T>::max());
        result.ec = std::errc::result_out_of_range;
        return result;
      }
    }
    if (likely(i != 10)) {
      val *= base;
    }
    val += cv;
  }
  if (unlikely(i > std::numeric_limits<T>::digits10 + 1 ||
               (i > std::numeric_limits<T>::digits10 &&
                val > static_cast<uint64_t>(std::numeric_limits<T>::max())))) {
    value = static_cast<T>(std::numeric_limits<T>::max());
    result.ec = std::errc::result_out_of_range;
    return result;
  }
  value = static_cast<T>(val);
  return result;
}

/* function: aton_signed
 * brief: Convert a string constant to an signed integer.
 * parameters:
 *   str(const char*): input string constant.
 *   len(int): bytes of the string.
 *   value(T): result value.
 *   base(int): integer base to use, default to 10.
 * return(from_chars_result):
 *   the error value and the current character pointer.
 */
template <typename T>
from_chars_result aton_signed(const char* str,
                              int len,
                              T& value,  // NOLINT
                              int base = 10) {
  from_chars_result result;
  result.ptr = str;

  if (unlikely(!str || len <= 0)) {
    result.ec = std::errc::invalid_argument;
    return result;
  }
  uint64_t val = 0;
  bool negative = (*str == '-');
  if (negative || *str == '+') {
    ++str;
    --len;
  }
  int i = 0;
  for (; i < len; ++i) {
    uint8_t cv = kAsciiToInt[reinterpret_cast<const uint8_t&>(str[i])];
    if (unlikely(cv >= base)) {
      value = static_cast<T>(val);
      result.ptr = str + i;
      return result;
    }
    if (likely(i != 0)) {
      val *= base;
    }
    val += cv;
  }
  if (likely(!negative)) {
    if (unlikely(i > std::numeric_limits<T>::digits10 + 1 ||
                 (i > std::numeric_limits<T>::digits10 &&
                  val > static_cast<int64_t>(std::numeric_limits<T>::max())))) {
      value = static_cast<T>(std::numeric_limits<T>::max());
      result.ec = std::errc::result_out_of_range;
      return result;
    }
    value = static_cast<T>(val);
    return result;
  }
  int64_t ret{static_cast<int64_t>(val)};
  if (negative) {
    ret *= -1;
  }
  if (i > std::numeric_limits<T>::digits10 + 1 ||
      ret < static_cast<int64_t>(std::numeric_limits<T>::min())) {
    value = static_cast<T>(std::numeric_limits<T>::min());
    result.ec = std::errc::result_out_of_range;
    return result;
  }
  value = static_cast<T>(ret);
  return result;
}

/* function: aton_float
 * brief: Convert a string constant to a float digit.
 * parameters:
 *   str(const char*): input string constant.
 *   len(int): bytes of the string.
 *   value(T): result value.
 * return(from_chars_result):
 *   the error value and the current character pointer.
 */
template <typename T>
from_chars_result aton_float(const char* str, int len, T& value) {  // NOLINT
  from_chars_result result;
  result.ptr = str;
  const uint8_t base = 10;

  if (unlikely(!str || len <= 0)) {
    result.ec = std::errc::invalid_argument;
    return result;
  }
  uint64_t lval = 0;
  uint64_t rval = 0;
  uint64_t rdiv = 1;
  bool negative = *str == '-';
  if (negative || *str == '+') {
    ++str;
    --len;
  }
  ssize_t dot_pos = -1;
  int i = 0;
  for (; i < len; ++i) {
    char c = str[i];
    if ('.' == c) {
      dot_pos = i;
      ++i;
      break;
    }
    uint8_t cv = kAsciiToInt[reinterpret_cast<const uint8_t&>(c)];
    if (unlikely(cv >= base)) {
      value = static_cast<T>(lval);
      result.ptr = str + i;
      return result;
    }
    if (i != 0) {
      lval *= 10;
    }
    lval += cv;
  }
  double val{static_cast<double>(lval)};
  if (-1 != dot_pos) {
    for (; i < len; ++i) {
      uint8_t cv = kAsciiToInt[reinterpret_cast<const uint8_t&>(str[i])];
      if (unlikely(cv >= base)) {
        result.ptr = str + i;
        return result;
      }
      if (i - dot_pos > 1) {
        rval *= 10.0;
      }
      rval += cv;
      rdiv *= 10;
    }
    val += static_cast<double>(rval) / rdiv;
  }

  if (!negative && val > static_cast<double>(std::numeric_limits<T>::max())) {
    value = static_cast<T>(std::numeric_limits<T>::max());
    result.ec = std::errc::result_out_of_range;
    return result;
  }
  if (!negative) {
    value = static_cast<T>(val);
    return result;
  }
  val *= -1;
  if (val < static_cast<double>(-std::numeric_limits<T>::max())) {
    value = static_cast<T>(std::numeric_limits<T>::min());
    result.ec = std::errc::result_out_of_range;
    return result;
  }
  value = static_cast<T>(val);
  return result;
}

// To simplify the number of interfaces, using template type
// deduction here.
template <typename T>
from_chars_result from_chars(const char* first,
                             const char* last,
                             T& value,  // NOLINT
                             int base = 10) = delete;

#define UNSIGNED_FROM_CHARS_INSTANCE(T)                          \
  template <>                                                    \
  inline from_chars_result from_chars<T>(                        \
      const char* first, const char* last, T& value, int base) { \
    return aton_unsigned(first, last - first, value, base);      \
  }
#define SIGNED_FROM_CHARS_INSTANCE(T)                            \
  template <>                                                    \
  inline from_chars_result from_chars<T>(                        \
      const char* first, const char* last, T& value, int base) { \
    return aton_signed(first, last - first, value, base);        \
  }
#define FLOAT_FROM_CHARS_INSTANCE(T)                             \
  template <>                                                    \
  inline from_chars_result from_chars<T>(                        \
      const char* first, const char* last, T& value, int base) { \
    return aton_float(first, last - first, value);               \
  }
UNSIGNED_FROM_CHARS_INSTANCE(uint8_t);
UNSIGNED_FROM_CHARS_INSTANCE(uint16_t);
UNSIGNED_FROM_CHARS_INSTANCE(uint32_t);
UNSIGNED_FROM_CHARS_INSTANCE(uint64_t);
SIGNED_FROM_CHARS_INSTANCE(int8_t);
SIGNED_FROM_CHARS_INSTANCE(int16_t);
SIGNED_FROM_CHARS_INSTANCE(int32_t);
SIGNED_FROM_CHARS_INSTANCE(int64_t);
FLOAT_FROM_CHARS_INSTANCE(double);
FLOAT_FROM_CHARS_INSTANCE(float);
#undef FLOAT_FROM_CHARS_INSTANCE
#undef SIGNED_FROM_CHARS_INSTANCE
#undef UNSIGNED_FROM_CHARS_INSTANCE

}  // namespace utils
}  // namespace lite
}  // namespace paddle
