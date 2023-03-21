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

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"

namespace paddle {
namespace lite {
namespace xpu {
namespace npy {

/* Compile-time test for byte order.
   If your compiler does not define these per default, you may want to define
   one of these constants manually.
   Defaults to little endian order. */
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN ||                 \
    defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) ||       \
    defined(__MIBSEB__)
const bool big_endian = true;
#else
const bool big_endian = false;
#endif

typedef uint64_t ndarray_len_t;  // NOLINT
const char magic_string[] = "\x93NUMPY";
const size_t magic_string_length = 6;

const char little_endian_char = '<';
const char big_endian_char = '>';
const char no_endian_char = '|';

constexpr std::array<char, 3> endian_chars = {
    little_endian_char, big_endian_char, no_endian_char};
constexpr std::array<char, 4> numtype_chars = {'f', 'i', 'u', 'c'};

constexpr char host_endian_char =
    (big_endian ? big_endian_char : little_endian_char);

typedef std::pair<char, char> version_t;

struct dtype_t {
  const char byteorder;
  const char kind;
  const unsigned int itemsize;

  // TODO(llohse): implement as constexpr
  inline std::string str() const {
    const size_t max_buflen = 16;
    char buf[max_buflen];  // NOLINT
    std::snprintf(buf, max_buflen, "%c%c%u", byteorder, kind, itemsize);
    return std::string(buf);
  }

  inline std::tuple<const char, const char, const unsigned int> tie() const {
    return std::tie(byteorder, kind, itemsize);
  }
};

struct header_t {
  const dtype_t dtype;
  const bool fortran_order;
  const std::vector<ndarray_len_t> shape;
};

inline void write_magic(std::ostream& ostream, version_t version) {
  ostream.write(magic_string, magic_string_length);
  ostream.put(version.first);
  ostream.put(version.second);
}

inline version_t read_magic(std::istream& istream) {
  char buf[magic_string_length + 2];  // NOLINT
  istream.read(buf, magic_string_length + 2);

  if (!istream) {
    throw std::runtime_error("io error: failed reading file");
  }

  if (0 != std::memcmp(buf, magic_string, magic_string_length)) {
    throw std::runtime_error("this file does not have a valid npy format.");
  }

  version_t version;
  version.first = buf[magic_string_length];
  version.second = buf[magic_string_length + 1];

  return version;
}

// typestring magic

template <typename T>
struct has_typestring {
  static const bool value = false;
};
template <>
struct has_typestring<float> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'f', sizeof(float)};
};
constexpr dtype_t has_typestring<float>::dtype;

// add fp16 for api
template <>
struct has_typestring<float16> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'f', sizeof(float16)};
};
constexpr dtype_t has_typestring<float16>::dtype;

template <>
struct has_typestring<double> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'f', sizeof(double)};
};
constexpr dtype_t has_typestring<double>::dtype;

template <>
struct has_typestring<char> {
  static const bool value = true;
  static constexpr dtype_t dtype = {no_endian_char, 'i', sizeof(char)};
};
constexpr dtype_t has_typestring<char>::dtype;
template <>
struct has_typestring<signed char> {
  static const bool value = true;
  static constexpr dtype_t dtype = {no_endian_char, 'i', sizeof(signed char)};
};
constexpr dtype_t has_typestring<signed char>::dtype;
template <>
struct has_typestring<int16_t> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'i', sizeof(int16_t)};
};
constexpr dtype_t has_typestring<int16_t>::dtype;
template <>
struct has_typestring<int> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'i', sizeof(int)};
};
constexpr dtype_t has_typestring<int>::dtype;
template <>  // load numpy can not show dtype int64 or int64_t
struct has_typestring<int64_t> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'i', sizeof(int64_t)};
};
constexpr dtype_t has_typestring<int64_t>::dtype;

template <>
struct has_typestring<unsigned char> {
  static const bool value = true;
  static constexpr dtype_t dtype = {no_endian_char, 'u', sizeof(unsigned char)};
};
constexpr dtype_t has_typestring<unsigned char>::dtype;
template <>
struct has_typestring<uint16_t> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'u', sizeof(uint16_t)};
};
constexpr dtype_t has_typestring<uint16_t>::dtype;
template <>
struct has_typestring<unsigned int> {
  static const bool value = true;
  static constexpr dtype_t dtype = {
      host_endian_char, 'u', sizeof(unsigned int)};
};
constexpr dtype_t has_typestring<unsigned int>::dtype;
template <>
struct has_typestring<uint64_t> {
  static const bool value = true;
  static constexpr dtype_t dtype = {host_endian_char, 'u', sizeof(uint64_t)};
};
constexpr dtype_t has_typestring<uint64_t>::dtype;

template <>
struct has_typestring<std::complex<float>> {
  static const bool value = true;
  static constexpr dtype_t dtype = {
      host_endian_char, 'c', sizeof(std::complex<float>)};
};
constexpr dtype_t has_typestring<std::complex<float>>::dtype;
template <>
struct has_typestring<std::complex<double>> {
  static const bool value = true;
  static constexpr dtype_t dtype = {
      host_endian_char, 'c', sizeof(std::complex<double>)};
};
constexpr dtype_t has_typestring<std::complex<double>>::dtype;

// helpers
inline bool is_digits(const std::string& str) {
  return std::all_of(str.begin(), str.end(), ::isdigit);
}

template <typename T, size_t N>
inline bool in_array(T val, const std::array<T, N>& arr) {
  return std::find(std::begin(arr), std::end(arr), val) != std::end(arr);
}

inline dtype_t parse_descr(std::string typestring) {
  if (typestring.length() < 3) {
    throw std::runtime_error("invalid typestring (length)");
  }

  char byteorder_c = typestring.at(0);
  char kind_c = typestring.at(1);
  std::string itemsize_s = typestring.substr(2);

  if (!in_array(byteorder_c, endian_chars)) {
    throw std::runtime_error("invalid typestring (byteorder)");
  }

  if (!in_array(kind_c, numtype_chars)) {
    throw std::runtime_error("invalid typestring (kind)");
  }

  if (!is_digits(itemsize_s)) {
    throw std::runtime_error("invalid typestring (itemsize)");
  }
  unsigned int itemsize = std::stoul(itemsize_s);

  return {byteorder_c, kind_c, itemsize};
}

namespace pyparse {

/**
  Removes leading and trailing whitespaces
  */
inline std::string trim(const std::string& str) {
  const std::string whitespace = " \t";
  auto begin = str.find_first_not_of(whitespace);

  if (begin == std::string::npos) {
    return "";
  }

  auto end = str.find_last_not_of(whitespace);

  return str.substr(begin, end - begin + 1);
}

inline std::string get_value_from_map(const std::string& mapstr) {
  size_t sep_pos = mapstr.find_first_of(":");
  if (sep_pos == std::string::npos) {
    return "";
  }

  std::string tmp = mapstr.substr(sep_pos + 1);
  return trim(tmp);
}

/**
   Parses the string representation of a Python dict

   The keys need to be known and may not appear anywhere else in the data.
 */
inline std::unordered_map<std::string, std::string> parse_dict(
    std::string in, const std::vector<std::string>& keys) {
  std::unordered_map<std::string, std::string> map;

  if (keys.size() == 0) {
    return map;
  }

  in = trim(in);

  // unwrap dictionary
  if ((in.front() == '{') && (in.back() == '}')) {
    in = in.substr(1, in.length() - 2);
  } else {
    throw std::runtime_error("Not a Python dictionary.");
  }

  std::vector<std::pair<size_t, std::string>> positions;

  for (auto const& value : keys) {
    size_t pos = in.find("'" + value + "'");

    if (pos == std::string::npos) {
      throw std::runtime_error("Missing '" + value + "' key.");
    }

    std::pair<size_t, std::string> position_pair{pos, value};
    positions.push_back(position_pair);
  }

  // sort by position in dict
  std::sort(positions.begin(), positions.end());

  for (size_t i = 0; i < positions.size(); ++i) {
    std::string raw_value;
    size_t begin{positions[i].first};
    size_t end{std::string::npos};

    std::string key = positions[i].second;

    if (i + 1 < positions.size()) {
      end = positions[i + 1].first;
    }

    raw_value = in.substr(begin, end - begin);

    raw_value = trim(raw_value);

    if (raw_value.back() == ',') {
      raw_value.pop_back();
    }

    map[key] = get_value_from_map(raw_value);
  }

  return map;
}

/**
  Parses the string representation of a Python boolean
  */
inline bool parse_bool(const std::string& in) {
  if (in == "True") {
    return true;
  }
  if (in == "False") {
    return false;
  }

  throw std::runtime_error("Invalid python boolan.");
}

/**
  Parses the string representation of a Python str
  */
inline std::string parse_str(const std::string& in) {
  if ((in.front() == '\'') && (in.back() == '\'')) {
    return in.substr(1, in.length() - 2);
  }

  throw std::runtime_error("Invalid python string.");
}

/**
  Parses the string represenatation of a Python tuple into a vector of its items
 */
inline std::vector<std::string> parse_tuple(std::string in) {
  std::vector<std::string> v;
  const char seperator = ',';

  in = trim(in);

  if ((in.front() == '(') && (in.back() == ')')) {
    in = in.substr(1, in.length() - 2);
  } else {
    throw std::runtime_error("Invalid Python tuple.");
  }

  std::istringstream iss(in);

  for (std::string token; std::getline(iss, token, seperator);) {
    v.push_back(token);
  }

  return v;
}

template <typename T>
inline std::string write_tuple(const std::vector<T>& v) {
  if (v.size() == 0) {
    return "()";
  }

  std::ostringstream ss;

  if (v.size() == 1) {
    ss << "(" << v.front() << ",)";
  } else {
    const std::string delimiter = ", ";
    // v.size() > 1
    ss << "(";
    std::copy(v.begin(),
              v.end() - 1,
              std::ostream_iterator<T>(ss, delimiter.c_str()));
    ss << v.back();
    ss << ")";
  }

  return ss.str();
}

inline std::string write_boolean(bool b) {
  if (b) {
    return "True";
  } else {
    return "False";
  }
}

}  // namespace pyparse

inline header_t parse_header(std::string header) {
  /*
     The first 6 bytes are a magic string: exactly "x93NUMPY".
     The next 1 byte is an unsigned byte: the major version number of the file
     format, e.g. x01.
     The next 1 byte is an unsigned byte: the minor version number of the file
     format, e.g. x00. Note: the version of the file format is not tied to the
     version of the numpy package.
     The next 2 bytes form a little-endian unsigned int16_t int: the length of
     the
     header data HEADER_LEN.
     The next HEADER_LEN bytes form the header data describing the array's
     format. It is an ASCII string which contains a Python literal expression of
     a dictionary. It is terminated by a newline ('n') and padded with spaces
     ('x20') to make the total length of the magic string + 4 + HEADER_LEN be
     evenly divisible by 16 for alignment purposes.
     The dictionary contains three keys:

     "descr" : dtype.descr
     An object that can be passed as an argument to the numpy.dtype()
     constructor to create the array's dtype.
     "fortran_order" : bool
     Whether the array data is Fortran-contiguous or not. Since
     Fortran-contiguous arrays are a common form of non-C-contiguity, we allow
     them to be written directly to disk for efficiency.
     "shape" : tuple of int
     The shape of the array.
     For repeatability and readability, this dictionary is formatted using
     pprint.pformat() so the keys are in alphabetic order.
   */

  // remove trailing newline
  if (header.back() != '\n') {
    throw std::runtime_error("invalid header");
  }
  header.pop_back();

  // parse the dictionary
  std::vector<std::string> keys{"descr", "fortran_order", "shape"};
  auto dict_map = npy::pyparse::parse_dict(header, keys);

  if (dict_map.size() == 0) {
    throw std::runtime_error("invalid dictionary in header");
  }

  std::string descr_s = dict_map["descr"];
  std::string fortran_s = dict_map["fortran_order"];
  std::string shape_s = dict_map["shape"];

  std::string descr = npy::pyparse::parse_str(descr_s);
  dtype_t dtype = parse_descr(descr);

  // convert literal Python bool to C++ bool
  bool fortran_order = npy::pyparse::parse_bool(fortran_s);

  // parse the shape tuple
  auto shape_v = npy::pyparse::parse_tuple(shape_s);

  std::vector<ndarray_len_t> shape;
  for (auto item : shape_v) {
    ndarray_len_t dim = static_cast<ndarray_len_t>(std::stoul(item));
    shape.push_back(dim);
  }

  return {dtype, fortran_order, shape};
}

inline std::string write_header_dict(const std::string& descr,
                                     bool fortran_order,
                                     const std::vector<ndarray_len_t>& shape) {
  std::string s_fortran_order = npy::pyparse::write_boolean(fortran_order);
  std::string shape_s = npy::pyparse::write_tuple(shape);

  return "{'descr': '" + descr + "', 'fortran_order': " + s_fortran_order +
         ", 'shape': " + shape_s + ", }";
}

inline void write_header(std::ostream& out, const header_t& header) {
  std::string header_dict =
      write_header_dict(header.dtype.str(), header.fortran_order, header.shape);

  size_t length = magic_string_length + 2 + 2 + header_dict.length() + 1;

  version_t version{1, 0};
  if (length >= 255 * 255) {
    length = magic_string_length + 2 + 4 + header_dict.length() + 1;
    version = {2, 0};
  }
  size_t padding_len = 16 - length % 16;
  std::string padding(padding_len, ' ');

  // write magic
  write_magic(out, version);

  // write header length
  if (version == version_t{1, 0}) {
    uint8_t header_len_le16[2];
    uint16_t header_len =
        static_cast<uint16_t>(header_dict.length() + padding.length() + 1);

    header_len_le16[0] = (header_len >> 0) & 0xff;
    header_len_le16[1] = (header_len >> 8) & 0xff;
    out.write(reinterpret_cast<char*>(header_len_le16), 2);
  } else {
    uint8_t header_len_le32[4];
    uint32_t header_len =
        static_cast<uint32_t>(header_dict.length() + padding.length() + 1);

    header_len_le32[0] = (header_len >> 0) & 0xff;
    header_len_le32[1] = (header_len >> 8) & 0xff;
    header_len_le32[2] = (header_len >> 16) & 0xff;
    header_len_le32[3] = (header_len >> 24) & 0xff;
    out.write(reinterpret_cast<char*>(header_len_le32), 4);
  }

  out << header_dict << padding << '\n';
}

inline std::string read_header(std::istream& istream) {
  // check magic bytes an version number
  version_t version = read_magic(istream);

  uint32_t header_length;
  if (version == version_t{1, 0}) {
    uint8_t header_len_le16[2];
    istream.read(reinterpret_cast<char*>(header_len_le16), 2);
    header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

    if ((magic_string_length + 2 + 2 + header_length) % 16 != 0) {
      // TODO(llohse): display warning
    }
  } else if (version == version_t{2, 0}) {
    uint8_t header_len_le32[4];
    istream.read(reinterpret_cast<char*>(header_len_le32), 4);

    header_length = (header_len_le32[0] << 0) | (header_len_le32[1] << 8) |
                    (header_len_le32[2] << 16) | (header_len_le32[3] << 24);

    if ((magic_string_length + 2 + 4 + header_length) % 16 != 0) {
      // TODO(llohse): display warning
    }
  } else {
    throw std::runtime_error("unsupported file format version");
  }

  auto buf_v = std::vector<char>();
  buf_v.reserve(header_length);
  istream.read(buf_v.data(), header_length);
  std::string header(buf_v.data(), header_length);

  return header;
}

inline ndarray_len_t comp_size(const std::vector<ndarray_len_t>& shape) {
  ndarray_len_t size = 1;
  for (ndarray_len_t i : shape) {
    size *= i;
  }

  return size;
}

template <typename Scalar>
void SaveArrayAsNumpy(const std::string& filename,
                      bool fortran_order,
                      const size_t shape_len,
                      const Scalar* data) {
  if (filename.empty()) {
    return;
  }

  static_assert(has_typestring<Scalar>::value, "scalar type not understood");
  dtype_t dtype = has_typestring<Scalar>::dtype;

  std::ofstream stream(filename, std::ofstream::out | std::ofstream::binary);
  if (!stream) {
    throw std::runtime_error("io error: failed to open a file.");
  }

  header_t header{
      dtype, fortran_order, {static_cast<ndarray_len_t>(shape_len)}};
  write_header(stream, header);

  auto size = shape_len;

  stream.write(reinterpret_cast<const char*>(data), sizeof(Scalar) * size);
}

}  // namespace npy
}  // namespace xpu
}  // namespace lite
}  // namespace paddle
