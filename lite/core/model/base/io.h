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

#include <memory>
#include <string>
#include <utility>
#include "lite/core/memory.h"

// Use the no_sanitize attribute on a function or a global variable declaration
// to specify that a particular instrumentation or set of instrumentations
// should
// not be applied.
#if defined(__clang__) && \
    (__clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 7))
#define LITE_SUPRESS_UBSAN(type) __attribute__((no_sanitize(type)))
#elif defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 409)
#define LITE_SUPRESS_UBSAN(type) __attribute__((no_sanitize_undefined))
#else
#define LITE_SUPRESS_UBSAN(type)
#endif

namespace paddle {
namespace lite {
namespace model_parser {

inline void* memcpy(void* dest, const void* src, std::size_t count) {
  TargetCopy(TargetType::kHost, dest, src, count);
  return dest;
}

// A simple inline package of core::Buffer.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;

  explicit Buffer(size_t size) { ResetLazy(size); }

  void CopyDataFrom(const Buffer& other);

  Buffer(Buffer&& other) {
    raw_ = other.Release();
    size_ = other.size();
  }
  Buffer& operator=(Buffer&& other) {
    raw_ = other.Release();
    size_ = other.size();
    return *this;
  }

  const void* data() const { return raw_->data(); }
  void* data() { return raw_->data(); }
  size_t capacity() const { return raw_->space(); }
  size_t size() const { return size_; }
  void ResetLazy(size_t size);

  std::unique_ptr<lite::Buffer> Release() { return std::move(raw_); }
  const lite::Buffer* raw() const { return raw_.get(); }

 private:
  std::unique_ptr<lite::Buffer> raw_{new lite::Buffer};
  size_t size_{0};
};

class ByteReader {
 public:
  ByteReader() = default;
  virtual void Read(void* dst, size_t size) const = 0;
  virtual std::string ReadToString(size_t size) const;
  virtual size_t length() const = 0;
  virtual size_t current() const = 0;
  virtual bool ReachEnd() const = 0;

  template <typename T,
            typename = typename std::enable_if<
                std::is_trivially_copyable<T>::value>::type>
  T Read() const {
    T tmp;
    Read(&tmp, sizeof(T));
    return tmp;
  }

  virtual ~ByteReader() = default;

 private:
  ByteReader(const ByteReader&) = delete;
  ByteReader& operator=(const ByteReader&) = delete;
};

class ByteWriter {
 public:
  ByteWriter() = default;
  virtual void Write(const void* src, size_t size) const = 0;

  template <typename T,
            typename = typename std::enable_if<
                std::is_trivially_copyable<T>::value>::type>
  void Write(T elem) const {
    Write(&elem, sizeof(T));
  }

  template <typename T, int N>
  void Write(const T (&array)[N]) const {
    Write(&array[0], sizeof(T) * N);
  }

  virtual size_t Align(size_t bytes_size) const = 0;

  virtual ~ByteWriter() = default;

 private:
  ByteWriter(const ByteWriter&) = delete;
  ByteWriter& operator=(const ByteWriter&) = delete;
};

class BinaryFileReader : public ByteReader {
 public:
  explicit BinaryFileReader(const std::string& path, size_t offset = 0);
  ~BinaryFileReader() {
    if (file_) {
      fclose(file_);
    }
  }
  void Read(void* dst, size_t size) const override;
  bool ReachEnd() const override { return cur_ >= length_; }
  size_t length() const override { return length_; }
  size_t current() const override { return cur_; }

 private:
  FILE* file_{};
  size_t length_{0};
  mutable size_t cur_{0};
};

class BinaryFileWriter : public ByteWriter {
 public:
  explicit BinaryFileWriter(const std::string& path) {
    file_ = fopen(path.c_str(), "wb");
    CHECK(file_) << "Unable to open file: " << path;
  }
  ~BinaryFileWriter() {
    if (file_) {
      fclose(file_);
    }
  }
  void Write(const void* src, size_t size) const override;

  // Fill a number of zero characters to align the number
  // of written bytes to a certain position.
  size_t Align(size_t scalar_size) const override {
    const size_t padding_bytes = PaddingBytes(cur_, scalar_size);
    for (size_t i = 0; i < padding_bytes; ++i) {
      ByteWriter::Write<uint8_t>(0U);
    }
    return padding_bytes;
  }

 private:
  FILE* file_{};
  mutable size_t cur_{0};

  // Calculate the minimum number of bytes required to be aligned.
  LITE_SUPRESS_UBSAN("unsigned-integer-overflow")
  size_t PaddingBytes(size_t buf_size, size_t scalar_size) const {
    return ((~buf_size) + 1) & (scalar_size - 1);
  }
};

class StringBufferReader : public ByteReader {
 public:
  explicit StringBufferReader(const std::string& buffer)
      : str_(buffer), buf_(str_.c_str()), length_(str_.size()) {
    CHECK(buf_);
  }
  ~StringBufferReader() = default;
  void Read(void* dst, size_t size) const override;
  bool ReachEnd() const override { return cur_ >= length_; }
  size_t length() const override { return length_; }
  size_t current() const override { return cur_; }

 private:
  const std::string& str_;
  const char* buf_;
  size_t length_;
  mutable size_t cur_{0};
};

}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
