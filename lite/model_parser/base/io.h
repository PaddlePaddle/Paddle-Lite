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

namespace paddle {
namespace lite {
namespace model_parser {

// A simple inline package of core::Buffer.
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;

  explicit Buffer(size_t size) { ResetLazy(size); }

  void CopyDataFrom(const Buffer& other);

  Buffer(Buffer&& other) { raw_ = other.Release(); }
  Buffer& operator=(Buffer&& other) {
    raw_ = other.Release();
    return *this;
  }

  const void* data() const {
    CHECK(raw_);
    return raw_->data();
  }
  void* data() {
    CHECK(raw_);
    return raw_->data();
  }
  size_t capacity() const {
    CHECK(raw_);
    return raw_->space();
  }
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
  virtual void ReadForward(void* dst, size_t size) const = 0;
  virtual std::string ReadForwardToString(size_t size) const;
  virtual size_t length() const = 0;
  virtual bool ReachEnd() const = 0;

  template <typename T,
            typename = typename std::enable_if<
                std::is_trivially_copyable<T>::value>::type>
  T ReadForward() const {
    T tmp;
    ReadForward(&tmp, sizeof(T));
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
  virtual void WriteForward(const void* src, size_t size) const = 0;

  template <typename T,
            typename = typename std::enable_if<
                std::is_trivially_copyable<T>::value>::type>
  void WriteForward(T elem) const {
    WriteForward(&elem, sizeof(T));
  }

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
  void ReadForward(void* dst, size_t size) const override;
  bool ReachEnd() const override { return cur_ >= length_; }
  size_t length() const override { return length_; }

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
  void WriteForward(const void* src, size_t size) const override;

 private:
  FILE* file_{};
  mutable size_t cur_{0};
};

class StringBufferReader : public ByteReader {
 public:
  explicit StringBufferReader(std::string&& buffer)
      : str_(std::forward<std::string>(buffer)),
        buf_(str_.c_str()),
        length_(str_.size()) {
    CHECK(buf_);
  }
  ~StringBufferReader() = default;
  void ReadForward(void* dst, size_t size) const override;
  bool ReachEnd() const override { return cur_ >= length_; }
  size_t length() const override { return length_; }

 private:
  std::string str_;
  const char* buf_;
  size_t length_;
  mutable size_t cur_{0};
};

}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
