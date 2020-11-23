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
#include <vector>
#include "lite/core/framework.pb.h"
#include "lite/core/scope.h"
#include "lite/model_parser/base/memory.h"
#include "lite/model_parser/base/param_desc.h"

namespace paddle {
namespace lite {
namespace model_parser {
namespace tensor {

inline void set_lod(lite::Tensor* tensor,
                    const std::vector<std::vector<uint64_t>>& lod) {
  tensor->set_lod(lod);
}

inline void set_allocation(lite::Tensor* tensor,
                           const std::vector<int64_t>& shape,
                           paddle::lite_api::PrecisionType precision) {
  tensor->Resize(shape);
  tensor->set_persistable(true);
  tensor->set_precision(precision);
  tensor->mutable_data(
      TargetType::kHost,
      tensor->numel() * lite_api::PrecisionTypeLength(precision));
}

inline size_t get_bytes_size(lite::Tensor* tensor) {
  return tensor->memory_size();
}

inline void* get_allocation(lite::Tensor* tensor) {
  CHECK(tensor->IsInitialized());
  return tensor->raw_data();
}
}  // namespace tensor

class BytesReader {
 public:
  BytesReader() = default;
  virtual void ReadForward(void* dst, size_t size) const = 0;
  virtual size_t length() const = 0;
  virtual bool end() const = 0;

  template <typename T>
  T ReadForward();

  virtual ~BytesReader() = default;

 private:
  BytesReader(const BytesReader&) = delete;
  BytesReader& operator=(const BytesReader&) = delete;
};

class BinaryFileReader : public BytesReader {
 public:
  explicit BinaryFileReader(const std::string& path, size_t offset = 0) {
    file_ = fopen(path.c_str(), "rb");
    CHECK(file_) << "Unable to open file: " << path;
    fseek(file_, 0L, SEEK_END);
    length_ = ftell(file_) - offset;
    fseek(file_, offset, SEEK_SET);
  }
  ~BinaryFileReader() { fclose(file_); }
  void ReadForward(void* dst, size_t size) const override {
    CHECK(dst);
    CHECK_EQ(fread(dst, 1, size, file_), size)
        << "Failed to read data correctly.";
    cur_ += size;
  }
  bool end() const { return cur_ < length_; }
  size_t length() const override { return length_; }

 private:
  FILE* file_;
  size_t length_{0};
  mutable size_t cur_{0};
};

class BinaryBufferReader : public BytesReader {
 public:
  explicit BinaryBufferReader(const std::string& buffer)
      : buf_(buffer.c_str()), length_(buffer.size()) {
    CHECK(buf_);
  }
  ~BinaryBufferReader() = default;
  void ReadForward(void* dst, size_t size) const override {
    CHECK(dst);
    std::memcpy(dst, buf_ + cur_, size);
    cur_ += size;
  }
  bool end() const { return cur_ < length_; }
  size_t length() const override { return length_; }

 private:
  const char* buf_;
  size_t length_;
  mutable size_t cur_{0};
};

class MetaInfo {};

class MetaInfoLoader {
 public:
  MetaInfoLoader(BytesReader* reader, MetaInfo* info)
      : reader_(reader), info_(info) {
    CHECK(reader_);
    CHECK(info_);
  }
  void Load() const {
    int64_t size = reader_->ReadForward<int64_t>();
    CHECK_EQ(size, 0);
  }

 private:
  BytesReader* reader_;
  MetaInfo* info_;
};

class TensorInfoPb : public TensorInfoReadAPI {
 public:
  TensorInfoPb(BytesReader* reader, Buffer* buffer) {
    CHECK(reader);
    int32_t size = reader->ReadForward<int32_t>();
    buffer->ReallocateDownward(size);
    reader->ReadForward(buffer->data(), size);
    CHECK(desc_.ParseFromArray(buffer->data(), size))
        << "Cannot parse tensor desc";
  }
  std::vector<int64_t> Dim() const {
    std::vector<int64_t> dims_vec;
    std::copy(
        desc_.dims().begin(), desc_.dims().end(), std::back_inserter(dims_vec));
    return dims_vec;
  }
  VarDataType GetDataType() const {
    return lite::ConvertPrecisionType(ConvertType(desc_.data_type()));
  }

 private:
  framework::proto::VarType::TensorDesc desc_;
  PrecisionType ConvertType(
      ::paddle::framework::proto::VarType_Type pb_type) const {
    typedef ::paddle::framework::proto::VarType_Type VarType_Type;
    PrecisionType type;
    switch (pb_type) {
      case VarType_Type::VarType_Type_FP64:
        type = PRECISION(kFP64);
        break;
      case VarType_Type::VarType_Type_FP32:
        type = PRECISION(kFloat);
        break;
      case VarType_Type::VarType_Type_INT8:
        type = PRECISION(kInt8);
        break;
      case VarType_Type::VarType_Type_UINT8:
        type = PRECISION(kUInt8);
        break;
      case VarType_Type::VarType_Type_INT16:
        type = PRECISION(kInt16);
        break;
      case VarType_Type::VarType_Type_INT32:
        type = PRECISION(kInt32);
        break;
      case VarType_Type::VarType_Type_INT64:
        type = PRECISION(kInt64);
        break;
      default:
        LOG(FATAL) << "unknown type " << pb_type;
    }
    return type;
  }
};

class LoDTensorLoader {
 public:
  explicit LoDTensorLoader(BytesReader* reader)
      : buf_(new Buffer), reader_(reader) {
    CHECK(reader_);
  }

  void LoadForward(lite::Tensor* tensor) {
    CHECK(tensor);
    // Load the lod-vector.
    {
      uint32_t version = reader_->ReadForward<uint32_t>();
      CHECK_EQ(version, 0L);
      uint64_t lod_level = reader_->ReadForward<uint64_t>();
      std::vector<std::vector<uint64_t>> lod{lod_level};
      for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size = reader_->ReadForward<uint64_t>();
        uint64_t elem_size = size / sizeof(uint64_t);
        lod[i].resize(elem_size);
        reader_->ReadForward(lod[i].data(), size);
      }
      tensor::set_lod(tensor, lod);
    }
    // Load the raw tensor.
    {
      // 1. Check the version of tensor.
      uint32_t version = reader_->ReadForward<uint32_t>();
      // 2. Load allocation and persistable.
      switch (version) {
        case 0: {
          TensorInfoPb tensor_info(reader_, buf_.get());
          tensor::set_allocation(
              tensor,
              tensor_info.Dim(),
              lite::ConvertPrecisionType(tensor_info.GetDataType()));
        } break;
        default:
          LOG(FATAL) << "The version of tensor is not supported.";
      }
      // 3. Load the raw data.
      void* data = tensor::get_allocation(tensor);
      size_t size = tensor::get_bytes_size(tensor);
      reader_->ReadForward(data, size);
    }
  }

 private:
  std::unique_ptr<Buffer> buf_;
  BytesReader* reader_;
};

}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
