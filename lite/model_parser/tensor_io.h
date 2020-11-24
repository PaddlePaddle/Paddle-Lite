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
#include "lite/core/scope.h"
#include "lite/model_parser/base/io.h"
#include "lite/model_parser/base/param_desc.h"
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/model_parser/pb/param_desc.h"
#endif  // LITE_ON_TINY_PUBLISH

namespace paddle {
namespace lite {
namespace model_parser {
namespace tensor {

inline void set_lod(lite::Tensor* tensor,
                    const std::vector<std::vector<uint64_t>>& lod) {
  tensor->set_lod(lod);
}

inline const std::vector<std::vector<uint64_t>>& get_lod(
    const lite::Tensor& tensor) {
  return tensor.lod();
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

inline void set_allocation(const lite::Tensor& tensor,
                           TensorInfoWriteAPI* tensor_info) {
  tensor_info->SetDim(tensor.dims().Vectorize());
  tensor_info->SetDataType(ConvertPrecisionType(tensor.precision()));
  tensor_info->Sync();
}

inline size_t get_bytes_size(const lite::Tensor& tensor) {
  return tensor.memory_size();
}

inline void* get_allocation(lite::Tensor* tensor) {
  CHECK(tensor->IsInitialized()) << "The input tensor has not initialized";
  return tensor->raw_data();
}

inline const void* get_allocation(const lite::Tensor& tensor) {
  CHECK(tensor.IsInitialized()) << "The input tensor has not initialized.";
  return tensor.raw_data();
}

}  // namespace tensor

class MetaInfo {};

class MetaInfoDeserializer {
 public:
  MetaInfoDeserializer(BytesReader* reader, MetaInfo* info)
      : reader_(reader), info_(info) {
    CHECK(reader_) << "The input reader is nullptr.";
    CHECK(info_) << "The input argument info is nullptr.";
  }
  void Load() const {
    int64_t size = reader_->ReadForward<int64_t>();
    CHECK_EQ(size, 0);
  }

 private:
  BytesReader* reader_;
  MetaInfo* info_;
};

class LoDTensorDeserializer {
 public:
  explicit LoDTensorDeserializer(BytesReader* reader)
      : buf_(new Buffer), reader_(reader) {
    CHECK(reader_) << "The input reader is nullptr.";
  }

  void LoadForward(lite::Tensor* tensor) {
    CHECK(tensor) << "The input tensor is nullptr.";
    CHECK(!reader_->end()) << "Nothing to read.";
    uint32_t version = reader_->ReadForward<uint32_t>();
    switch (version) {
      case 0:
#ifndef LITE_ON_TINY_PUBLISH
      {
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
        {
          uint32_t inner_version = reader_->ReadForward<uint32_t>();
          CHECK_EQ(inner_version, 0L) << "The version of tensor is wrong.";
          lite::pb::TensorInfoReader tensor_reader(reader_, buf_.get());
          tensor::set_allocation(
              tensor,
              tensor_reader.Dim(),
              lite::ConvertPrecisionType(tensor_reader.GetDataType()));
          void* data = tensor::get_allocation(tensor);
          size_t size = tensor::get_bytes_size(*tensor);
          reader_->ReadForward(data, size);
        }
#else
      {
        LOG(FATAL) << "Tiny-publish mode is not supported to read the 0 "
                      "version model.";
      }
#endif  // LITE_ON_TINY_PUBLISH
        break;
      default:
        LOG(FATAL) << "The version of tensor is not supported.";
    }
  }

 private:
  std::unique_ptr<Buffer> buf_;
  BytesReader* reader_;
};

#ifndef LITE_ON_TINY_PUBLISH
class LoDTensorSerializer {
 public:
  explicit LoDTensorSerializer(BytesWriter* writer)
      : buf_(new Buffer), writer_(writer) {
    CHECK(writer_) << "The input writer is nullptr.";
  }
  void SaveForward(const lite::Tensor& tensor, uint32_t version = 0) {
    CHECK(tensor.target() == TARGET(kHost))
        << "Only host tensor is supported to be serialized.";
    switch (version) {
      case 0: {
        // Save the lod-vector.
        writer_->WriteForward<uint32_t>(version);
        const auto& lod = tensor::get_lod(tensor);
        writer_->WriteForward<uint64_t>(lod.size());
        for (const auto& each : lod) {
          const uint64_t size = each.size() * sizeof(each.front());
          writer_->WriteForward<uint64_t>(size);
          writer_->WriteForward(each.data(), size);
        }
      }
        {
          // Save the raw tensor.
          writer_->WriteForward<uint32_t>(version);
          lite::pb::TensorInfoWriter tensor_writer(writer_, buf_.get());
          tensor::set_allocation(tensor, &tensor_writer);
          writer_->WriteForward(tensor::get_allocation(tensor),
                                tensor::get_bytes_size(tensor));
        }
        break;
      default:
        LOG(FATAL) << "The version of tensor is not supported.";
    }
  }

 private:
  std::unique_ptr<Buffer> buf_;
  BytesWriter* writer_;
};
#endif  // LITE_ON_TINY_PUBLISH

}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
