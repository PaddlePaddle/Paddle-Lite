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

#include "lite/model_parser/pb/tensor_io.h"

namespace paddle {
namespace lite {
namespace model_parser {
namespace tensor {

void set_lod(lite::Tensor* tensor,
             const std::vector<std::vector<uint64_t>>& lod) {
  tensor->set_lod(lod);
}

const std::vector<std::vector<uint64_t>>& get_lod(const lite::Tensor& tensor) {
  return tensor.lod();
}

void set_allocation(lite::Tensor* tensor,
                    const std::vector<int64_t>& shape,
                    paddle::lite_api::PrecisionType precision) {
  tensor->Resize(shape);
  tensor->set_persistable(true);
  tensor->set_precision(precision);
  tensor->mutable_data(
      TargetType::kHost,
      tensor->numel() * lite_api::PrecisionTypeLength(precision));
}

void set_allocation(const lite::Tensor& tensor,
                    TensorInfoWriteAPI* tensor_info) {
  tensor_info->SetDim(tensor.dims().Vectorize());
  tensor_info->SetDataType(ConvertPrecisionType(tensor.precision()));
  tensor_info->Sync();
}

size_t get_bytes_size(const lite::Tensor& tensor) {
  return tensor.memory_size();
}

void* get_allocation(lite::Tensor* tensor) {
  CHECK(tensor->IsInitialized()) << "The input tensor has not initialized";
  return tensor->raw_data();
}

const void* get_allocation(const lite::Tensor& tensor) {
  CHECK(tensor.IsInitialized()) << "The input tensor has not initialized.";
  return tensor.raw_data();
}

}  // namespace tensor

namespace pb {

void LoDTensorDeserializer::ForwardRead(lite::Tensor* tensor,
                                        ByteReader* reader) {
  CHECK(tensor) << "The input tensor is nullptr.";
  CHECK(reader) << "The input reader is nullptr.";
  CHECK(!reader->ReachEnd()) << "Nothing to read.";
  uint32_t version = reader->Read<uint32_t>();
  switch (version) {
    case 0: {
#ifndef LITE_ON_TINY_PUBLISH
      // Load the lod-tensor.
      uint64_t lod_level = reader->Read<uint64_t>();
      std::vector<std::vector<uint64_t>> lod(lod_level);
      for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size = reader->Read<uint64_t>();
        uint64_t elem_size = size / sizeof(uint64_t);
        lod[i].resize(elem_size);
        reader->Read(lod[i].data(), size);
      }
      tensor::set_lod(tensor, lod);
      // Load the raw tensor.
      uint32_t inner_version = reader->Read<uint32_t>();
      CHECK_EQ(inner_version, 0L)
          << "Tensor inner version should be 0, but get " << inner_version;
      lite::pb::TensorInfoReader tensor_reader(reader, buf_.get());
      tensor::set_allocation(
          tensor,
          tensor_reader.Dim(),
          lite::ConvertPrecisionType(tensor_reader.GetDataType()));
      void* data = tensor::get_allocation(tensor);
      size_t size = tensor::get_bytes_size(*tensor);
      reader->Read(data, size);
#else
      LOG(FATAL) << "Tiny-publish mode is not supported to read the 0 "
                    "version model.";
#endif  // LITE_ON_TINY_PUBLISH
      break;
    }
    default:
      LOG(FATAL) << "The version of tensor " << version << " is not supported.";
  }
}

#ifndef LITE_ON_TINY_PUBLISH
void LoDTensorSerializer::ForwardWrite(const lite::Tensor& tensor,
                                       ByteWriter* writer,
                                       uint32_t version) {
  CHECK(writer) << "The input writer is nullptr.";
  CHECK(tensor.target() == TARGET(kHost))
      << "Only host tensor is supported to be serialized.";
  switch (version) {
    case 0: {
      // Save the lod-vector.
      writer->Write<uint32_t>(version);
      const auto& lod = tensor::get_lod(tensor);
      writer->Write<uint64_t>(lod.size());
      for (const auto& each : lod) {
        const uint64_t size = each.size() * sizeof(each.front());
        writer->Write<uint64_t>(size);
        writer->Write(each.data(), size);
      }
      // Save the raw tensor.
      writer->Write<uint32_t>(version);
      lite::pb::TensorInfoWriter tensor_writer(writer, buf_.get());
      tensor::set_allocation(tensor, &tensor_writer);
      writer->Write(tensor::get_allocation(tensor),
                    tensor::get_bytes_size(tensor));
      break;
    }
    default:
      LOG(FATAL) << "The version of tensor " << version << " is not supported.";
  }
}
#endif  // LITE_ON_TINY_PUBLISH

}  // namespace pb
}  // namespace model_parser
}  // namespace lite
}  // namespace paddle
