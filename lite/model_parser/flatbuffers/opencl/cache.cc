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

#include "lite/model_parser/flatbuffers/opencl/cache.h"
#include <utility>

namespace paddle {
namespace lite {
namespace fbs {
namespace opencl {

Cache::Cache(const model_parser::Buffer& buf) {
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf.data()),
                                 buf.size());
  CHECK(verifier.VerifyBuffer<paddle::lite::fbs::opencl::proto::Cache>(nullptr))
      << "OpenCL Cache verification failed.";
  SyncFromFbs(proto::GetCache(buf.data()));
}

void Cache::CopyDataToBuffer(model_parser::Buffer* buffer) const {
  CHECK(buffer);
  flatbuffers::DetachedBuffer buf{SyncToFbs()};
  buffer->ResetLazy(buf.size());
  model_parser::memcpy(buffer->data(), buf.data(), buf.size());
}

void Cache::SyncFromFbs(const paddle::lite::fbs::opencl::proto::Cache* desc) {
  CHECK(desc);
  const auto* binary_map_desc = desc->binary_map();
  CHECK(binary_map_desc);
  for (const auto& pair : *binary_map_desc) {
    std::vector<std::vector<int8_t>> binary_paths;
    for (const auto& value : *(pair->value())) {
      const size_t size = value->data()->size();
      binary_paths.emplace_back(std::vector<int8_t>(size));
      model_parser::memcpy(
          binary_paths.back().data(), value->data()->data(), size);
    }
    binary_map_.insert({pair->key()->str(), std::move(binary_paths)});
  }
}

flatbuffers::DetachedBuffer Cache::SyncToFbs() const {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<flatbuffers::Offset<proto::Cache_::BinaryPair>> binary_map;
  for (const auto& pair : binary_map_) {
    std::vector<flatbuffers::Offset<proto::Cache_::BinaryVector>> value;
    for (const auto& data : pair.second) {
      value.emplace_back(proto::Cache_::CreateBinaryVectorDirect(fbb, &data));
    }
    binary_map.emplace_back(
        proto::Cache_::CreateBinaryPairDirect(fbb, pair.first.c_str(), &value));
  }
  fbb.Finish(proto::CreateCacheDirect(fbb, &binary_map));
  return fbb.Release();
}

}  // namespace opencl
}  // namespace fbs
}  // namespace lite
}  // namespace paddle
