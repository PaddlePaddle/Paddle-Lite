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

#include "lite/backends/opencl/utils/cache.h"
#include <utility>
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace fbs {
namespace opencl {

Cache::Cache(const std::vector<uint8_t>& buffer) {
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
  CHECK(verifier.VerifyBuffer<paddle::lite::fbs::opencl::proto::Cache>(nullptr))
      << "OpenCL Cache verification failed.";
  SyncFromFbs(proto::GetCache(buffer.data()));
}

void Cache::CopyDataToBuffer(std::vector<uint8_t>* buffer) const {
  CHECK(buffer);
  flatbuffers::DetachedBuffer buf{SyncToFbs()};
  buffer->resize(buf.size());
  // To avoid additional dependencies, standard library components are used
  // here.
  std::memcpy(buffer->data(), buf.data(), buf.size());
}

void Cache::SyncFromFbs(const paddle::lite::fbs::opencl::proto::Cache* desc) {
  CHECK(desc);
  const auto* binary_map_desc = desc->binary_map();
  CHECK(binary_map_desc);
  for (const auto& pair : *binary_map_desc) {
    std::vector<std::vector<uint8_t>> binary_paths;
    for (const auto& value : *(pair->value())) {
      const size_t size = value->data()->size();
      binary_paths.emplace_back(std::vector<uint8_t>(size));
      // To avoid additional dependencies, standard library components are used
      // here.
      std::memcpy(binary_paths.back().data(), value->data()->data(), size);
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

// Tuned Cache: for Tuned file
TuneCache::TuneCache(const std::vector<int>& buffer) {
  SyncFromFbs(proto::GetTuneCache(buffer.data()));
}

void TuneCache::CopyDataToBuffer(std::vector<int>* buffer) const {
  CHECK(buffer);
  flatbuffers::DetachedBuffer buf{SyncToFbs()};
  buffer->resize(buf.size());
  // To avoid additional dependencies, standard library components are used
  // here.
  std::memcpy(buffer->data(), buf.data(), buf.size());
}

void TuneCache::SyncFromFbs(
    const paddle::lite::fbs::opencl::proto::TuneCache* desc) {
  CHECK(desc);
  const auto* tune_map_desc = desc->tune_map();
  CHECK(tune_map_desc);
  for (const auto& pair : *tune_map_desc) {
    std::vector<int> tune_vec;
    const auto& value = *(pair->value());
    for (const auto& element : value) {
      tune_vec.push_back(element);
    }
    // To avoid additional dependencies, standard library components are used
    // here.
    tune_map_.insert({pair->key()->str(), std::move(tune_vec)});
  }
}

flatbuffers::DetachedBuffer TuneCache::SyncToFbs() const {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<flatbuffers::Offset<proto::TuneCache_::TunePair>> tune_map;
  for (const auto& pair : tune_map_) {
    tune_map.emplace_back(proto::TuneCache_::CreateTunePairDirect(
        fbb, pair.first.c_str(), &pair.second));
  }
  fbb.Finish(proto::CreateTuneCacheDirect(fbb, &tune_map));
  return fbb.Release();
}

}  // namespace opencl
}  // namespace fbs
}  // namespace lite
}  // namespace paddle
