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

#include "utility/cache.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {

static const uint32_t NNADAPTER_CACHE_MAGIC_NUMBER =
    ('.' << 24) + ('n' << 16) + ('n' << 8) + 'c';
static const uint32_t NNADAPTER_CACHE_VERSION_CODE = 1;

static inline uint64_t align4(uint64_t size) { return (size + 3) & ~3; }

NNADAPTER_EXPORT void Cache::Clear() { entries_.clear(); }

bool Cache::Set(const std::string& key, const std::vector<uint8_t>& value) {
  if (entries_.count(key)) return false;
  entries_[key] = value;
  return true;
}

NNADAPTER_EXPORT bool Cache::Set(const std::string& key,
                                 const void* value,
                                 uint64_t size) {
  if (entries_.count(key)) return false;
  entries_[key] = std::vector<uint8_t>();
  entries_[key].resize(size);
  memcpy(entries_[key].data(), value, size);
  return true;
}

NNADAPTER_EXPORT bool Cache::Set(const std::string& key,
                                 const std::string& value) {
  if (entries_.count(key)) return false;
  entries_[key] = std::vector<uint8_t>(value.begin(), value.end());
  return true;
}

NNADAPTER_EXPORT bool Cache::Get(const std::string& key,
                                 std::vector<uint8_t>* value) {
  if (!entries_.count(key)) return false;
  *value = entries_[key];
  return true;
}

NNADAPTER_EXPORT bool Cache::Get(const std::string& key,
                                 void* value,
                                 uint64_t size) {
  if (!entries_.count(key)) return false;
  if (entries_[key].size() != size) return false;
  memcpy(value, entries_[key].data(), size);
  return true;
}

NNADAPTER_EXPORT bool Cache::Get(const std::string& key, std::string* value) {
  if (!entries_.count(key)) return false;
  value->assign(entries_[key].begin(), entries_[key].end());
  return true;
}

NNADAPTER_EXPORT uint64_t Cache::GetSerializedSize() {
  auto size = align4(sizeof(CacheHeader));
  for (const auto& e : entries_) {
    size += align4(sizeof(EntryHeader) + e.first.size() + e.second.size());
  }
  return size;
}

NNADAPTER_EXPORT bool Cache::Serialize(void* buffer, uint64_t size) {
  if (size < GetSerializedSize()) {
    NNADAPTER_LOG(ERROR) << "No enough space, expected " << GetSerializedSize()
                         << " but recevied " << size << " bytes.";
    return false;
  }
  auto cache_header = reinterpret_cast<CacheHeader*>(buffer);
  cache_header->magic_number_ = NNADAPTER_CACHE_MAGIC_NUMBER;
  cache_header->version_code_ = NNADAPTER_CACHE_VERSION_CODE;
  cache_header->num_entries_ = static_cast<uint64_t>(entries_.size());
  auto aligned_cache_header_size = align4(sizeof(CacheHeader));
  auto ptr = reinterpret_cast<uint8_t*>(buffer);
  auto offset = aligned_cache_header_size;
  for (const auto& e : entries_) {
    uint64_t key_size = e.first.size();
    uint64_t value_size = e.second.size();
    auto entry_size = sizeof(EntryHeader) + key_size + value_size;
    auto aligned_entry_size = align4(entry_size);
    auto entry_header = reinterpret_cast<EntryHeader*>(ptr + offset);
    entry_header->key_size_ = key_size;
    entry_header->value_size_ = value_size;
    memcpy(entry_header->payload_,
           reinterpret_cast<const void*>(e.first.c_str()),
           key_size);
    memcpy(entry_header->payload_ + key_size,
           reinterpret_cast<const void*>(e.second.data()),
           value_size);
    if (aligned_entry_size > entry_size) {
      memset(entry_header->payload_ + key_size + value_size,
             0,
             aligned_entry_size - entry_size);
    }
    offset += aligned_entry_size;
  }
  cache_header->crc32c_ = CRC32C(ptr + aligned_cache_header_size,
                                 offset - aligned_cache_header_size);
  return true;
}

NNADAPTER_EXPORT bool Cache::Deserialize(void* buffer, uint64_t size) {
  Clear();
  auto aligned_cache_header_size = align4(sizeof(CacheHeader));
  if (size < aligned_cache_header_size) {
    NNADAPTER_LOG(ERROR) << "No enough space, expected > "
                         << aligned_cache_header_size << " but recevied "
                         << size << " bytes.";
    return false;
  }
  auto cache_header = reinterpret_cast<const CacheHeader*>(buffer);
  if (cache_header->magic_number_ != NNADAPTER_CACHE_MAGIC_NUMBER) {
    NNADAPTER_LOG(ERROR) << "Bad magic number, expected "
                         << string_format("0x%X", NNADAPTER_CACHE_MAGIC_NUMBER)
                         << " but recevied "
                         << string_format("0x%X", cache_header->magic_number_);
    return false;
  }
  if (cache_header->version_code_ != NNADAPTER_CACHE_VERSION_CODE) {
    NNADAPTER_LOG(ERROR) << "Bad version code, expected "
                         << NNADAPTER_CACHE_VERSION_CODE << " but recevied "
                         << cache_header->version_code_
                         << ", should reproduce the cache buffer or file.";
    return false;
  }
  auto ptr = reinterpret_cast<const uint8_t*>(buffer);
  auto offset = aligned_cache_header_size;
  auto crc32c = CRC32C(ptr + offset, size - offset);
  if (cache_header->crc32c_ != crc32c) {
    NNADAPTER_LOG(ERROR) << "crc32c mismatch, expected "
                         << cache_header->crc32c_ << " but recevied " << crc32c
                         << ".";
    return false;
  }
  auto num_entries = cache_header->num_entries_;
  for (uint64_t i = 0; i < num_entries; i++) {
    if (offset + sizeof(EntryHeader) > size) {
      Clear();
      NNADAPTER_LOG(ERROR) << "No enough space for parsing Entry Header.";
      return false;
    }
    auto entry_header = reinterpret_cast<const EntryHeader*>(ptr + offset);
    auto key_size = entry_header->key_size_;
    auto value_size = entry_header->value_size_;
    auto entry_size = sizeof(EntryHeader) + key_size + value_size;
    auto aligned_entry_size = align4(entry_size);
    if (offset + aligned_entry_size > size) {
      Clear();
      NNADAPTER_LOG(ERROR) << "No enough space for parsing Entry Content.";
      return false;
    }
    auto key_data = reinterpret_cast<const char*>(entry_header->payload_);
    auto value_data = entry_header->payload_ + key_size;
    std::string key(key_data, key_size);
    std::vector<uint8_t> value(value_data, value_data + value_size);
    Set(key, value);
    offset += aligned_entry_size;
  }
  return true;
}

}  // namespace nnadapter
