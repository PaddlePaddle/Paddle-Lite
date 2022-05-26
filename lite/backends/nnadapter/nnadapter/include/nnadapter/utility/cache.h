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

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

namespace nnadapter {

class Cache {
  struct CacheHeader {
    uint32_t magic_number_;
    uint32_t version_code_;
    uint32_t crc32c_;
    uint64_t num_entries_;
  };
  struct EntryHeader {
    uint64_t key_size_;
    uint64_t value_size_;
    uint8_t payload_[];
  };

 public:
  void Clear();
  bool Set(const std::string& key, const std::vector<uint8_t>& value);
  bool Set(const std::string& key, const void* value, uint64_t size);
  bool Set(const std::string& key, const std::string& value);
  bool Get(const std::string& key, std::vector<uint8_t>* value);
  bool Get(const std::string& key, void* value, uint64_t size);
  bool Get(const std::string& key, std::string* value);

  uint64_t GetSerializedSize();
  // Serializes all of cache entries into the memory
  bool Serialize(void* buffer, uint64_t size);
  // Clear and deserializes the buffer to the cache entries
  bool Deserialize(void* buffer, uint64_t size);

 private:
  std::map<std::string, std::vector<uint8_t>> entries_;
};

}  // namespace nnadapter
