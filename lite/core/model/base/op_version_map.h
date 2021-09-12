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

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "lite/core/model/base/traits.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {

class OpVersionMapReadAPI {
 public:
  virtual std::map<std::string, int32_t> GetOpVersionMap() const = 0;
  virtual int32_t GetOpVersionByName(const std::string& name) const = 0;
  virtual ~OpVersionMapReadAPI() = default;
};

class OpVersionMapWriteAPI {
 public:
  virtual void SetOpVersionMap(
      const std::map<std::string, int32_t>& op_version_map) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void AddOpVersion(const std::string& op_name, int32_t op_version) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual ~OpVersionMapWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class OpVersionMapAPI : public OpVersionMapReadAPI,
                        public OpVersionMapWriteAPI {
 public:
  virtual ~OpVersionMapAPI() = default;
};

}  // namespace lite
}  // namespace paddle
