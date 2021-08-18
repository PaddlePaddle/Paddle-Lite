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

#include <map>
#include <string>
#include "lite/model_parser/base/traits.h"
#include "lite/utils/cp_logging.h"
namespace paddle {
namespace lite {

class ProgramDescReadAPI {
 public:
  virtual size_t BlocksSize() const = 0;
  virtual bool HasVersion() const = 0;
  virtual int64_t Version() const = 0;

  virtual bool HasOpVersionMap() const = 0;

  template <typename T>
  T* GetOpVersionMap();

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  T const* GetBlock(int32_t idx) const;

  virtual ~ProgramDescReadAPI() = default;
};

class ProgramDescWriteAPI {
 public:
  virtual void ClearBlocks() { LITE_MODEL_INTERFACE_NOT_IMPLEMENTED; }

  virtual void SetVersion(int64_t version) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  void SetOpVersionMap(std::map<std::string, int32_t> op_version_map) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  template <typename T>
  T* AddBlock() {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  virtual ~ProgramDescWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class ProgramDescAPI : public ProgramDescReadAPI, public ProgramDescWriteAPI {
 public:
  virtual ~ProgramDescAPI() = default;
};

}  // namespace lite
}  // namespace paddle
