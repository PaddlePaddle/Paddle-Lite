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

#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class ProgramDescReadAPI {
 public:
  virtual size_t BlocksSize() const = 0;
  virtual bool HasVersion() const = 0;
  virtual int64_t Version() const = 0;

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  T const* GetBlock(int32_t idx) const;

  virtual ~ProgramDescReadAPI() = default;
};

class ProgramDescWriteAPI {
 public:
  virtual void ClearBlocks() { NotImplemented(); }
  virtual void SetVersion(int64_t version) { NotImplemented(); }

  template <typename T>
  T* AddBlock() {
    NotImplemented();
    return nullptr;
  }

  virtual ~ProgramDescWriteAPI() = default;

 private:
  void NotImplemented() const {
    LOG(FATAL)
        << "ProgramDescWriteAPI is not available in model read-only mode.";
  }
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
