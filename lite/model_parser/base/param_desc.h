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
#include <string>
#include <vector>
#include "lite/model_parser/base/traits.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class ParamDescReadAPI {
 public:
  virtual std::string Name() const = 0;
  virtual std::vector<int64_t> Dim() const = 0;
  virtual VarDataType GetDataType() const = 0;
  virtual const void *GetData() const = 0;
  virtual size_t byte_size() const = 0;

  virtual ~ParamDescReadAPI() = default;
};

class ParamDescWriteAPI {
 public:
  virtual void SetName(const std::string &name) { NotImplemented(); }
  virtual void SetDim(const std::vector<int64_t> &dim) { NotImplemented(); }
  virtual void SetDataType(VarDataType data_type) { NotImplemented(); }
  virtual void SetData(const void *data, size_t byte_size) { NotImplemented(); }

  virtual ~ParamDescWriteAPI() = default;

 private:
  void NotImplemented() const {
    LOG(FATAL) << "ParamDescWriteAPI is not available in model read-only mode.";
  }
};

class CombinedParamsDescReadAPI {
 public:
  virtual const ParamDescReadAPI *GetParamDesc(size_t idx) const = 0;
  virtual size_t GetParamsSize() const = 0;
  virtual ~CombinedParamsDescReadAPI() = default;
};

class CombinedParamsDescWriteAPI {
 public:
  virtual ParamDescWriteAPI *AddParamDesc() {
    NotImplemented();
    return nullptr;
  }
  virtual ~CombinedParamsDescWriteAPI() = default;

 private:
  void NotImplemented() const {
    LOG(FATAL) << "CombinedParamsDescWriteAPI is not available in model "
                  "read-only mode.";
  }
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class ParamDescAPI : public ParamDescReadAPI, public ParamDescWriteAPI {
 public:
  virtual ~ParamDescAPI() = default;
};

class CombinedParamsDescAPI : public CombinedParamsDescReadAPI,
                              public CombinedParamsDescWriteAPI {
 public:
  virtual ~CombinedParamsDescAPI() = default;
};

}  // namespace lite
}  // namespace paddle
