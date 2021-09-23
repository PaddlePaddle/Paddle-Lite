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
#include "lite/core/model/base/traits.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

class OpDescReadAPI {
 public:
  virtual std::string Type() const = 0;
  virtual std::vector<std::string> Input(const std::string& param) const = 0;
  virtual std::vector<std::string> InputArgumentNames() const = 0;
  virtual std::vector<std::string> Output(const std::string& param) const = 0;
  virtual std::vector<std::string> OutputArgumentNames() const = 0;
  virtual bool HasAttr(const std::string& name) const = 0;
  virtual OpAttrType GetAttrType(const std::string& name) const = 0;
  virtual std::vector<std::string> AttrNames() const = 0;

  template <typename T>
  T GetAttr(const std::string& name) const;

  std::string Repr() const {
    STL::stringstream ss;
    ss << Type();
    ss << "(";
    for (auto& arg : InputArgumentNames()) {
      ss << arg << ":";
      for (auto val : Input(arg)) {
        ss << val << " ";
      }
    }
    ss << ") -> (";
    for (auto& arg : OutputArgumentNames()) {
      ss << arg << ":";
      for (auto val : Output(arg)) {
        ss << val << " ";
      }
    }
    ss << ")";
    return ss.str();
  }

  virtual ~OpDescReadAPI() = default;
};

class OpDescWriteAPI {
 public:
  virtual void SetType(const std::string& type) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void SetInput(const std::string& param,
                        const std::vector<std::string>& args) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void SetOutput(const std::string& param,
                         const std::vector<std::string>& args) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  template <typename T>
  void SetAttr(const std::string& name, const T& v) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  virtual ~OpDescWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class OpDescAPI : public OpDescReadAPI, public OpDescWriteAPI {
 public:
  using AttrType = OpAttrType;
  virtual ~OpDescAPI() = default;
};

}  // namespace lite
}  // namespace paddle
