/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>

#include "common/log.h"
#include "common/type_define.h"
#include "framework/framework.pb-c.h"

namespace paddle_mobile {
namespace framework {

class OpDesc {
 public:
  friend class ProgramOptimize;
  friend class FusionOpMatcher;
  friend class Node;

  explicit OpDesc(PaddleMobile__Framework__Proto__OpDesc *op_desc);
  OpDesc(const OpDesc &op_desc) : type_(op_desc.type_) {
    this->inputs_ = op_desc.inputs_;
    this->outputs_ = op_desc.outputs_;
    this->attrs_ = op_desc.attrs_;
    this->proto_attrs_ = op_desc.proto_attrs_;
  }

  OpDesc() {}
  const std::vector<std::string> &Input(const std::string &name) const;
  const std::vector<std::string> &Output(const std::string &name) const;
  Attribute GetAttr(const std::string &name) const;

  const std::vector<PaddleMobile__Framework__Proto__OpDesc__Attr>
      &GetProtoAttr() const;

  void SetBlockAttr(const std::string &name, BlockDesc *block);
  void SetBlocksAttr(const std::string &name, std::vector<BlockDesc *> block);

  VariableNameMap &GetInputs() { return inputs_; }

  VariableNameMap &GetOutputs() { return outputs_; }

  AttributeMap &GetAttrMap();

  const std::string &Type() { return type_; }

  void SetInputs(VariableNameMap inputs) { inputs_ = inputs; }

  void SetOutputs(VariableNameMap outputs) { outputs_ = outputs; }

  void SetAttrMap(AttributeMap attrs) { attrs_ = attrs; }

 private:
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;
  std::vector<PaddleMobile__Framework__Proto__OpDesc__Attr> proto_attrs_;
};

Print &operator<<(Print &printer, const OpDesc &op_desc);

}  // namespace framework
}  // namespace paddle_mobile
