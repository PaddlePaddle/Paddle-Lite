/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#include "common/type_define.h"
#include "framework.pb.h"
#include "paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class OpDesc : PaddleMobileObject {
 public:
  OpDesc(const proto::OpDesc &desc);
  const std::vector<std::string> &Input(const std::string &name) const;
  const std::vector<std::string> &Output(const std::string &name) const;
  Attribute GetAttr(const std::string &name) const;

  const VariableNameMap &GetInputs() { return inputs_; }

  const VariableNameMap &GetOutputs() { return outputs_; }

  const AttributeMap &GetAttrMap() const;

  const std::string &Type() { return desc_.type(); };

 private:
  proto::OpDesc desc_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;
};

}  // namespace framework
}  // namespace paddle_mobile
