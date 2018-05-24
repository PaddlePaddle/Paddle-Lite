#pragma once

#include <vector>
#include <string>

#include "common/log.h"
#include "common/type_define.h"
#include "framework/framework.pb.h"
#include "framework/paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class OpDesc : PaddleMobileObject {
 public:
  friend class ProgramOptimize;
  friend class FusionOpMatcher;
  friend class Node;
  explicit OpDesc(const proto::OpDesc &desc);

  OpDesc(const OpDesc &op_desc): type_(op_desc.type_) {
    this->inputs_ = op_desc.inputs_;
    this->outputs_ = op_desc.outputs_;
    this->attrs_ = op_desc.attrs_;
  }

  OpDesc() {
  }
  const std::vector<std::string> &Input(const std::string &name) const;
  const std::vector<std::string> &Output(const std::string &name) const;
  Attribute GetAttr(const std::string &name) const;

  VariableNameMap &GetInputs() { return inputs_; }

  VariableNameMap &GetOutputs() { return outputs_; }

  AttributeMap &GetAttrMap();

  const std::string &Type() { return type_; }

  void SetInputs(VariableNameMap inputs){
    inputs_ = inputs;
  }

  void SetOutputs(VariableNameMap outputs){
    outputs_ = outputs;
  }

  void SetAttrMap(AttributeMap attrs){
    attrs_ = attrs;
  }

 private:
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;
};

Print &operator<<(Print &printer, const OpDesc &op_desc);

}  // namespace framework
}  // namespace paddle_mobile
