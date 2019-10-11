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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/enforce.h"
#include "common/type_define.h"
#include "common/types.h"
#include "common/variant.h"
#include "framework/attribute.h"
#include "framework/op_info.h"
#include "framework/op_kernel_type.h"
#include "framework/op_registry.h"
#include "framework/program/block_desc.h"
#include "framework/program/program-optimize/node.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"
#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_helper.h"
#include "framework/cl/cl_scope.h"
#endif

namespace paddle_mobile {
namespace framework {

template <typename T>
static T *GetVarValue(const std::string &key, const VariableNameMap &var_map,
                      const Scope &scope) {
  auto var_vec = var_map.at(key);
  if (!var_vec.empty()) {
    auto var = scope.FindVar(var_vec[0]);
    return var->GetMutable<T>();
  } else {
    return nullptr;
  }
}

template <typename Dtype>
class OperatorBase {
 public:
  OperatorBase(const std::string &type, const VariableNameMap &inputs,
               const VariableNameMap &outputs, const AttributeMap &attrs,
               framework::Scope *scope);
  virtual ~OperatorBase() {}

  virtual void Init() = 0;
  virtual void InferShape() const = 0;
  virtual void Run();
  virtual void RunImpl() = 0;

  std::vector<std::string> GetOutKeys() const;
  std::vector<std::string> GetInputKeys() const;

  const VariableNameMap &Inputs() const { return inputs_; }
  const VariableNameMap &Outputs() const { return outputs_; }
  const std::string &Type() const { return type_; }
  const AttributeMap &Attrs() const { return attrs_; }
  void setPrePostType(int prePostType) { pre_post_type_ = prePostType; }

  void ClearVariables(const std::vector<std::string> &var_names) const {
    if (this->scope_) {
      this->scope_->EraseVars(var_names);
    }
  }
#ifdef PADDLE_MOBILE_FPGA
  void InsertTensors();
#endif

 protected:
  framework::Scope *scope_;
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;
  int pre_post_type_ = 0;

 private:
  void CheckAllInputOutputSet() const;
};

template <typename Dtype, typename ParamType, typename KernelType>
class OperatorWithKernel : public OperatorBase<Dtype> {
 public:
  OperatorWithKernel(const std::string &type, const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     framework::Scope *scope)
      : OperatorBase<Dtype>(type, inputs, outputs, attrs, scope),
        param_(inputs, outputs, attrs, scope) {
#ifdef PADDLE_MOBILE_CL
    kernel_.InitCLHelper(scope->GetCLScpoe());
#endif
  }
  virtual void RunImpl() { this->kernel_.Compute(this->param_); }

  virtual void InferShape() const = 0;

  void Init() {
    if (this->pre_post_type_ != NONE_PRE_POST) {
      kernel_.setPrePostType(this->pre_post_type_);
    }
    PADDLE_MOBILE_ENFORCE(kernel_.Init(&param_), "  %s kernel init failed",
                          this->type_.c_str());
  }

 protected:
  KernelType kernel_;
  ParamType param_;
};

template <typename Dtype, typename P>
class OpKernelBase {
 public:
  OpKernelBase() = default;

#ifdef PADDLE_MOBILE_CL
  virtual void InitCLHelper(CLScope *clScope) {
    cl_helper_ = CLHelper(clScope);
  }
#endif

  virtual void Compute(const P &para) = 0;
  virtual bool Init(P *para) { return true; }
  virtual ~OpKernelBase() = default;
  virtual void setPrePostType(int prePostType) { pre_post_type_ = prePostType; }

 protected:
#ifdef PADDLE_MOBILE_CL
  CLHelper cl_helper_;
#endif
  int pre_post_type_ = 0;

 private:
};

class FusionOpMatcher {
 public:
  FusionOpMatcher() {}

  virtual std::string Type() = 0;

  virtual void FolderNodes(
      Node *node,
      std::vector<std::shared_ptr<framework::Node>> *removed_nodes) {
    node->Folder(node_.Depth(), Type(), {}, removed_nodes);
  }

  virtual Node &BeginNode() { return node_; }

  std::string BeginType() { return node_.Type(); }

  virtual std::vector<std::pair<int, std::string>> NeedCheck() { return {}; }

 protected:
  Node node_;
  std::string type_;
  std::shared_ptr<OpDesc> new_opdesc_;
};

#define DECLARE_OPERATOR(OpName, OpParam, OpKernel)                           \
  template <typename DeviceType, typename T>                                  \
  class OpName##Op : public framework::OperatorWithKernel<                    \
                         DeviceType, OpParam<DeviceType>,                     \
                         operators::OpKernel<DeviceType, T>> {                \
   public:                                                                    \
    OpName##Op(const std::string &type, const VariableNameMap &inputs,        \
               const VariableNameMap &outputs,                                \
               const framework::AttributeMap &attrs, framework::Scope *scope) \
        : framework::OperatorWithKernel<DeviceType, OpParam<DeviceType>,      \
                                        operators::OpKernel<DeviceType, T>>(  \
              type, inputs, outputs, attrs, scope) {}                         \
                                                                              \
    void InferShape() const override;                                         \
  };

#define DECLARE_KERNEL(OpName, OpParam)                                   \
  template <typename DeviceType, typename T>                              \
  class OpName##Kernel                                                    \
      : public framework::OpKernelBase<DeviceType, OpParam<DeviceType>> { \
   public:                                                                \
    bool Init(OpParam<DeviceType> *param);                                \
    void Compute(const OpParam<DeviceType> &param);                       \
  };

#define DEFINE_OP_CONSTRUCTOR(cls, parent_cls)                                 \
  cls(const std::string &type, const ::paddle_mobile::VariableNameMap &inputs, \
      const ::paddle_mobile::VariableNameMap &outputs,                         \
      const ::paddle_mobile::framework::AttributeMap &attrs,                   \
      ::paddle_mobile::framework::Scope *scope)                                \
      : parent_cls<Dtype, T>(type, inputs, outputs, attrs, scope) {}

}  // namespace framework
}  // namespace paddle_mobile
