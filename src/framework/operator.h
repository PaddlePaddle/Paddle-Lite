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

#include <map>
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
using std::string;
using std::vector;

template <typename T>
static T *GetVarValue(const string &key, const VariableNameMap &var_map,
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
  /*
   *  @b op 基类的实例化方法, op 获取到了 输入、参数以及提前分配好的输出 tensor
   * */
  OperatorBase(const std::string &type, const VariableNameMap &inputs,
               const VariableNameMap &outputs, const AttributeMap &attrs,
               std::shared_ptr<Scope> scope);
  virtual ~OperatorBase() {}
  void Run();
  std::vector<string> GetOutKeys() const;
  std::vector<string> GetInputKeys() const;
  virtual void RunImpl() = 0;

  virtual void Init() = 0;
  /*
   * @b op 运算所需的输入, 如上一层的输出结果、卷积核
   * */
  const VariableNameMap &Inputs() const { return inputs_; }
  /*
   * @b op 的输出, 内存会提前被分配好, 运算结果会被存到分配好的内存内
   * */
  const VariableNameMap &Outputs() const { return outputs_; }
  /*
   * @b op 类型
   * */
  const std::string &Type() const { return type_; }
  /*
   * @b op 运算所需要用到的参数: 如 conv 运算所需要用到的 stride
   * */
  const AttributeMap &Attrs() const { return attrs_; }
  void ClearVariables(const std::vector<std::string> &var_names) const {
    if (this->scope_) {
      this->scope_->EraseVars(var_names);
    }
  }
  /*
   * @b 根据输入形状和参数计算出输出形状
   * */
  virtual void InferShape() const = 0;

 protected:
  std::shared_ptr<Scope> scope_;
  std::string type_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;

 private:
  void CheckAllInputOutputSet() const;
};

/*
 * @b 这个类为所有带有运算的 op 的父类, 这个 op 继承与 OperatorBase
 * */
template <typename Dtype, typename ParamType, typename KernelType>
class OperatorWithKernel : public OperatorBase<Dtype> {
 public:
  OperatorWithKernel(const std::string &type, const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     std::shared_ptr<Scope> scope)
      : OperatorBase<Dtype>(type, inputs, outputs, attrs, scope),
        param_(inputs, outputs, attrs, *scope) {
#ifdef PADDLE_MOBILE_CL
    kernel_.InitCLHelper(scope->GetCLScpoe());
#endif
  }

  virtual void RunImpl() { this->kernel_.Compute(this->param_); }

  virtual void InferShape() const = 0;

  void Init() {
    //    for (auto i : this->inputs_) {
    //      DLOG << i.first;
    //      DLOG << i.second;
    //    }

    PADDLE_MOBILE_ENFORCE(kernel_.Init(&param_), "  %s kernel init failed",
                          this->type_.c_str());
  }

 protected:
  KernelType kernel_;
  ParamType param_;
};

/*
 * @b 所有kernel的父类
 * */
template <typename Dtype, typename P>
class OpKernelBase {
 public:
  OpKernelBase() = default;

#ifdef PADDLE_MOBILE_CL
  virtual void InitCLHelper(CLScope *clScope) {
    cl_helper_ = CLHelper(clScope);
  }
#endif

    /*
     * @b 所有kernel 需实现 Compute 方法
     * @p para 这个参数为 kernel 运算时所需要用到参数组成的一个结构体,
     *    所有结构体存在与: paddle-mobile/src/operators/op_param.h
     * */
#ifdef PADDLE_McOBILE_MALI_GPU
  OpKernelBase() { acl_op_ = nullptr; }
  void *GetAclOp() const { return acl_op_; }
  void SetAclOp(void *op, void *ob) const {
    reinterpret_cast<OpKernelBase<Dtype, P> *>(ob)->acl_op_ = op;
  }
#endif
  virtual void Compute(const P &para) = 0;
  virtual bool Init(P *para) { return true; }
  virtual ~OpKernelBase() = default;

 protected:
#ifdef PADDLE_MOBILE_CL
  CLHelper cl_helper_;
#endif

 private:
#ifdef PADDLE_MOBILE_MALI_GPU
  void *acl_op_;
#endif
};

#define DEFINE_OP_CONSTRUCTOR(cls, parent_cls)                                 \
  cls(const std::string &type, const ::paddle_mobile::VariableNameMap &inputs, \
      const ::paddle_mobile::VariableNameMap &outputs,                         \
      const ::paddle_mobile::framework::AttributeMap &attrs,                   \
      std::shared_ptr<::paddle_mobile::framework::Scope> scope)                \
      : parent_cls<Dtype, T>(type, inputs, outputs, attrs, scope) {}

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

  //  virtual  bool Fusion();
 protected:
  Node node_;
  std::string type_;
  std::shared_ptr<OpDesc> new_opdesc_;
};

}  // namespace framework
}  // namespace paddle_mobile
