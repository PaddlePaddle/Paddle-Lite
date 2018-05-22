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

#include <string>
#include <tuple>
#include "common/log.h"
#include "common/type_define.h"
#include "framework/op_info.h"
#include "framework/operator.h"

namespace paddle_mobile {
namespace framework {

class Registrar {
 public:
  void Touch() {}
};

template <typename Dtype, size_t I, bool at_end, typename... ARGS>
class OperatorRegistrarRecursive;

template <typename Dtype, typename... ARGS>
struct OperatorRegistrar : public Registrar {
  explicit OperatorRegistrar(const std::string& op_type) {
    if (OpInfoMap<Dtype>::Instance().Has(op_type)) {
      LOG(paddle_mobile::kLOG_DEBUG1)
          << op_type << " is registered more than once.";
      return;
    }
    if (sizeof...(ARGS) == 0) {
      LOG(paddle_mobile::kLOG_DEBUG1)
          << "OperatorRegistrar should be invoked at least by OpClass";
      return;
    }
    OpInfo<Dtype> info;
    OperatorRegistrarRecursive<Dtype, 0, false, ARGS...>(op_type, &info);
    OpInfoMap<Dtype>::Instance().Insert(op_type, info);
  }
};

template <typename Dtype, typename T>
struct OpInfoFiller {
  void operator()(const std::string& op_type, OpInfo<Dtype>* info) const {
    info->creator_ = [](const std::string& type, const VariableNameMap& inputs,
                        const VariableNameMap& outputs,
                        const AttributeMap& attrs,
                        std::shared_ptr<Scope> scope) {
      return new T(type, inputs, outputs, attrs, scope);
    };
  }
};

template <typename Dtype, size_t I, typename... ARGS>
class OperatorRegistrarRecursive<Dtype, I, false, ARGS...> {
 public:
  using T = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
  OperatorRegistrarRecursive(const std::string& op_type, OpInfo<Dtype>* info) {
    OpInfoFiller<Dtype, T> fill;
    fill(op_type, info);
    constexpr auto size = sizeof...(ARGS);
    OperatorRegistrarRecursive<Dtype, I + 1, I + 1 == size, ARGS...> reg(
        op_type, info);
    (void)(reg);
  }
};

template <typename Dtype, size_t I, typename... ARGS>
class OperatorRegistrarRecursive<Dtype, I, true, ARGS...> {
 public:
  OperatorRegistrarRecursive(const std::string& op_type, OpInfo<Dtype>* info) {}
};

template <typename Dtype>
class OpRegistry {
 public:
  static std::shared_ptr<OperatorBase<Dtype>> CreateOp(
      const std::string& type, const VariableNameMap& inputs,
      const VariableNameMap& outputs, const AttributeMap attrs,
      std::shared_ptr<paddle_mobile::framework::Scope> scope) {
    LOG(paddle_mobile::kLOG_DEBUG1) << " type: " << type;
    LOG(paddle_mobile::kLOG_DEBUG1) << " input size: " << inputs.size();
    LOG(paddle_mobile::kLOG_DEBUG1) << " output size: " << outputs.size();
    LOG(paddle_mobile::kLOG_DEBUG1) << " attr size: " << attrs.size();
    LOG(paddle_mobile::kLOG_DEBUG1)
        << " OpInfoMap size: " << OpInfoMap<Dtype>::Instance().map().size();
    LOG(paddle_mobile::kLOG_DEBUG1) << " has type: " << type << " "
                                    << OpInfoMap<Dtype>::Instance().Has(type);
    auto& info = OpInfoMap<Dtype>::Instance().Get(type);
    auto op = info.Creator()(type, inputs, outputs, attrs, scope);
    return std::shared_ptr<OperatorBase<Dtype>>(op);
  }
};

#define REGISTER_OPERATOR(op_type, op_class)                                \
  template <typename Dtype, typename T>                                     \
  class _OpClass_##op_type##_ : public op_class<Dtype, T> {                 \
   public:                                                                  \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_, op_class);                 \
  };                                                                        \
  static paddle_mobile::framework::OperatorRegistrar<                       \
      paddle_mobile::CPU, _OpClass_##op_type##_<paddle_mobile::CPU, float>> \
      __op_registrar_##op_type##__(#op_type);                               \
  int TouchOpRegistrar_##op_type() {                                        \
    __op_registrar_##op_type##__.Touch();                                   \
    return 0;                                                               \
  }

#define USE_OP(op_type)                                           \
  extern int TouchOpRegistrar_##op_type();                        \
  static int use_op_itself_##op_type##_ __attribute__((unused)) = \
      TouchOpRegistrar_##op_type()

}  // namespace framework
}  // namespace paddle_mobile
