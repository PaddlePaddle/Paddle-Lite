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

#include <memory>
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
    if (OpInfoMap<Dtype>::Instance()->Has(op_type)) {
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
    OpInfoMap<Dtype>::Instance()->Insert(op_type, info);
  }
};

template <typename Dtype, typename T>
struct OpInfoFiller {
  void operator()(const std::string& op_type, OpInfo<Dtype>* info) const {
    info->creator_ = [](const std::string& type, const VariableNameMap& inputs,
                        const VariableNameMap& outputs,
                        const AttributeMap& attrs, framework::Scope* scope) {
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
      paddle_mobile::framework::Scope* scope) {
    auto& info = OpInfoMap<Dtype>::Instance()->Get(type);
    auto op = info.Creator()(type, inputs, outputs, attrs, scope);
    return std::shared_ptr<OperatorBase<Dtype>>(op);
  }
};

#define REGISTER_OPERATOR(op_type, op_class, device_name, device_type)     \
  template class op_class<device_type, float>;                             \
  template <typename Dtype, typename T>                                    \
  class _OpClass_##op_type##_##device_name : public op_class<Dtype, T> {   \
   public:                                                                 \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_##device_name, op_class);   \
  };                                                                       \
  static paddle_mobile::framework::OperatorRegistrar<                      \
      device_type, _OpClass_##op_type##_##device_name<device_type, float>> \
      __op_registrar_##op_type##_##device_name(#op_type);                  \
  int TouchOpRegistrar_##op_type##_##device_name() {                       \
    __op_registrar_##op_type##_##device_name.Touch();                      \
    return 0;                                                              \
  }

#define REGISTER_OPERATOR_CPU(op_type, op_class) \
  REGISTER_OPERATOR(op_type, op_class, cpu, paddle_mobile::CPU);

#define REGISTER_OPERATOR_FPGA(op_type, op_class) \
  REGISTER_OPERATOR(op_type, op_class, fpga, paddle_mobile::FPGA);

#define REGISTER_OPERATOR_CL(op_type, op_class) \
  REGISTER_OPERATOR(op_type, op_class, cl, paddle_mobile::GPU_CL);

}  // namespace framework
}  // namespace paddle_mobile
