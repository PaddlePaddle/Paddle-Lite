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
    auto& info = OpInfoMap<Dtype>::Instance()->Get(type);
    auto op = info.Creator()(type, inputs, outputs, attrs, scope);
    return std::shared_ptr<OperatorBase<Dtype>>(op);
  }
};

#ifdef PADDLE_MOBILE_CPU

#define REGISTER_OPERATOR_CPU(op_type, op_class)                                \
  template <typename Dtype, typename T>                                     \
  class _OpClass_##op_type##_cpu : public op_class<Dtype, T> {                 \
   public:                                                                  \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_cpu, op_class);                 \
  };                                                                        \
  static paddle_mobile::framework::OperatorRegistrar<                       \
      paddle_mobile::CPU, _OpClass_##op_type##_cpu<paddle_mobile::CPU, float>> \
      __op_registrar_##op_type##__cpu(#op_type);                               \
  int TouchOpRegistrar_##op_type##_cpu() {                                        \
    __op_registrar_##op_type##__cpu.Touch();                                   \
    return 0;                                                               \
  }

#define USE_OP_CPU(op_type)                                           \
  extern int TouchOpRegistrar_##op_type##_cpu();                        \
  static int use_op_itself_##op_type##_ __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_cpu()

#endif


#ifdef PADDLE_MOBILE_MALI_GPU
#define REGISTER_OPERATOR_MALI_GPU(op_type, op_class)                                \
  template <typename Dtype, typename T>                                     \
  class _OpClass_##op_type##_mali_gpu : public op_class<Dtype, T> {                 \
   public:                                                                  \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_mali_gpu, op_class);                 \
  };                                                                        \
  static paddle_mobile::framework::OperatorRegistrar<                       \
      paddle_mobile::CPU, _OpClass_##op_type##_mali_gpu<paddle_mobile::CPU, float>> \
      __op_registrar_##op_type##__mali_gpu(#op_type);                               \
  int TouchOpRegistrar_##op_type##_mali_gpu() {                                        \
    __op_registrar_##op_type##__mali_gpu.Touch();                                   \
    return 0;                                                               \
  }

#define USE_OP_MALI_GPU(op_type)                                           \
  extern int TouchOpRegistrar_##op_type##_mali_gpu();                        \
  static int use_op_itself_##op_type##_ __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_mali_gpu()

#endif

#ifdef PADDLE_MOBILE_FPGA
#define REGISTER_OPERATOR_FPGA(op_type, op_class)                                \
  template <typename Dtype, typename T>                                         \
  class _OpClass_##op_type##_fpga : public op_class<Dtype, T> {                 \
   public:                                                                      \
    DEFINE_OP_CONSTRUCTOR(_OpClass_##op_type##_fpga, op_class);                 \
  };                                                                            \
  static paddle_mobile::framework::OperatorRegistrar<                           \
      paddle_mobile::CPU, _OpClass_##op_type##_fpga<paddle_mobile::CPU, float>> \
      __op_registrar_##op_type##__fpga(#op_type);                               \
  int TouchOpRegistrar_##op_type##_fpga() {                                        \
    __op_registrar_##op_type##__fpga.Touch();                                   \
    return 0;                                                               \
  }

#define USE_OP_FPGA(op_type)                                           \
  extern int TouchOpRegistrar_##op_type##_fpga();                        \
  static int use_op_itself_##op_type##_ __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_fpga()

#endif

}  // namespace framework
}  // namespace paddle_mobile
