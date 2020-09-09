// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "lite/api/paddle_lite_factory_helper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/all.h"
#include "lite/utils/macros.h"

using LiteType = paddle::lite::Type;

class OpKernelInfoCollector {
 public:
  static OpKernelInfoCollector& Global() {
    static auto* x = new OpKernelInfoCollector;
    return *x;
  }
  void AddOp2path(const std::string& op_name, const std::string& op_path) {
    size_t index = op_path.find_last_of('/');
    if (index != std::string::npos) {
      op2path_.insert(std::pair<std::string, std::string>(
          op_name, op_path.substr(index + 1)));
    }
  }
  void AddKernel2path(const std::string& kernel_name,
                      const std::string& kernel_path) {
    size_t index = kernel_path.find_last_of('/');
    if (index != std::string::npos) {
      kernel2path_.insert(std::pair<std::string, std::string>(
          kernel_name, kernel_path.substr(index + 1)));
    }
  }
  void SetKernel2path(
      const std::map<std::string, std::string>& kernel2path_map) {
    kernel2path_ = kernel2path_map;
  }
  const std::map<std::string, std::string>& GetOp2PathDict() {
    return op2path_;
  }
  const std::map<std::string, std::string>& GetKernel2PathDict() {
    return kernel2path_;
  }

 private:
  std::map<std::string, std::string> op2path_;
  std::map<std::string, std::string> kernel2path_;
};

namespace paddle {
namespace lite {

class OpLiteFactory {
 public:
  // Register a function to create an op
  void RegisterCreator(const std::string& op_type,
                       std::function<std::shared_ptr<OpLite>()> fun) {
    op_registry_[op_type] = fun;
  }

  static OpLiteFactory& Global() {
    static OpLiteFactory* x = new OpLiteFactory;
    return *x;
  }

  std::shared_ptr<OpLite> Create(const std::string& op_type) const {
    auto it = op_registry_.find(op_type);
    if (it == op_registry_.end()) return nullptr;
    return it->second();
  }

  std::string DebugString() const {
    STL::stringstream ss;
    for (const auto& item : op_registry_) {
      ss << " - " << item.first << "\n";
    }
    return ss.str();
  }

  std::vector<std::string> GetAllOps() const {
    std::vector<std::string> res;
    for (const auto& op : op_registry_) {
      res.push_back(op.first);
    }
    return res;
  }

 protected:
  std::map<std::string, std::function<std::shared_ptr<OpLite>()>> op_registry_;
};

using LiteOpRegistry = OpLiteFactory;

// Register OpLite by initializing a static OpLiteRegistrar instance
class OpLiteRegistrar {
 public:
  OpLiteRegistrar(const std::string& op_type,
                  std::function<std::shared_ptr<OpLite>()> fun) {
    OpLiteFactory::Global().RegisterCreator(op_type, fun);
  }
  // Touch function is used to guarantee registrar was initialized.
  void touch() {}
};

class KernelFactory {
 public:
  // Register a function to create kernels
  void RegisterCreator(const std::string& op_type,
                       TargetType target,
                       PrecisionType precision,
                       DataLayoutType layout,
                       std::function<std::unique_ptr<KernelBase>()> fun) {
    op_registry_[op_type][std::make_tuple(target, precision, layout)].push_back(
        fun);
  }

  static KernelFactory& Global() {
    static KernelFactory* x = new KernelFactory;
    return *x;
  }

  /**
   * Create all kernels belongs to an op.
   */
  std::list<std::unique_ptr<KernelBase>> Create(const std::string& op_type) {
    std::list<std::unique_ptr<KernelBase>> res;
    if (op_registry_.find(op_type) == op_registry_.end()) return res;
    auto& kernel_registry = op_registry_[op_type];
    for (auto it = kernel_registry.begin(); it != kernel_registry.end(); ++it) {
      for (auto& fun : it->second) {
        res.emplace_back(fun());
      }
    }
    return res;
  }

  /**
   * Create a specific kernel. Return a list for API compatible.
   */
  std::list<std::unique_ptr<KernelBase>> Create(const std::string& op_type,
                                                TargetType target,
                                                PrecisionType precision,
                                                DataLayoutType layout) {
    std::list<std::unique_ptr<KernelBase>> res;
    if (op_registry_.find(op_type) == op_registry_.end()) return res;
    auto& kernel_registry = op_registry_[op_type];
    auto it = kernel_registry.find(std::make_tuple(target, precision, layout));
    if (it == kernel_registry.end()) return res;
    for (auto& fun : it->second) {
      res.emplace_back(fun());
    }
    return res;
  }

  std::string DebugString() const {
    STL::stringstream ss;
    for (const auto& item : op_registry_) {
      ss << " - " << item.first << "\n";
    }
    return ss.str();
  }

 protected:
  // Outer map: op -> a map of kernel.
  // Inner map: kernel -> creator function.
  // Each kernel was represented by a combination of <TargetType, PrecisionType,
  // DataLayoutType>
  std::map<std::string,
           std::map<std::tuple<TargetType, PrecisionType, DataLayoutType>,
                    std::list<std::function<std::unique_ptr<KernelBase>()>>>>
      op_registry_;
};

using KernelRegistry = KernelFactory;

// Register Kernel by initializing a static KernelRegistrar instance
class KernelRegistrar {
 public:
  KernelRegistrar(const std::string& op_type,
                  TargetType target,
                  PrecisionType precision,
                  DataLayoutType layout,
                  std::function<std::unique_ptr<KernelBase>()> fun) {
    KernelFactory::Global().RegisterCreator(
        op_type, target, precision, layout, fun);
  }
  // Touch function is used to guarantee registrar was initialized.
  void touch() {}
};

}  // namespace lite
}  // namespace paddle

// Register an op.
#define REGISTER_LITE_OP(op_type__, OpClass)                                   \
  static paddle::lite::OpLiteRegistrar op_type__##__registry(                  \
      #op_type__, []() {                                                       \
        return std::unique_ptr<paddle::lite::OpLite>(new OpClass(#op_type__)); \
      });                                                                      \
  int touch_op_##op_type__() {                                                 \
    op_type__##__registry.touch();                                             \
    OpKernelInfoCollector::Global().AddOp2path(#op_type__, __FILE__);          \
    return 0;                                                                  \
  }

// Register a kernel.
#define REGISTER_LITE_KERNEL(                                                 \
    op_type__, target__, precision__, layout__, KernelClass, alias__)         \
  static paddle::lite::KernelRegistrar                                        \
      op_type__##target__##precision__##layout__##alias__##_kernel_registry(  \
          #op_type__,                                                         \
          TARGET(target__),                                                   \
          PRECISION(precision__),                                             \
          DATALAYOUT(layout__),                                               \
          []() {                                                              \
            std::unique_ptr<KernelClass> x(new KernelClass);                  \
            x->set_op_type(#op_type__);                                       \
            x->set_alias(#alias__);                                           \
            return x;                                                         \
          });                                                                 \
  int touch_##op_type__##target__##precision__##layout__##alias__() {         \
    op_type__##target__##precision__##layout__##alias__##_kernel_registry     \
        .touch();                                                             \
    OpKernelInfoCollector::Global().AddKernel2path(                           \
        #op_type__ "," #target__ "," #precision__ "," #layout__ "," #alias__, \
        __FILE__);                                                            \
    return 0;                                                                 \
  }                                                                           \
  static auto                                                                 \
      op_type__##target__##precision__##layout__##alias__##param_register     \
          UNUSED = paddle::lite::ParamTypeRegistry::NewInstance<              \
              TARGET(target__),                                               \
              PRECISION(precision__),                                         \
              DATALAYOUT(layout__)>(#op_type__ "/" #alias__)
