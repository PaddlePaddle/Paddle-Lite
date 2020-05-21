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
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
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
  static OpKernelInfoCollector &Global() {
    static auto *x = new OpKernelInfoCollector;
    return *x;
  }
  void AddOp2path(const std::string &op_name, const std::string &op_path) {
    size_t index = op_path.find_last_of('/');
    if (index != std::string::npos) {
      op2path_.insert(std::pair<std::string, std::string>(
          op_name, op_path.substr(index + 1)));
    }
  }
  void AddKernel2path(const std::string &kernel_name,
                      const std::string &kernel_path) {
    size_t index = kernel_path.find_last_of('/');
    if (index != std::string::npos) {
      kernel2path_.insert(std::pair<std::string, std::string>(
          kernel_name, kernel_path.substr(index + 1)));
    }
  }
  void SetKernel2path(
      const std::map<std::string, std::string> &kernel2path_map) {
    kernel2path_ = kernel2path_map;
  }
  const std::map<std::string, std::string> &GetOp2PathDict() {
    return op2path_;
  }
  const std::map<std::string, std::string> &GetKernel2PathDict() {
    return kernel2path_;
  }

 private:
  std::map<std::string, std::string> op2path_;
  std::map<std::string, std::string> kernel2path_;
};

namespace paddle {
namespace lite {

const std::map<std::string, std::string> &GetOp2PathDict();

using KernelFunc = std::function<void()>;
using KernelFuncCreator = std::function<std::unique_ptr<KernelFunc>()>;
class LiteOpRegistry final : public Factory<OpLite, std::shared_ptr<OpLite>> {
 public:
  static LiteOpRegistry &Global() {
    static auto *x = new LiteOpRegistry;
    return *x;
  }

 private:
  LiteOpRegistry() = default;
};

template <typename OpClass>
class OpLiteRegistor : public Registor<OpClass> {
 public:
  explicit OpLiteRegistor(const std::string &op_type)
      : Registor<OpClass>([&] {
          LiteOpRegistry::Global().Register(
              op_type, [op_type]() -> std::unique_ptr<OpLite> {
                return std::unique_ptr<OpLite>(new OpClass(op_type));
              });
        }) {}
};
template <TargetType Target, PrecisionType Precision, DataLayoutType Layout>
using KernelRegistryForTarget =
    Factory<KernelLite<Target, Precision, Layout>, std::unique_ptr<KernelBase>>;

class KernelRegistry final {
 public:
  using any_kernel_registor_t =
      variant<KernelRegistryForTarget<TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC)> *,  //

              KernelRegistryForTarget<TARGET(kX86),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kX86),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //

              KernelRegistryForTarget<TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kNCHW)> *,  //

              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC)> *,  //

              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageNW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageFolder)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageNW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageDefault)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageFolder)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageNW)> *,  //

              KernelRegistryForTarget<TARGET(kNPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kNPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kNPU),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //

              KernelRegistryForTarget<TARGET(kAPU),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kXPU),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //

              KernelRegistryForTarget<TARGET(kBM),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kBM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kBM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //

              KernelRegistryForTarget<TARGET(kRKNPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kRKNPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kRKNPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kRKNPU),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //

              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //

              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kInt16),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kMLU),
                                      PRECISION(kInt16),
                                      DATALAYOUT(kNCHW)> *  //
              >;

  KernelRegistry();

  static KernelRegistry &Global();

  template <TargetType Target, PrecisionType Precision, DataLayoutType Layout>
  void Register(
      const std::string &name,
      typename KernelRegistryForTarget<Target, Precision, Layout>::creator_t
          &&creator) {
    using kernel_registor_t =
        KernelRegistryForTarget<Target, Precision, Layout>;
    auto &varient = registries_[GetKernelOffset<Target, Precision, Layout>()];
    auto *reg = varient.template get<kernel_registor_t *>();
    CHECK(reg) << "Can not be empty of " << name;
    reg->Register(name, std::move(creator));
#ifdef LITE_ON_MODEL_OPTIMIZE_TOOL
    kernel_info_map_[name].push_back(
        std::make_tuple(Target, Precision, Layout));
#endif  // LITE_ON_MODEL_OPTIMIZE_TOOL
  }

  template <TargetType Target,
            PrecisionType Precision = PRECISION(kFloat),
            DataLayoutType Layout = DATALAYOUT(kNCHW)>
  std::list<std::unique_ptr<KernelBase>> Create(const std::string &op_type) {
    using kernel_registor_t =
        KernelRegistryForTarget<Target, Precision, Layout>;
    std::list<std::unique_ptr<KernelBase>> kernel_list;
    if (registries_[GetKernelOffset<Target, Precision, Layout>()].valid()) {
      kernel_list = registries_[GetKernelOffset<Target, Precision, Layout>()]
                        .template get<kernel_registor_t *>()
                        ->Creates(op_type);
    }
    return kernel_list;
  }

  std::list<std::unique_ptr<KernelBase>> Create(const std::string &op_type,
                                                TargetType target,
                                                PrecisionType precision,
                                                DataLayoutType layout);

  // Get a kernel registry offset in all the registries.
  template <TargetType Target, PrecisionType Precision, DataLayoutType Layout>
  static int GetKernelOffset() {
    CHECK_LT(static_cast<int>(Target), static_cast<int>(TARGET(NUM)));
    CHECK_LT(static_cast<int>(Precision), static_cast<int>(PRECISION(NUM)));
    CHECK_LT(static_cast<int>(Layout), static_cast<int>(DATALAYOUT(NUM)));
    return static_cast<int>(Target) * static_cast<int>(PRECISION(NUM)) *
               static_cast<int>(DATALAYOUT(NUM)) +                            //
           static_cast<int>(Precision) * static_cast<int>(DATALAYOUT(NUM)) +  //
           static_cast<int>(Layout);
  }

  std::string DebugString() const {
#ifndef LITE_ON_MODEL_OPTIMIZE_TOOL
    return "No more debug info";
#else   // LITE_ON_MODEL_OPTIMIZE_TOOL
    STL::stringstream ss;
    ss << "\n";
    ss << "Count of kernel kinds: ";
    int count = 0;
    for (auto &item : kernel_info_map_) {
      count += item.second.size();
    }
    ss << count << "\n";

    ss << "Count of registered kernels: " << kernel_info_map_.size() << "\n";
    for (auto &item : kernel_info_map_) {
      ss << "op: " << item.first << "\n";
      for (auto &kernel : item.second) {
        ss << "   - (" << TargetToStr(std::get<0>(kernel)) << ",";
        ss << PrecisionToStr(std::get<1>(kernel)) << ",";
        ss << DataLayoutToStr(std::get<2>(kernel));
        ss << ")";
        ss << "\n";
      }
    }

    return ss.str();
#endif  // LITE_ON_MODEL_OPTIMIZE_TOOL
  }

 private:
  mutable std::vector<any_kernel_registor_t> registries_;
#ifndef LITE_ON_TINY_PUBLISH
  mutable std::map<
      std::string,
      std::vector<std::tuple<TargetType, PrecisionType, DataLayoutType>>>
      kernel_info_map_;
#endif
};

template <TargetType target,
          PrecisionType precision,
          DataLayoutType layout,
          typename KernelType>
class KernelRegistor : public lite::Registor<KernelType> {
 public:
  KernelRegistor(const std::string &op_type, const std::string &alias)
      : Registor<KernelType>([=] {
          KernelRegistry::Global().Register<target, precision, layout>(
              op_type, [=]() -> std::unique_ptr<KernelType> {
                std::unique_ptr<KernelType> x(new KernelType);
                x->set_op_type(op_type);
                x->set_alias(alias);
                return x;
              });
        }) {}
};

}  // namespace lite
}  // namespace paddle

// Operator registry
#define LITE_OP_REGISTER_INSTANCE(op_type__) op_type__##__registry__instance__
#define REGISTER_LITE_OP(op_type__, OpClass)                              \
  static paddle::lite::OpLiteRegistor<OpClass> LITE_OP_REGISTER_INSTANCE( \
      op_type__)(#op_type__);                                             \
  int touch_op_##op_type__() {                                            \
    OpKernelInfoCollector::Global().AddOp2path(#op_type__, __FILE__);     \
    return LITE_OP_REGISTER_INSTANCE(op_type__).Touch();                  \
  }

// Kernel registry
#define LITE_KERNEL_REGISTER(op_type__, target__, precision__) \
  op_type__##__##target__##__##precision__##__registor__
#define LITE_KERNEL_REGISTER_INSTANCE(                   \
    op_type__, target__, precision__, layout__, alias__) \
  op_type__##__##target__##__##precision__##__##layout__##registor__instance__##alias__  // NOLINT

#define LITE_KERNEL_REGISTER_FAKE(op_type__, target__, precision__, alias__) \
  LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__, alias__)

#define REGISTER_LITE_KERNEL(                                                 \
    op_type__, target__, precision__, layout__, KernelClass, alias__)         \
  static paddle::lite::KernelRegistor<TARGET(target__),                       \
                                      PRECISION(precision__),                 \
                                      DATALAYOUT(layout__),                   \
                                      KernelClass>                            \
      LITE_KERNEL_REGISTER_INSTANCE(                                          \
          op_type__, target__, precision__, layout__, alias__)(#op_type__,    \
                                                               #alias__);     \
  static KernelClass LITE_KERNEL_INSTANCE(                                    \
      op_type__, target__, precision__, layout__, alias__);                   \
  int touch_##op_type__##target__##precision__##layout__##alias__() {         \
    OpKernelInfoCollector::Global().AddKernel2path(                           \
        #op_type__ "," #target__ "," #precision__ "," #layout__ "," #alias__, \
        __FILE__);                                                            \
    LITE_KERNEL_INSTANCE(op_type__, target__, precision__, layout__, alias__) \
        .Touch();                                                             \
    return 0;                                                                 \
  }                                                                           \
  static bool LITE_KERNEL_PARAM_INSTANCE(                                     \
      op_type__, target__, precision__, layout__, alias__) UNUSED =           \
      paddle::lite::ParamTypeRegistry::NewInstance<TARGET(target__),          \
                                                   PRECISION(precision__),    \
                                                   DATALAYOUT(layout__)>(     \
          #op_type__ "/" #alias__)

#define LITE_KERNEL_INSTANCE(                            \
    op_type__, target__, precision__, layout__, alias__) \
  op_type__##target__##precision__##layout__##alias__
#define LITE_KERNEL_PARAM_INSTANCE(                      \
    op_type__, target__, precision__, layout__, alias__) \
  op_type__##target__##precision__##layout__##alias__##param_register
