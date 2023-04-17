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

#include "lite/utils/any.h"

#ifdef LITE_WITH_METAL
#include "lite/backends/metal/context.h"
#endif
#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_context.h"
#include "lite/backends/opencl/cl_runtime.h"
#endif
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/xpu_header_sitter.h"
#endif
#ifdef LITE_WITH_NNADAPTER
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#endif

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/device_info.h"
#include "lite/core/scope.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/utils/all.h"
#include "lite/utils/env.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

template <TargetType Type>
class Context;

using HostContext = Context<TargetType::kHost>;
using X86Context = Context<TargetType::kX86>;
using ARMContext = Context<TargetType::kARM>;
using XPUContext = Context<TargetType::kXPU>;
using OpenCLContext = Context<TargetType::kOpenCL>;
using NNAdapterContext = Context<TargetType::kNNAdapter>;
using MTLContext = Context<TargetType::kMetal>;

template <>
class Context<TargetType::kHost> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedTo(HostContext* ctx) {}

  std::string name() const { return "HostContext"; }
};

#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
template <>
class Context<TargetType::kNNAdapter> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}
  void CopySharedTo(NNAdapterContext* ctx) {}

  std::string name() const { return "NNAdapterContext"; }

  static void SetNNAdapterModelCacheDir(Scope* scope,
                                        const std::string& model_cache_dir) {
    auto var = scope->Var("NNADAPTER_MODEL_CACHE_DIR");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = model_cache_dir;
  }

  static std::string NNAdapterModelCacheDir(Scope* scope) {
    auto var = scope->FindVar("NNADAPTER_MODEL_CACHE_DIR");
    if (!var) return "";
    return var->Get<std::string>();
  }

  static void SetNNAdapterDynamicShapeInfo(
      Scope* scope,
      const std::map<std::string, std::vector<std::vector<int64_t>>>&
          nnadapter_dynamic_shape_info) {
    auto var = scope->Var("NNADAPTER_DYNAMIC_SHAPE_INFO");
    CHECK(var);
    auto data = var->GetMutable<
        std::map<std::string, std::vector<std::vector<int64_t>>>>();
    CHECK(data);
    *data = nnadapter_dynamic_shape_info;
  }

  static std::map<std::string, std::vector<std::vector<int64_t>>>
  NNAdapterDynamicShapeInfo(Scope* scope) {
    auto var = scope->FindVar("NNADAPTER_DYNAMIC_SHAPE_INFO");
    if (!var) return std::map<std::string, std::vector<std::vector<int64_t>>>();
    return var->Get<std::map<std::string, std::vector<std::vector<int64_t>>>>();
  }

  static void SetNNAdapterModelCacheBuffers(
      Scope* scope,
      const std::map<std::string, std::vector<char>>& model_cache_buffers) {
    for (const auto& model_cache_buffer : model_cache_buffers) {
      auto& key = model_cache_buffer.first;
      auto var = scope->Var("NNADAPTER_MODEL_CACHE_BUFFERS_" + key);
      CHECK(var);
      auto data = var->GetMutable<std::vector<char>>();
      CHECK(data);
      *data = model_cache_buffer.second;
    }
  }

  static bool NNAdapterModelCacheBuffers(
      Scope* scope,
      const std::string& model_cache_token,
      std::vector<char>* model_cache_buffer) {
    CHECK(model_cache_buffer);
    model_cache_buffer->clear();
    auto var =
        scope->FindVar("NNADAPTER_MODEL_CACHE_BUFFERS_" + model_cache_token);
    if (!var) return false;
    auto data = var->GetMutable<std::vector<char>>();
    *model_cache_buffer = *data;
    // Reset to reduce memory consumption
    std::vector<char>().swap(*data);
    return true;
  }

#ifdef LITE_WITH_NNADAPTER
  static bool CheckNNAdapterDeviceName(const std::string& device_name) {
    NNAdapterDevice* device = nullptr;
    int result = NNAdapterDevice_acquire_invoke(device_name.c_str(), &device);
    bool found = result == NNADAPTER_NO_ERROR && device != nullptr;
    if (found) {
      NNAdapterDevice_release_invoke(device);
    }
    return found;
  }
#endif

  static void SetNNAdapterDeviceNames(
      Scope* scope, const std::vector<std::string>& device_names) {
    auto var = scope->Var("NNADAPTER_DEVICE_NAMES");
    CHECK(var);
    auto data = var->GetMutable<std::vector<std::string>>();
    CHECK(data);
    *data = device_names;
  }

  static std::vector<std::string> NNAdapterDeviceNames(Scope* scope) {
    auto var = scope->FindVar("NNADAPTER_DEVICE_NAMES");
    if (!var) return std::vector<std::string>();
    return var->Get<std::vector<std::string>>();
  }

  static void SetNNAdapterContextProperties(
      Scope* scope, const std::string& context_properties) {
    auto var = scope->Var("NNADAPTER_CONTEXT_PROPERTIES");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = context_properties;
  }

  static std::string NNAdapterContextProperties(Scope* scope) {
    auto var = scope->FindVar("NNADAPTER_CONTEXT_PROPERTIES");
    if (!var) return "";
    return var->Get<std::string>();
  }

  static void SetNNAdapterContextCallback(
      Scope* scope, int (*context_callback)(int event_id, void* user_data)) {
    auto var = scope->Var("NNADAPTER_CONTEXT_CALLBACK");
    CHECK(var);
    auto data = var->GetMutable<int (*)(int event_id, void* user_data)>();
    CHECK(data);
    *data = context_callback;
  }

  static int (*NNAdapterContextCallback(Scope* scope))(int event_id,  // NOLINT
                                                       void* user_data) {
    auto var = scope->FindVar("NNADAPTER_CONTEXT_CALLBACK");
    if (!var) return nullptr;
    return var->Get<int (*)(int event_id, void* user_data)>();
  }

  static void SetNNAdapterSubgraphPartitionConfigPath(
      Scope* scope, const std::string& subgraph_partition_config_path) {
    auto var = scope->Var("NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = subgraph_partition_config_path;
  }

  static std::string NNAdapterSubgraphPartitionConfigPath(Scope* scope) {
    auto var = scope->FindVar("NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH");
    if (!var) return "";
    return var->Get<std::string>();
  }

  static void SetNNAdapterSubgraphPartitionConfigBuffer(
      Scope* scope, const std::string& subgraph_partition_config_buffer) {
    auto var = scope->Var("NNADAPTER_SUBGRAPH_PARTITION_CONFIG_BUFFER");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = subgraph_partition_config_buffer;
  }

  static std::string NNAdapterSubgraphPartitionConfigBuffer(Scope* scope) {
    auto var = scope->FindVar("NNADAPTER_SUBGRAPH_PARTITION_CONFIG_BUFFER");
    if (!var) return "";
    return var->Get<std::string>();
  }

  static void SetNNAdapterMixedPrecisionQuantizationConfigPath(
      Scope* scope,
      const std::string& mixed_precision_quantization_config_path) {
    auto var = scope->Var("NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = mixed_precision_quantization_config_path;
  }

  static std::string NNAdapterMixedPrecisionQuantizationConfigPath(
      Scope* scope) {
    auto var =
        scope->FindVar("NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH");
    if (!var) return "";
    return var->Get<std::string>();
  }

  static void SetNNAdapterMixedPrecisionQuantizationConfigBuffer(
      Scope* scope,
      const std::string& mixed_precision_quantization_config_buffer) {
    auto var =
        scope->Var("NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_BUFFER");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = mixed_precision_quantization_config_buffer;
  }

  static std::string NNAdapterMixedPrecisionQuantizationConfigBuffer(
      Scope* scope) {
    auto var =
        scope->FindVar("NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_BUFFER");
    if (!var) return "";
    return var->Get<std::string>();
  }
};
#endif

#ifdef LITE_WITH_XPU
template <>
class Context<TargetType::kXPU> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedTo(XPUContext* ctx) {}

  // TODO(miaotianxiang): remove this
  static xdnn::Context* GetRawContext() {
    return TargetWrapperXPU::GetRawContext();
  }

  std::string name() const { return "XPUContext"; }
};
#endif

#ifdef LITE_WITH_ARM
template <>
class Context<TargetType::kARM> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() { DeviceInfo::Init(); }

  void CopySharedTo(ARMContext* ctx) {}

  void SetRunMode(lite_api::PowerMode mode, int threads) {
    return DeviceInfo::Global().SetRunMode(mode, threads);
  }
  void SetCache(int l1size, int l2size, int l3size) {
    return DeviceInfo::Global().SetCache(l1size, l2size, l3size);
  }
  void SetArch(ARMArch arch) { return DeviceInfo::Global().SetArch(arch); }

  lite_api::PowerMode mode() const { return DeviceInfo::Global().mode(); }
  int threads() const { return DeviceInfo::Global().threads(); }
  ARMArch arch() const { return DeviceInfo::Global().arch(); }
  int l1_cache_size() const { return DeviceInfo::Global().l1_cache_size(); }
  int l2_cache_size() const { return DeviceInfo::Global().l2_cache_size(); }
  int l3_cache_size() const { return DeviceInfo::Global().l3_cache_size(); }
  int llc_size() const { return DeviceInfo::Global().llc_size(); }
  bool has_dot() const { return DeviceInfo::Global().has_dot(); }
  bool has_fp16() const { return DeviceInfo::Global().has_fp16(); }
  bool has_a53_valid() const { return DeviceInfo::Global().set_a53_valid(); }
  bool has_sve2() const { return DeviceInfo::Global().has_sve2(); }
  bool has_sve2_i8mm() const { return DeviceInfo::Global().has_sve2_i8mm(); }
  bool has_sve2_f32mm() const { return DeviceInfo::Global().has_sve2_f32mm(); }

  template <typename T>
  T* workspace_data() {
    return DeviceInfo::Global().workspace_data<T>();
  }

  bool ExtendWorkspace(size_t size) {
    return DeviceInfo::Global().ExtendWorkspace(size);
  }

#ifdef LITE_WITH_ARM_DNN_LIBRARY
  void* device() { return DeviceInfo::Global().device(); }
  void* context() { return DeviceInfo::Global().context(); }
#endif

  std::string name() const { return "ARMContext"; }
};
#endif

#ifdef LITE_WITH_X86
template <>
class Context<TargetType::kX86> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedTo(X86Context* ctx) {}

  std::string name() const { return "X86Context"; }

  SSEType sse_level() { return device_sse_level(); }
  AVXType avx_level() { return device_avx_level(); }
  FMAType fma_level() { return device_fma_level(); }

 private:
  // overall information
  //
  // kernel information
};
#endif

#ifdef LITE_WITH_OPENCL
template <>
class Context<TargetType::kOpenCL> {
  std::shared_ptr<CLContext> cl_context_{nullptr};

 public:
  CLContext* cl_context() { return cl_context_.get(); }

  void InitOnce() {
    if (CLRuntime::Global()->IsInitSuccess() == false) {
      // gpu is not support , can use cpu instead . do not fatal..
      LOG(ERROR) << "OpenCL runtime init failed";
    }
    cl_context_ = std::make_shared<CLContext>();
  }

  void CopySharedTo(OpenCLContext* ctx) {
    if (ctx && cl_context_) {
      ctx->cl_context_ = cl_context_;
    }
  }
};
#endif

#ifdef LITE_WITH_METAL
template <>
class Context<TargetType::kMetal> {
 public:
  void InitOnce() { context_ = std::make_shared<MetalContext>(); }

  void CopySharedTo(MTLContext* ctx) {
    if (ctx && context_) {
      ctx->context_ = context_;
    }
  }

  MetalContext* context() { return context_.get(); }

 private:
  std::shared_ptr<MetalContext> context_{nullptr};
};
#endif

// Context for running a kernel.
// Holds the necessary resource and information.
class KernelContext {
 public:
  template <typename ContextT>
  ContextT& As() {
    if (!ctx_.valid()) {
      ctx_.set<ContextT>();
    }
    return *ctx_.get_mutable<ContextT>();
  }

 private:
  Any ctx_;
};

// The ContextScheduler helps to assign different context for each kernel.
class ContextScheduler {
 public:
  static ContextScheduler& Global() {
    static auto* x = new ContextScheduler;
    return *x;
  }

  std::unique_ptr<KernelContext> NewContext(
      TargetType target,
      /*only used for cuda context*/ int exec_stream_id = 0) {
    std::unique_ptr<KernelContext> ctx(new KernelContext);
    switch (target) {
      case TARGET(kHost):
        kernel_contexts_[TargetType::kHost].As<HostContext>().CopySharedTo(
            &ctx->As<HostContext>());
        break;
#ifdef LITE_WITH_X86
      case TARGET(kX86):
        kernel_contexts_[TargetType::kX86].As<X86Context>().CopySharedTo(
            &ctx->As<X86Context>());
        break;
#endif
#ifdef LITE_WITH_ARM
      case TARGET(kARM):
        kernel_contexts_[TargetType::kARM].As<ARMContext>().CopySharedTo(
            &ctx->As<ARMContext>());
        break;
#endif
#ifdef LITE_WITH_XPU
      case TARGET(kXPU):
        kernel_contexts_[TargetType::kXPU].As<XPUContext>().CopySharedTo(
            &ctx->As<XPUContext>());
        break;
#endif
#ifdef LITE_WITH_OPENCL
      case TARGET(kOpenCL):
        kernel_contexts_[TargetType::kOpenCL].As<OpenCLContext>().CopySharedTo(
            &ctx->As<OpenCLContext>());
        break;
#endif
#ifdef LITE_WITH_METAL
      case TARGET(kMetal):
        kernel_contexts_[TargetType::kMetal].As<MTLContext>().CopySharedTo(
            &ctx->As<MTLContext>());
        break;
#endif
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
      case TARGET(kNNAdapter):
        kernel_contexts_[TargetType::kNNAdapter]
            .As<NNAdapterContext>()
            .CopySharedTo(&ctx->As<NNAdapterContext>());
        break;
#endif
      default:
#if (!defined LITE_ON_MODEL_OPTIMIZE_TOOL) && (!defined LITE_WITH_PYTHON)
        LOG(FATAL) << "unsupported target " << TargetToStr(target);
#endif
        break;
    }
    return ctx;
  }

 private:
  template <TargetType Type, typename ContextT>
  void InitContext() {
    kernel_contexts_[Type].As<ContextT>().InitOnce();
  }

  ContextScheduler() {
    InitContext<TargetType::kHost, HostContext>();
#ifdef LITE_WITH_X86
    InitContext<TargetType::kX86, X86Context>();
#endif
#ifdef LITE_WITH_ARM
    InitContext<TargetType::kARM, ARMContext>();
#endif
#ifdef LITE_WITH_OPENCL
    InitContext<TargetType::kOpenCL, OpenCLContext>();
#endif
#ifdef LITE_WITH_METAL
    InitContext<TargetType::kMetal, MTLContext>();
#endif
#ifdef LITE_WITH_XPU
    InitContext<TargetType::kXPU, XPUContext>();
#endif
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
    InitContext<TargetType::kNNAdapter, NNAdapterContext>();
#endif
  }

 private:
  std::map<TargetType, KernelContext> kernel_contexts_;
};

}  // namespace lite
}  // namespace paddle
