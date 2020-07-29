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
#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/context.h"
#endif
#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_context.h"
#include "lite/backends/opencl/cl_runtime.h"
#endif
#ifdef LITE_WITH_MLU
#include <cnml.h>
#include <cnrt.h>
#include <mutex>  // NOLINT
#include "lite/backends/mlu/mlu_utils.h"
#endif
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/xpu_header_sitter.h"
#endif

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

namespace paddle {
namespace lite {

template <TargetType Type>
class Context;

using HostContext = Context<TargetType::kHost>;
using X86Context = Context<TargetType::kX86>;
using ARMContext = Context<TargetType::kARM>;
using NPUContext = Context<TargetType::kNPU>;
using APUContext = Context<TargetType::kAPU>;
using XPUContext = Context<TargetType::kXPU>;
using OpenCLContext = Context<TargetType::kOpenCL>;
using FPGAContext = Context<TargetType::kFPGA>;
using BMContext = Context<TargetType::kBM>;
using MLUContext = Context<TargetType::kMLU>;
using RKNPUContext = Context<TargetType::kRKNPU>;
using HuaweiAscendNPUContext = Context<TargetType::kHuaweiAscendNPU>;

template <>
class Context<TargetType::kHost> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedTo(HostContext* ctx) {}

  std::string name() const { return "HostContext"; }
};

#ifdef LITE_WITH_NPU
template <>
class Context<TargetType::kNPU> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}
  void CopySharedTo(NPUContext* ctx) {}

  NPUContext& operator=(const NPUContext& ctx) {}
  std::string name() const { return "NPUContext"; }

  static void SetSubgraphModelCacheDir(Scope* scope,
                                       std::string subgraph_model_cache_dir) {
    auto var = scope->Var("SUBGRAPH_MODEL_CACHE_DIR");
    CHECK(var);
    auto data = var->GetMutable<std::string>();
    CHECK(data);
    *data = subgraph_model_cache_dir;
  }
  static std::string SubgraphModelCacheDir(Scope* scope) {
    auto var = scope->FindVar("SUBGRAPH_MODEL_CACHE_DIR");
    if (!var) return "";
    return var->Get<std::string>();
  }
};
#endif

#ifdef LITE_WITH_HUAWEI_ASCEND_NPU
template <>
class Context<TargetType::kHuaweiAscendNPU> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}
  void CopySharedTo(HuaweiAscendNPUContext* ctx) {}

  HuaweiAscendNPUContext& operator=(const HuaweiAscendNPUContext& ctx) {
    return *this;
  }
  std::string name() const { return "HuaweiAscendNPUContext"; }

  static void SetSubgraphModelCacheDir(std::string subgraph_model_cache_dir) {
    subgraph_model_cache_dir_ = subgraph_model_cache_dir;
  }
  static std::string SubgraphModelCacheDir() {
    return subgraph_model_cache_dir_;
  }

  static void SetHuaweiAscendDeviceID(int huawei_ascend_device_id) {
    huawei_ascend_device_id_ = huawei_ascend_device_id;
  }
  static int HuaweiAscendDeviceID() { return huawei_ascend_device_id_; }

 private:
  static thread_local std::string subgraph_model_cache_dir_;
  static thread_local int huawei_ascend_device_id_;
};
#endif

#ifdef LITE_WITH_APU
template <>
class Context<TargetType::kAPU> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}
  void CopySharedTo(APUContext* ctx) {}

  APUContext& operator=(const APUContext& ctx) {}
  std::string name() const { return "APUContext"; }
};
#endif

#ifdef LITE_WITH_BM
template <>
class Context<TargetType::kBM> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() { TargetWrapperBM::SetDevice(TargetWrapperBM::GetDevice()); }
  void CopySharedTo(BMContext* ctx) {}
  void* GetHandle() { return TargetWrapperBM::GetHandle(); }

  std::string name() const { return "BMContext"; }
};
#endif

#ifdef LITE_WITH_RKNPU
template <>
class Context<TargetType::kRKNPU> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}
  void CopySharedTo(RKNPUContext* ctx) {}

  RKNPUContext& operator=(const RKNPUContext& ctx) {}
  std::string name() const { return "RKNPUContext"; }
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

  template <typename T>
  T* workspace_data() {
    return DeviceInfo::Global().workspace_data<T>();
  }

  bool ExtendWorkspace(size_t size) {
    return DeviceInfo::Global().ExtendWorkspace(size);
  }

  std::string name() const { return "ARMContext"; }
};
#endif

#ifdef LITE_WITH_FPGA
// TODO(tianxiaogang): add needed implementation to context
template <>
class Context<TargetType::kFPGA> {
 public:
  void InitOnce() {}

  FPGAContext& operator=(const FPGAContext& ctx) {}

  void CopySharedTo(FPGAContext* ctx) {}

  std::string name() const { return "FPGAContext"; }
};
#endif

#ifdef LITE_WITH_MLU
template <>
class Context<TargetType::kMLU> {
 public:
  typename Env<TargetType::kMLU>::Devs& devs = Env<TargetType::kMLU>::Global();

  void InitOnce() {}

  MLUContext& operator=(const MLUContext& ctx) {
    this->Init(ctx.device_id_, ctx.exec_queue_id_);
    return *this;
  }

  void Init(int dev_id, int exec_queue_id = 0) {
    CHECK_GT(devs.size(), 0UL)
        << "Env is not initialized or current target is not exit!";
    if (dev_id >= static_cast<int>(devs.size())) {
      LOG(WARNING) << "device index exceeds the number of devices, set to "
                      "default device(0)!";
      device_id_ = 0;
    } else {
      device_id_ = dev_id;
    }
    SetMluDevice(device_id_);

    // get queue id from map
    std::unique_lock<std::mutex> lk(map_mutex_);
    if (queue_id_map_.find(exec_queue_id) == queue_id_map_.end()) {
      queue_id_map_[exec_queue_id] =
          next_queue_id_++ % devs[dev_id].max_queue();
    }
    exec_queue_id_ = queue_id_map_[exec_queue_id];
    VLOG(4) << "pick mlu queue id: " << exec_queue_id_;
    lk.unlock();

    io_queue_ = devs[dev_id].io_queues()[exec_queue_id_];
    exec_queue_ = devs[dev_id].exec_queues()[exec_queue_id_];
  }

  void CopySharedTo(MLUContext* ctx) { ctx->forward_param_ = forward_param_; }

  const cnrtQueue_t& exec_queue() const { return exec_queue_; }
  void SetExecQueue(cnrtQueue_t queue) { exec_queue_ = queue; }

  const cnrtQueue_t& io_queue() const { return io_queue_; }
  void SetIoQueue(cnrtQueue_t queue) { io_queue_ = queue; }

  cnmlCoreVersion_t MLUCoreVersion() {
    return paddle::lite::TargetWrapperMlu::MLUCoreVersion();
  }

  int MLUCoreNumber() {
    return paddle::lite::TargetWrapperMlu::MLUCoreNumber();
  }

  u32_t affinity() { return affinity_; }

  cnrtInvokeFuncParam_t forward_param() { return forward_param_; }

  int device_id() { return device_id_; }

  std::string name() const { return "MLUContext"; }

 private:
  static int next_queue_id_;
  static std::map<int, int> queue_id_map_;
  static std::mutex map_mutex_;
  int device_id_;
  // overall information
  int exec_queue_id_;
  cnrtQueue_t io_queue_;
  cnrtQueue_t exec_queue_;

  std::vector<cnrtNotifier_t> input_notifiers_;
  std::vector<cnrtNotifier_t> output_notifiers_;

  cnrtInvokeFuncParam_t forward_param_;
  u32_t affinity_ = 0x01;
};
#endif  // LITE_WITH_MLU

#ifdef LITE_WITH_X86
template <>
class Context<TargetType::kX86> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedTo(X86Context* ctx) {}

  std::string name() const { return "X86Context"; }

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
#ifdef LITE_WITH_CUDA
      case TARGET(kCUDA): {
        int dev_id = TargetWrapper<TargetType::kCUDA>::GetCurDevice();
        auto& context = ctx->As<CUDAContext>();
        context.Init(dev_id, exec_stream_id);
        kernel_contexts_[TargetType::kCUDA].As<CUDAContext>().CopySharedTo(
            &context);
      } break;
#endif
#ifdef LITE_WITH_ARM
      case TARGET(kARM):
        kernel_contexts_[TargetType::kARM].As<ARMContext>().CopySharedTo(
            &ctx->As<ARMContext>());
        break;
#endif
#ifdef LITE_WITH_NPU
      case TARGET(kNPU):
        kernel_contexts_[TargetType::kNPU].As<NPUContext>().CopySharedTo(
            &ctx->As<NPUContext>());
        break;
#endif
#ifdef LITE_WITH_HUAWEI_ASCEND_NPU
      case TARGET(kHuaweiAscendNPU):
        kernel_contexts_[TargetType::kHuaweiAscendNPU]
            .As<HuaweiAscendNPUContext>()
            .CopySharedTo(&ctx->As<HuaweiAscendNPUContext>());
        break;
#endif
#ifdef LITE_WITH_APU
      case TARGET(kAPU):
        kernel_contexts_[TargetType::kAPU].As<APUContext>().CopySharedTo(
            &ctx->As<APUContext>());
        break;
#endif
#ifdef LITE_WITH_RKNPU
      case TARGET(kRKNPU):
        kernel_contexts_[TargetType::kRKNPU].As<RKNPUContext>().CopySharedTo(
            &ctx->As<RKNPUContext>());
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
#ifdef LITE_WITH_FPGA
      case TARGET(kFPGA):
        kernel_contexts_[TargetType::kFPGA].As<FPGAContext>().CopySharedTo(
            &ctx->As<FPGAContext>());
        break;
#endif
#ifdef LITE_WITH_BM
      case TARGET(kBM):
        kernel_contexts_[TargetType::kBM].As<BMContext>().CopySharedTo(
            &ctx->As<BMContext>());
        break;
#endif
#ifdef LITE_WITH_MLU
      case TARGET(kMLU): {
        int dev_id = TargetWrapper<TargetType::kMLU>::GetCurDevice();
        auto& context = ctx->As<MLUContext>();
        context.Init(dev_id, exec_stream_id);
        kernel_contexts_[TargetType::kMLU].As<MLUContext>().CopySharedTo(
            &context);
        LOG(INFO) << "New Context for MLU";
      } break;
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
#ifdef LITE_WITH_CUDA
    InitContext<TargetType::kCUDA, CUDAContext>();
#endif
#ifdef LITE_WITH_ARM
    InitContext<TargetType::kARM, ARMContext>();
#endif
#ifdef LITE_WITH_OPENCL
    InitContext<TargetType::kOpenCL, OpenCLContext>();
#endif
#ifdef LITE_WITH_FPGA
    InitContext<TargetType::kFPGA, FPGAContext>();
#endif
#ifdef LITE_WITH_NPU
    InitContext<TargetType::kNPU, NPUContext>();
#endif
#ifdef LITE_WITH_HUAWEI_ASCEND_NPU
    InitContext<TargetType::kHuaweiAscendNPU, HuaweiAscendNPUContext>();
#endif
#ifdef LITE_WITH_APU
    InitContext<TargetType::kAPU, APUContext>();
#endif
#ifdef LITE_WITH_RKNPU
    InitContext<TargetType::kRKNPU, RKNPUContext>();
#endif
#ifdef LITE_WITH_XPU
    InitContext<TargetType::kXPU, XPUContext>();
#endif
#ifdef LITE_WITH_BM
    InitContext<TargetType::kBM, BMContext>();
#endif
#ifdef LITE_WITH_MLU
    InitContext<TargetType::kMLU, MLUContext>();
#endif
  }

 private:
  std::map<TargetType, KernelContext> kernel_contexts_;
};

}  // namespace lite
}  // namespace paddle
