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

#include "lite/api/paddle_api.h"

#include <utility>

#include "lite/core/context.h"
#include "lite/core/device_info.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

#ifdef LITE_WITH_XPU
#include <functional>
#include <mutex>  // NOLINT
#include "lite/backends/xpu/runtime_option.h"
#include "lite/backends/xpu/target_wrapper.h"
#endif

#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_global.h"
#include "lite/backends/opencl/cl_runtime.h"
#endif

#ifdef LITE_WITH_METAL
#include "lite/backends/metal/target_wrapper.h"
#endif

namespace paddle {
namespace lite_api {

bool EnableOpenCLBackend(bool enable) {
  VLOG(4) << "External EnableOpenCLBackend : " << enable;
#ifdef LITE_WITH_OPENCL
  paddle::lite::ClGlobalDelegate::Global().SetUseOpenCL(enable);
  return enable;
#endif
  return enable;
}

bool IsOpenCLBackendValid(bool check_fp16_valid) {
#ifdef LITE_WITH_OPENCL
  return paddle::lite::ClGlobalDelegate::Global().IsOpenCLBackendValid(
      check_fp16_valid);
#endif
  return false;
}

int GetOpenCLDeviceType() {
#ifdef LITE_WITH_OPENCL
  return paddle::lite::ClGlobalDelegate::Global().GetOpenCLDeviceType();
#endif
  return -1;
}

Tensor::Tensor(void *raw) : raw_tensor_(raw) {}

// TODO(Superjomn) refine this by using another `const void* const_raw`;
Tensor::Tensor(const void *raw) { raw_tensor_ = const_cast<void *>(raw); }

lite::Tensor *tensor(void *x) { return static_cast<lite::Tensor *>(x); }
const lite::Tensor *ctensor(void *x) {
  return static_cast<const lite::Tensor *>(x);
}

void Tensor::Resize(const shape_t &shape) {
  tensor(raw_tensor_)->Resize(shape);
}

bool Tensor::IsInitialized() const {
  return tensor(raw_tensor_)->IsInitialized();
}

template <typename T>
const T *Tensor::data() const {
  return ctensor(raw_tensor_)->data<T>();
}

void Tensor::ShareExternalMemory(void *data,
                                 size_t memory_size,
                                 TargetType target) {
  auto buf =
      std::make_shared<lite::Buffer>(lite::Buffer(data, target, memory_size));
  tensor(raw_tensor_)->ResetBuffer(buf, memory_size);
}

template <typename T>
T *Tensor::mutable_data(TargetType type) const {
  return tensor(raw_tensor_)->mutable_data<T>(type);
}

void *Tensor::mutable_metal_data(void *ptr) const {
#ifdef LITE_WITH_METAL
  return tensor(raw_tensor_)->mutable_metal_data(ptr);
#else
  return nullptr;
#endif
}

template const double *Tensor::data<double>() const;
template const float *Tensor::data<float>() const;
template const int64_t *Tensor::data<int64_t>() const;
template const int32_t *Tensor::data<int32_t>() const;
template const int16_t *Tensor::data<int16_t>() const;
template const int8_t *Tensor::data<int8_t>() const;
template const uint16_t *Tensor::data<uint16_t>() const;
template const uint8_t *Tensor::data<uint8_t>() const;
template const bool *Tensor::data<bool>() const;
template const void *Tensor::data<void>() const;

template double *Tensor::mutable_data(TargetType type) const;
template float *Tensor::mutable_data(TargetType type) const;
template int64_t *Tensor::mutable_data(TargetType type) const;
template int *Tensor::mutable_data(TargetType type) const;
template int16_t *Tensor::mutable_data(TargetType type) const;
template int8_t *Tensor::mutable_data(TargetType type) const;
template uint16_t *Tensor::mutable_data(TargetType type) const;
template uint8_t *Tensor::mutable_data(TargetType type) const;
template bool *Tensor::mutable_data(TargetType type) const;
#ifdef ENABLE_ARM_FP16
template const __fp16 *Tensor::data<__fp16>() const;
template __fp16 *Tensor::mutable_data(TargetType type) const;
#endif

template <typename T, TargetType type>
void Tensor::CopyFromCpu(const T *src_data) {
  T *data = tensor(raw_tensor_)->mutable_data<T>(type);
  int64_t num = tensor(raw_tensor_)->numel();
  CHECK(num > 0) << "You should call Resize interface first";
  if (type == TargetType::kHost || type == TargetType::kARM) {
    lite::TargetWrapperHost::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoH);
  } else if (type == TargetType::kMetal) {
#ifdef LITE_WITH_METAL
    lite::TargetWrapperMetal::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoD);
#else
    LOG(FATAL) << "Please compile the lib with METAL.";
#endif
  } else {
    LOG(FATAL) << "The CopyFromCpu interface just support kHost, kARM";
  }
}
template <typename T>
void Tensor::CopyToCpu(T *data) const {
  const T *src_data = tensor(raw_tensor_)->data<T>();
  int64_t num = tensor(raw_tensor_)->numel();
  if (num == 0) {
    LOG(WARNING) << "Tensor does not hold data.";
    return;
  }
  auto type = tensor(raw_tensor_)->target();
  if (type == TargetType::kHost || type == TargetType::kARM) {
    lite::TargetWrapperHost::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoH);
  } else if (type == TargetType::kMetal) {
#ifdef LITE_WITH_METAL
    lite::TargetWrapperMetal::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoD);
#else
    LOG(FATAL) << "Please compile the lib with METAL.";
#endif
  } else {
    LOG(FATAL) << "The CopyToCpu interface just support kHost, kARM";
  }
}

template void Tensor::CopyFromCpu<int, TargetType::kHost>(const int *);
template void Tensor::CopyFromCpu<int64_t, TargetType::kHost>(const int64_t *);
template void Tensor::CopyFromCpu<float, TargetType::kHost>(const float *);
template void Tensor::CopyFromCpu<int8_t, TargetType::kHost>(const int8_t *);
template void Tensor::CopyFromCpu<uint8_t, TargetType::kHost>(const uint8_t *);

template void Tensor::CopyFromCpu<int, TargetType::kARM>(const int *);
template void Tensor::CopyFromCpu<int64_t, TargetType::kARM>(const int64_t *);
template void Tensor::CopyFromCpu<float, TargetType::kARM>(const float *);
template void Tensor::CopyFromCpu<int8_t, TargetType::kARM>(const int8_t *);
template void Tensor::CopyFromCpu<uint8_t, TargetType::kARM>(const uint8_t *);

template void Tensor::CopyToCpu(float *) const;
template void Tensor::CopyToCpu(int *) const;
template void Tensor::CopyToCpu(int64_t *) const;
template void Tensor::CopyToCpu(int8_t *) const;
template void Tensor::CopyToCpu(uint8_t *) const;

shape_t Tensor::shape() const {
  return ctensor(raw_tensor_)->dims().Vectorize();
}

TargetType Tensor::target() const {
  auto type = ctensor(raw_tensor_)->target();
  if (type == TargetType::kUnk) {
    CHECK(false) << "This tensor was not initialized.";
  }
  return type;
}

PrecisionType Tensor::precision() const {
  auto precision = ctensor(raw_tensor_)->precision();
  if (precision == PrecisionType::kUnk) {
    CHECK(false) << "This tensor was not initialized.";
  }
  return precision;
}

void Tensor::SetPrecision(PrecisionType precision) {
  tensor(raw_tensor_)->set_precision(precision);
}

lod_t Tensor::lod() const { return ctensor(raw_tensor_)->lod(); }

void Tensor::SetLoD(const lod_t &lod) { tensor(raw_tensor_)->set_lod(lod); }

std::unique_ptr<Tensor> PaddlePredictor::GetMutableTensor(
    const std::string &name) {
  LOG(FATAL)
      << "The GetMutableTensor API is only supported by CxxConfig predictor.";
  return nullptr;
}

std::vector<std::string> PaddlePredictor::GetParamNames() {
  std::vector<std::string> null_result = {};
  LOG(FATAL)
      << "The GetParamNames API is only supported by CxxConfig predictor.";
  return null_result;
}

void PaddlePredictor::SaveOptimizedModel(const std::string &model_dir,
                                         LiteModelType model_type,
                                         bool record_info) {
  LOG(FATAL)
      << "The SaveOptimizedModel API is only supported by CxxConfig predictor.";
}

template <typename ConfigT>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT &) {
  return std::shared_ptr<PaddlePredictor>();
}

ConfigBase::ConfigBase(PowerMode mode, int threads) {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Init();
  lite::DeviceInfo::Global().SetRunMode(mode, threads);
  mode_ = lite::DeviceInfo::Global().mode();
  threads_ = lite::DeviceInfo::Global().threads();
#endif
#ifdef LITE_WITH_XPU
  std::shared_ptr<void> runtime_option =
      std::shared_ptr<lite::XPURunTimeOption>(new lite::XPURunTimeOption);
  target_configs_.emplace(TARGET(kXPU), std::move(runtime_option));
#endif
}

void ConfigBase::set_opencl_binary_path_name(const std::string &path,
                                             const std::string &name) {
#ifdef LITE_WITH_OPENCL
  if (paddle::lite_api::IsOpenCLBackendValid()) {
    opencl_bin_path_ = path;
    opencl_bin_name_ = name;
    lite::CLRuntime::Global()->SetBinaryPathName(path, name);
#ifdef LITE_WITH_LOG
    LOG(INFO) << "opencl binary path and file name:"
              << (lite::CLRuntime::Global()->GetBinaryPathName())[0] << "/"
              << (lite::CLRuntime::Global()->GetBinaryPathName())[1];
#endif
  }
#endif
}

void ConfigBase::set_opencl_tune(CLTuneMode tune_mode,
                                 const std::string &path,
                                 const std::string &name,
                                 size_t lws_repeats) {
#ifdef LITE_WITH_OPENCL
  if (paddle::lite_api::IsOpenCLBackendValid()) {
    opencl_tune_mode_ = tune_mode;
    paddle::lite::CLRuntime::Global()->set_auto_tune(
        opencl_tune_mode_, path, name, lws_repeats);
#ifdef LITE_WITH_LOG
    LOG(INFO) << "set opencl_tune_mode: "
              << CLTuneModeToStr(lite::CLRuntime::Global()->auto_tune())
              << ", lws_repeats:" << lws_repeats;
    LOG(INFO) << "tuned file path & name:" << path << "/" << name;
#endif
  }
#endif
}

void ConfigBase::set_opencl_precision(CLPrecisionType p) {
#ifdef LITE_WITH_OPENCL
  if (paddle::lite_api::IsOpenCLBackendValid()) {
    opencl_precision_ = p;
    paddle::lite::CLRuntime::Global()->set_precision(p);
#ifdef LITE_WITH_LOG
    LOG(INFO) << "set opencl precision: "
              << CLPrecisionTypeToStr(
                     lite::CLRuntime::Global()->get_precision());
#endif
  }
#endif
}

void ConfigBase::set_power_mode(paddle::lite_api::PowerMode mode) {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Global().SetRunMode(mode, threads_);
  mode_ = lite::DeviceInfo::Global().mode();
  threads_ = lite::DeviceInfo::Global().threads();
#endif
}

void ConfigBase::set_threads(int threads) {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Global().SetRunMode(mode_, threads);
  mode_ = lite::DeviceInfo::Global().mode();
  threads_ = lite::DeviceInfo::Global().threads();
#endif
}

void ConfigBase::set_metal_device(void *device) {
#ifdef LITE_WITH_METAL
  metal_device_ = device;
#endif
  return;
}

void ConfigBase::set_metal_lib_path(const std::string &path) {
#ifdef LITE_WITH_METAL
  metal_path_ = path;
#endif
  return;
}

void ConfigBase::set_metal_use_mps(bool flag) {
#ifdef LITE_WITH_METAL
  metal_use_mps_ = flag;
#endif
  return;
}

void ConfigBase::set_metal_use_aggressive(bool flag) {
#ifdef LITE_WITH_METAL
  metal_use_aggressive_ = flag;
#endif
  return;
}

void ConfigBase::set_metal_use_memory_reuse(bool flag) {
#ifdef LITE_WITH_METAL
  metal_use_memory_reuse_ = flag;
#endif
  return;
}

void ConfigBase::add_discarded_pass(const std::string pass) {
  discarded_passes_.push_back(pass);
  return;
}

// Set external custom allocator
void ConfigBase::set_custom_allocator(TargetType target_type,
                                      CustomAllocator custom_allocator) {
  // TODO(shentanyue): TargetType will be supported in the future.
  lite::Allocator::Global().SetCustomAllocator(custom_allocator);
}

#ifdef LITE_WITH_X86
void ConfigBase::set_x86_math_num_threads(int threads) {
  x86_math_num_threads_ = threads;
}
int ConfigBase::x86_math_num_threads() const { return x86_math_num_threads_; }
#endif

void ConfigBase::set_subgraph_model_cache_buffers(
    const std::string &key,
    const std::vector<char> &cfg,
    const std::vector<char> &bin) {
  CHECK(!key.empty());
  CHECK(!cfg.empty());
  CHECK(!bin.empty());
  CHECK_EQ(subgraph_model_cache_buffers_.count(key), 0);
  subgraph_model_cache_buffers_[key] =
      std::pair<std::vector<char>, std::vector<char>>(cfg, bin);
}

bool ConfigBase::check_nnadapter_device_name(
    const std::string &nnadapter_device_name) {
  bool found = false;
#ifdef LITE_WITH_NNADAPTER
  found = lite::Context<TargetType::kNNAdapter>::CheckNNAdapterDeviceName(
      nnadapter_device_name);
#else
  LOG(WARNING) << "The invoking of the function 'check_nnadapter_device' is "
                  "ignored, please rebuild it with LITE_WITH_NNADAPTER=ON.";
#endif
  return found;
}

void ConfigBase::set_nnadapter_model_cache_buffers(
    const std::string &model_cache_token,
    const std::vector<char> &model_cache_buffer) {
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
  CHECK(!model_cache_token.empty());
  CHECK(!model_cache_buffer.empty());
  CHECK_EQ(nnadapter_model_cache_buffers_.count(model_cache_token), 0);
  nnadapter_model_cache_buffers_[model_cache_token] = model_cache_buffer;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_nnadapter_model_cache_buffers' is ignored, please "
                  "rebuild it with LITE_WITH_NNADAPTER=ON.";
#endif
}

// **DEPRECATED**, use set_xpu_l3_cache_method() in the future
void ConfigBase::set_xpu_workspace_l3_size_per_thread(int l3_size) {
#ifdef LITE_WITH_XPU
  ConfigBase::set_xpu_l3_cache_method(l3_size, false);
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_workspace_l3_size_per_thread' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

// local_l3 <= 0 , locked == false: NO USE L3
// local_l3 > 0, locked == false : USE local l3
// locked == true : USE Shared L3
// default : locked = false, local_l3 = max_l3_size;
void ConfigBase::set_xpu_l3_cache_method(size_t l3_size, bool locked) {
#ifdef LITE_WITH_XPU
  static std::mutex set_l3_mutex;
  const std::lock_guard<std::mutex> lock(set_l3_mutex);
  if (locked) {
    if (!lite::TargetWrapperXPU::IsSharedL3Created()) {
      lite::TargetWrapperXPU::shared_l3_size =
          lite::TargetWrapperXPU::shared_l3_size > l3_size
              ? lite::TargetWrapperXPU::shared_l3_size
              : l3_size;
    } else {
      CHECK(lite::TargetWrapperXPU::shared_l3_size >= l3_size)
          << "Enlarge XPU Shared L3 Cache Is Not Allowed.";
    }
    reinterpret_cast<lite::XPURunTimeOption *>(
        target_configs()[TARGET(kXPU)].get())
        ->xpu_local_l3_size = 0;
    lite::TargetWrapperXPU::need_l3_mutex = true;
  } else {
    reinterpret_cast<lite::XPURunTimeOption *>(
        target_configs()[TARGET(kXPU)].get())
        ->xpu_local_l3_size = l3_size;
    lite::TargetWrapperXPU::need_l3_mutex = false;
  }
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_l3_cache_method' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_l3_cache_autotune(bool autotune) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_local_l3_autotune = autotune;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_l3_cache_autotune' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_gm_workspace_method(size_t gm_size) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_local_gm_size = gm_size;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_gm_workspace_method' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_dev_per_thread(int dev_no) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_dev_num = dev_no;
#else
  LOG(WARNING) << "The invoking of the function 'set_xpu_dev_per_thread' is "
                  "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::enable_xpu_multi_stream() {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_enable_multi_stream = true;
#else
  LOG(WARNING)
      << "The invoking of the function 'enable_xpu_stream_per_thread' is "
         "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

// **DEPRECATED**, use set_xpu_multi_encoder_method() in the future
void ConfigBase::set_xpu_multi_encoder_precision(const std::string &precision) {
#ifdef LITE_WITH_XPU
  ConfigBase::set_xpu_multi_encoder_method(precision, false);
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_multi_encoder_precision' is "
                  "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_multi_encoder_method(const std::string &precision,
                                              bool adaptive_seqlen) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->multi_encoder_precision = precision;
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->multi_encoder_adaptive_seqlen = adaptive_seqlen;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_multi_encoder_method' is "
                  "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_local_quant(bool local_quant) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->local_quant = local_quant;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_local_quant' is "
                  "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_compute_precision(const std::string &precision) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->compute_precision = precision;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'xpu_compute_precision' is "
                  "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_conv_autotune(bool autotune,
                                       const std::string &autotune_file) {
#ifdef LITE_WITH_XPU
  LOG(WARNING)
      << "This function "
         "'set_xpu_conv_autotune' is deprecated, "
         "if you want to use autotune, please refer to "
         "http://agroup.baidu.com/share/md/f9233d84df11452488a1fdd4f859647f";

#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_conv_autotune' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_cluster_num(const int num) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_cluster_num = num;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_cluster_num' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_sdnn_num(const int num) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_sdnn_num = num;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_sdnn_num' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_dump_tensor_path(const std::string &dump_tensor_path) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_dump_tensor_path = dump_tensor_path;
  add_discarded_pass("xpu_memory_optimize_pass");
  add_discarded_pass("memory_optimize_pass");
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_dump_tensor_path' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void ConfigBase::set_xpu_dump_log_path(const std::string &dump_log_path) {
#ifdef LITE_WITH_XPU
  reinterpret_cast<lite::XPURunTimeOption *>(
      target_configs()[TARGET(kXPU)].get())
      ->xpu_dump_log_path = dump_log_path;
  add_discarded_pass("xpu_memory_optimize_pass");
  add_discarded_pass("memory_optimize_pass");
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_dump_log_path' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

// user's XpuConfig -> configbase's XPURunTimeOption
void ConfigBase::set_xpu_config(const XpuConfig &xpu_config) {
#ifdef LITE_WITH_XPU
  lite::XPURunTimeOption *xpu_runtime_opt =
      reinterpret_cast<lite::XPURunTimeOption *>(
          target_configs()[TARGET(kXPU)].get());
  CHECK_GE(xpu_config.device_id, 0) << "xpu_config.device_id should >= 0";
  xpu_runtime_opt->xpu_dev_num = xpu_config.device_id;
  // just use local L3 , no lock
  set_xpu_l3_cache_method(xpu_config.l3_size, false);
  if (xpu_config.l3_ptr) {
    LOG(WARNING) << "lite ignore the pre allocated L3 buffer, and malloc "
                    "itself on every inference";
  }
  if (xpu_config.stream) {
    xpu_runtime_opt->xpu_stream.SetXPUStream(xpu_config.stream);
  }
  // l3_autotune_size is for lite framework, the remaining is for api_l3_reserve
  CHECK_LE(xpu_config.l3_autotune_size, xpu_config.l3_size)
      << "the l3_autotune_size should not greater than l3_size";
  xpu_runtime_opt->api_l3_reserve =
      xpu_config.l3_size - xpu_config.l3_autotune_size;
  if (std::getenv("API_L3_SIZE_RES") || std::getenv("XPU_FC_AUTOTUNE") ||
      std::getenv("XPU_FC_AUTOTUNE_FILE") ||
      std::getenv("XPU_FC_AUTOTUNE_WRITEBACK") ||
      std::getenv("XPU_PRECISION_MODE") || std::getenv("XPU_LOCAL_QUANT") ||
      std::getenv("XPU_ENCODER_PRECISION") ||
      std::getenv("QUANT_GELU_OUT_THRESHOLD")) {
    LOG(WARNING) << "envs(except XPU_VISIBLE_DEVICES) are redundant when using "
                    "set_xpu_config";
  }
  if (xpu_config.stream) {
    LOG(WARNING) << "lite ignore pre allocated stream, and create stream "
                    "itself when needed";
  }
  CHECK_GE(xpu_config.fc_autotune_level, 0) << "fc_autotune_level should >= 0";
  CHECK_LT(xpu_config.fc_autotune_level, 10) << "fc_autotune_level should < 10";
  xpu_runtime_opt->xpu_fc_autotune_level = xpu_config.fc_autotune_level;
  if (!xpu_config.fc_autotune_file.empty()) {
    xpu_runtime_opt->xpu_fc_autotune_file = xpu_config.fc_autotune_file;
  }
  if (xpu_config.fc_autotune_file_writeback) {
    CHECK(!xpu_config.fc_autotune_file.empty())
        << "need config fc_autotune_file if fc_autotune_file_writeback ON";
    xpu_runtime_opt->xpu_fc_autotune_writeback =
        xpu_config.fc_autotune_file_writeback;
  }
  // to indicate the un-quanted fc's compute precision
  std::vector<std::string> encoder_compute_precision{"int8", "int16", "int31"};
  CHECK(xpu_config.gemm_compute_precision >= 0 &&
        xpu_config.gemm_compute_precision < 3)
      << "gemm_compute_precision should be [0/1/2]";
  set_xpu_multi_encoder_method(
      encoder_compute_precision[xpu_config.gemm_compute_precision],
      xpu_config.transformer_encoder_adaptive_seqlen);
  xpu_runtime_opt->transformer_softmax_optimize_level =
      xpu_config.transformer_softmax_optimize_level;
  xpu_runtime_opt->quant_post_static_gelu_out_threshold =
      xpu_config.quant_post_static_gelu_out_threshold;
  CHECK(xpu_config.quant_post_dynamic_activation_method >= 0 &&
        xpu_config.quant_post_dynamic_activation_method < 3)
      << "quant_post_dynamic_activation_method should be [0/1/2]";
  // kunlun1: 0 per tensor, 1 per batch, 2 per head
  // kunlun2: 0 per tensor, non-zero local quant
  xpu_runtime_opt->local_quant =
      xpu_config.quant_post_dynamic_activation_method > 0;
  xpu_runtime_opt->quant_post_dynamic_activation_method =
      xpu_config.quant_post_dynamic_activation_method;
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_config' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

CxxModelBuffer::CxxModelBuffer(const char *program_buffer,
                               size_t program_buffer_size,
                               const char *params_buffer,
                               size_t params_buffer_size) {
  program_ = std::string(program_buffer, program_buffer + program_buffer_size);
  params_ = std::string(params_buffer, params_buffer + params_buffer_size);
}

CxxModelBuffer::CxxModelBuffer(std::string &&program_buffer,
                               std::string &&params_buffer) {
  program_ = std::forward<std::string>(program_buffer);
  params_ = std::forward<std::string>(params_buffer);
}

const std::string &CxxModelBuffer::get_program() const {
  CHECK(!program_.empty());
  return program_;
}

const std::string &CxxModelBuffer::get_params() const { return params_; }

bool CxxModelBuffer::is_empty() const { return program_.empty(); }

const CxxModelBuffer &CxxConfig::get_model_buffer() const {
  CHECK(model_buffer_) << "Cannot get an empty model buffer.";
  return *model_buffer_;
}

template <class T>
void CxxConfig::set_preferred_inputs_for_warmup(const int group_idx,
                                                const int tensor_idx,
                                                const shape_t &shape,
                                                const lod_t &lod,
                                                const T fill_value,
                                                const void *data) {
#ifdef LITE_WITH_XPU
  if (preferred_inputs_for_warmup_.count(group_idx) == 0) {
    preferred_inputs_for_warmup_[group_idx] =
        std::vector<std::shared_ptr<void>>{};
  }
  auto &input_tensors = preferred_inputs_for_warmup_[group_idx];
  while (input_tensors.size() < tensor_idx + 1) {
    std::shared_ptr<void> input_tensor(
        static_cast<void *>(new lite::Tensor),
        [](void *x) { delete static_cast<lite::Tensor *>(x); });
    input_tensors.emplace_back(input_tensor);
  }

  auto input_tensor =
      static_cast<lite::Tensor *>(input_tensors[tensor_idx].get());
  input_tensor->Resize(shape);
  input_tensor->set_lod(lod);
  auto input_data = input_tensor->mutable_data<T>();
  int64_t size = std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  if (data != nullptr) {
    memcpy(input_data, data, sizeof(T) * size);
  } else {
    for (int64_t i = 0; i < size; i++) {
      input_data[i] = fill_value;
    }
  }
#else
  LOG(WARNING)
      << "'set_preferred_inputs_for_warmup' is only for xpu now, please "
         "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

#define _SetPreferredInputsForWarmup(dtype)                        \
  template void CxxConfig::set_preferred_inputs_for_warmup<dtype>( \
      const int group_idx,                                         \
      const int tensor_idx,                                        \
      const shape_t &shape,                                        \
      const lod_t &lod,                                            \
      const dtype fill_value,                                      \
      const void *data);

_SetPreferredInputsForWarmup(float);
_SetPreferredInputsForWarmup(double);
_SetPreferredInputsForWarmup(int32_t);
_SetPreferredInputsForWarmup(int64_t);
#undef _SetPreferredInputsForWarmup

// set model data in combined format, `set_model_from_file` refers to loading
// model from file, set_model_from_buffer refers to loading model from memory
// buffer
void MobileConfig::set_model_from_file(const std::string &x) {
  lite_model_file_ = x;
}

void MobileConfig::set_model_from_buffer(const std::string &x) {
  lite_model_file_ = x;
  model_from_memory_ = true;
}

void MobileConfig::set_model_from_buffer(std::string &&x) {
  lite_model_file_.assign(std::forward<std::string>(x));
  model_from_memory_ = true;
}

void MobileConfig::set_model_from_buffer(const char *buffer, size_t length) {
  lite_model_buffer_ptr_ = buffer;
  lite_model_buffer_size_ = length;
  model_from_memory_ = true;
}

void MobileConfig::set_model_buffer(const char *model_buffer,
                                    size_t model_buffer_size,
                                    const char *param_buffer,
                                    size_t param_buffer_size) {
  LOG(WARNING) << "warning: `set_model_buffer` will be abandened in "
                  "release/v3.0.0, new method `set_model_from_buffer(const "
                  "std::string &x)` is recommended.";
  model_buffer_ = std::string(model_buffer, model_buffer + model_buffer_size);
  param_buffer_ = std::string(param_buffer, param_buffer + param_buffer_size);
  model_from_memory_ = true;
}

// This is the method for allocating workspace_size according to L3Cache size
void MobileConfig::SetArmL3CacheSize(L3CacheSetMethod method,
                                     int absolute_val) {
#ifdef LITE_WITH_ARM
  lite::DeviceInfo::Global().SetArmL3CacheSize(method, absolute_val);
#endif
}

// This is the method for check fp16 instruction is valid
bool MobileConfig::check_fp16_valid() {
#ifdef LITE_WITH_ARM
  return lite::DeviceInfo::Global().has_fp16();
#else
  return false;
#endif
}

}  // namespace lite_api
}  // namespace paddle
