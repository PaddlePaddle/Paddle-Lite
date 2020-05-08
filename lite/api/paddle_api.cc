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
#include "lite/core/context.h"
#include "lite/core/device_info.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/target_wrapper.h"
#endif

namespace paddle {
namespace lite_api {

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

// Tensor::data
template <>
const float *Tensor::data() const {
  return ctensor(raw_tensor_)->data<float>();
}
template <>
const int8_t *Tensor::data() const {
  return ctensor(raw_tensor_)->data<int8_t>();
}
template <>
const uint8_t *Tensor::data() const {
  return ctensor(raw_tensor_)->data<uint8_t>();
}
template <>
const int64_t *Tensor::data() const {
  return ctensor(raw_tensor_)->data<int64_t>();
}
template <>
const int32_t *Tensor::data() const {
  return ctensor(raw_tensor_)->data<int32_t>();
}

// Tensor::mutable_data
template <>
int *Tensor::mutable_data(TargetType type) const {
  return tensor(raw_tensor_)->mutable_data<int>(type);
}
template <>
float *Tensor::mutable_data(TargetType type) const {
  return tensor(raw_tensor_)->mutable_data<float>(type);
}
template <>
int8_t *Tensor::mutable_data(TargetType type) const {
  return tensor(raw_tensor_)->mutable_data<int8_t>(type);
}
template <>
uint8_t *Tensor::mutable_data(TargetType type) const {
  return tensor(raw_tensor_)->mutable_data<uint8_t>(type);
}
template <>
int64_t *Tensor::mutable_data(TargetType type) const {
  return tensor(raw_tensor_)->mutable_data<int64_t>(type);
}

template <typename T, TargetType type>
void Tensor::CopyFromCpu(const T *src_data) {
  T *data = tensor(raw_tensor_)->mutable_data<T>(type);
  int64_t num = tensor(raw_tensor_)->numel();
  CHECK(num > 0) << "You should call Resize interface first";
  if (type == TargetType::kHost || type == TargetType::kARM) {
    lite::TargetWrapperHost::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoH);
  } else if (type == TargetType::kCUDA) {
#ifdef LITE_WITH_CUDA
    lite::TargetWrapperCuda::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoD);
#else
    LOG(FATAL) << "Please compile the lib with CUDA.";
#endif
  } else {
    LOG(FATAL) << "The CopyFromCpu interface just support kHost, kARM, kCUDA";
  }
}
template <typename T>
void Tensor::CopyToCpu(T *data) const {
  const T *src_data = tensor(raw_tensor_)->data<T>();
  int64_t num = tensor(raw_tensor_)->numel();
  CHECK(num > 0) << "You should call Resize interface first";
  auto type = tensor(raw_tensor_)->target();
  if (type == TargetType::kHost || type == TargetType::kARM) {
    lite::TargetWrapperHost::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::HtoH);
  } else if (type == TargetType::kCUDA) {
#ifdef LITE_WITH_CUDA
    lite::TargetWrapperCuda::MemcpySync(
        data, src_data, num * sizeof(T), lite::IoDirection::DtoH);
#else
    LOG(FATAL) << "Please compile the lib with CUDA.";
#endif
  } else {
    LOG(FATAL) << "The CopyToCpu interface just support kHost, kARM, kCUDA";
  }
}

template void Tensor::CopyFromCpu<int, TargetType::kHost>(const int *);
template void Tensor::CopyFromCpu<float, TargetType::kHost>(const float *);
template void Tensor::CopyFromCpu<int8_t, TargetType::kHost>(const int8_t *);
template void Tensor::CopyFromCpu<uint8_t, TargetType::kHost>(const uint8_t *);

template void Tensor::CopyFromCpu<int, TargetType::kARM>(const int *);
template void Tensor::CopyFromCpu<float, TargetType::kARM>(const float *);
template void Tensor::CopyFromCpu<int8_t, TargetType::kARM>(const int8_t *);
template void Tensor::CopyFromCpu<uint8_t, TargetType::kARM>(const uint8_t *);

template void Tensor::CopyFromCpu<int, TargetType::kCUDA>(const int *);
template void Tensor::CopyFromCpu<int64_t, TargetType::kCUDA>(const int64_t *);
template void Tensor::CopyFromCpu<float, TargetType::kCUDA>(const float *);
template void Tensor::CopyFromCpu<int8_t, TargetType::kCUDA>(const int8_t *);

template void Tensor::CopyToCpu(float *) const;
template void Tensor::CopyToCpu(int *) const;
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

#ifdef LITE_WITH_MLU
void CxxConfig::set_mlu_core_version(lite_api::MLUCoreVersion core_version) {
  mlu_core_version_ = core_version;
}
void CxxConfig::set_mlu_core_number(int core_number) {
  mlu_core_number_ = core_number;
}
void CxxConfig::set_mlu_input_layout(DataLayoutType layout) {
  mlu_input_layout_ = layout;
}
void CxxConfig::set_mlu_use_first_conv(bool use_first_conv) {
  mlu_use_first_conv_ = use_first_conv;
}
void CxxConfig::set_mlu_first_conv_mean(const std::vector<float> &mean) {
  mlu_first_conv_mean_ = mean;
}
void CxxConfig::set_mlu_first_conv_std(const std::vector<float> &std) {
  mlu_first_conv_std_ = std;
}
lite_api::MLUCoreVersion CxxConfig::mlu_core_version() const {
  return mlu_core_version_;
}
int CxxConfig::mlu_core_number() const { return mlu_core_number_; }
DataLayoutType CxxConfig::mlu_input_layout() const { return mlu_input_layout_; }
bool CxxConfig::mlu_use_first_conv() const { return mlu_use_first_conv_; }
const std::vector<float> &CxxConfig::mlu_first_conv_mean() const {
  return mlu_first_conv_mean_;
}
const std::vector<float> &CxxConfig::mlu_first_conv_std() const {
  return mlu_first_conv_std_;
}
#endif

void CxxConfig::set_xpu_workspace_l3_size_per_thread(int l3_size) {
#ifdef LITE_WITH_XPU
  lite::Context<TargetType::kXPU>::SetWorkspaceL3Size(l3_size);
#else
  LOG(WARNING) << "The invoking of the function "
                  "'set_xpu_workspace_l3_size_per_thread' is ignored, please "
                  "rebuild it with LITE_WITH_XPU=ON.";
#endif
}

void CxxConfig::set_xpu_dev_per_thread(int dev_no) {
#ifdef LITE_WITH_XPU
  lite::Context<TargetType::kXPU>::SetDev(dev_no);
#else
  LOG(WARNING) << "The invoking of the function 'set_xpu_dev_per_thread' is "
                  "ignored, please rebuild it with LITE_WITH_XPU=ON.";
#endif
}

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

}  // namespace lite_api
}  // namespace paddle
