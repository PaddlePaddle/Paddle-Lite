// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "runtime/device.h"
#include <stdlib.h>
#include "arm_dnn_library/runtime/device.h"
#include "utilities/logging.h"

namespace armdnnlibrary {

void* DefaultDeviceOpen() { return nullptr; }

void DefaultDeviceClose(void* device) {}

Status DefaultDeviceSetParam(void* device, ParamKey key, ParamValue value) {
  return SUCCESS;
}

Status DefaultDeviceGetParam(void* device, ParamKey key, ParamValue* value) {
  return SUCCESS;
}

void* DefaultContextCreate(void* device) { return nullptr; }

void DefaultContextDestroy(void* context) {}

Status DefaultContextSetParam(void* context, ParamKey key, ParamValue value) {
  return SUCCESS;
}

Status DefaultContextGetParam(void* context, ParamKey key, ParamValue* value) {
  return SUCCESS;
}

void* DefaultMemoryAlloc(void* context, size_t size) { return malloc(size); }

void DefaultMemoryFree(void* context, void* ptr) {
  if (ptr) {
    free(ptr);
  }
}

void* DefaultMemoryAlignedAlloc(void* context, size_t size, size_t alignment) {
  size_t offset = sizeof(void*) + alignment - 1;
  char* p = static_cast<char*>(malloc(offset + size));
  // Byte alignment
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(alignment - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}

void DefaultMemoryAlignedFree(void* context, void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

}  // namespace armdnnlibrary

armdnnlibrary::Callback g_DefaultCallback = {
    .device_open = armdnnlibrary::DefaultDeviceOpen,
    .device_close = armdnnlibrary::DefaultDeviceClose,
    .device_setparam = armdnnlibrary::DefaultDeviceSetParam,
    .device_getparam = armdnnlibrary::DefaultDeviceGetParam,
    .context_create = armdnnlibrary::DefaultContextCreate,
    .context_destroy = armdnnlibrary::DefaultContextDestroy,
    .context_setparam = armdnnlibrary::DefaultContextSetParam,
    .context_getparam = armdnnlibrary::DefaultContextGetParam,
    .memory_alloc = armdnnlibrary::DefaultMemoryAlloc,
    .memory_free = armdnnlibrary::DefaultMemoryFree,
    .memory_aligned_alloc = armdnnlibrary::DefaultMemoryAlignedAlloc,
    .memory_aligned_free = armdnnlibrary::DefaultMemoryAlignedFree,
};

namespace armdnnlibrary {

Device::Device(const Callback* callback) : callback_(callback) {
  if (!callback_) {
    callback_ = &g_DefaultCallback;
  }
  ARM_DNN_LIBRARY_CHECK(callback_->device_open);
  device_ = callback_->device_open();
  // Initialize device and dump CPUInfo.
  CPUInfo::Singleton().DumpAllInfo();
}

Status Device::SetParam(ParamKey key, ParamValue value) {
  switch (key) {
    case DEVICE_ARCH:
    case DEVICE_SUPPORT_ARM_FP16:
    case DEVICE_SUPPORT_ARM_BF16:
    case DEVICE_SUPPORT_ARM_DOTPROD:
    case DEVICE_SUPPORT_ARM_SVE2:
    case DEVICE_SUPPORT_ARM_SVE2_I8MM:
    case DEVICE_SUPPORT_ARM_SVE2_F32MM:
      ARM_DNN_LIBRARY_LOG(ERROR)
          << "Unsupported key(" << static_cast<int32_t>(key)
          << ") for device_setparam() beacause the param is readonly!";
      return INVALID_PARAMETER;
    default:
      break;
  }
  params_[key] = value;
  ARM_DNN_LIBRARY_CHECK(callback_);
  ARM_DNN_LIBRARY_CHECK(callback_->device_setparam);
  return callback_->device_setparam(device_, key, value);
}

Status Device::GetParam(ParamKey key, ParamValue* value) {
  switch (key) {
    case DEVICE_ARCH:
      value->i32 = arch();
      return SUCCESS;
    case DEVICE_SUPPORT_ARM_FP16:
      value->b = support_arm_fp16();
      return SUCCESS;
    case DEVICE_SUPPORT_ARM_BF16:
      value->b = support_arm_bf16();
      return SUCCESS;
    case DEVICE_SUPPORT_ARM_DOTPROD:
      value->b = support_arm_dotprod();
      return SUCCESS;
    case DEVICE_SUPPORT_ARM_SVE2:
      value->b = support_arm_sve2();
      return SUCCESS;
    case DEVICE_SUPPORT_ARM_SVE2_I8MM:
      value->b = support_arm_sve2_i8mm();
      return SUCCESS;
    case DEVICE_SUPPORT_ARM_SVE2_F32MM:
      value->b = support_arm_sve2_f32mm();
      return SUCCESS;
    default:
      break;
  }
  if (!params_.count(key)) {
    memset(value, 0, sizeof(ParamValue));
    return INVALID_PARAMETER;
  }
  *value = params_[key];
  ARM_DNN_LIBRARY_CHECK(callback_);
  ARM_DNN_LIBRARY_CHECK(callback_->device_getparam);
  return callback_->device_getparam(device_, key, value);
}

Device::~Device() {
  ARM_DNN_LIBRARY_CHECK(callback_);
  ARM_DNN_LIBRARY_CHECK(callback_->device_close);
  callback_->device_close(device_);
  callback_ = nullptr;
  device_ = nullptr;
}

int32_t Device::max_thread_num() {
  if (!params_.count(DEVICE_MAX_THREAD_NUM)) {
    return 1;
  }
  return params_[DEVICE_MAX_THREAD_NUM].i32;
}

PowerMode Device::power_mode() {
  if (!params_.count(DEVICE_POWER_MODE)) {
    return PowerMode::NO_BIND;
  }
  return static_cast<PowerMode>(params_[DEVICE_POWER_MODE].i32);
}

CPUArch Device::arch() { return CPUInfo::Singleton().arch(); }

bool Device::support_arm_fp16() {
  return CPUInfo::Singleton().support_arm_fp16();
}

bool Device::support_arm_bf16() {
  return CPUInfo::Singleton().support_arm_bf16();
}

bool Device::support_arm_dotprod() {
  return CPUInfo::Singleton().support_arm_dotprod();
}

bool Device::support_arm_sve2() {
  return CPUInfo::Singleton().support_arm_sve2();
}

bool Device::support_arm_sve2_i8mm() {
  return CPUInfo::Singleton().support_arm_sve2_i8mm();
}

bool Device::support_arm_sve2_f32mm() {
  return CPUInfo::Singleton().support_arm_sve2_f32mm();
}

ARM_DNN_LIBRARY_DLL_EXPORT void* device_open(const Callback* callback) {
  return reinterpret_cast<void*>(new Device(callback));
}

ARM_DNN_LIBRARY_DLL_EXPORT void device_close(void* device) {
  if (device) {
    delete reinterpret_cast<Device*>(device);
  }
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status
device_setparam<ParamValue>(void* device, ParamKey key, ParamValue value) {
  if (!device) {
    return INVALID_PARAMETER;
  }
  return reinterpret_cast<Device*>(device)->SetParam(key, value);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status
device_getparam<ParamValue>(void* device, ParamKey key, ParamValue* value) {
  if (!device) {
    return INVALID_PARAMETER;
  }
  return reinterpret_cast<Device*>(device)->GetParam(key, value);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_setparam<bool>(void* device,
                                                        ParamKey key,
                                                        bool value) {
  ParamValue v;
  v.b = value;
  return device_setparam(device, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_setparam<int32_t>(void* device,
                                                           ParamKey key,
                                                           int32_t value) {
  ParamValue v;
  v.i32 = value;
  return device_setparam(device, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_setparam<int64_t>(void* device,
                                                           ParamKey key,
                                                           int64_t value) {
  ParamValue v;
  v.i64 = value;
  return device_setparam(device, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_setparam<float>(void* device,
                                                         ParamKey key,
                                                         float value) {
  ParamValue v;
  v.f32 = value;
  return device_setparam(device, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_setparam<double>(void* device,
                                                          ParamKey key,
                                                          double value) {
  ParamValue v;
  v.f64 = value;
  return device_setparam(device, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_setparam<void*>(void* device,
                                                         ParamKey key,
                                                         void* value) {
  ParamValue v;
  v.ptr = value;
  return device_setparam(device, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_getparam<bool>(void* device,
                                                        ParamKey key,
                                                        bool* value) {
  ParamValue v;
  auto status = device_getparam(device, key, &v);
  if (status != SUCCESS) return status;
  *value = v.b;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_getparam<int32_t>(void* device,
                                                           ParamKey key,
                                                           int32_t* value) {
  ParamValue v;
  auto status = device_getparam(device, key, &v);
  if (status != SUCCESS) return status;
  *value = v.i32;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_getparam<int64_t>(void* device,
                                                           ParamKey key,
                                                           int64_t* value) {
  ParamValue v;
  auto status = device_getparam(device, key, &v);
  if (status != SUCCESS) return status;
  *value = v.i64;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_getparam<float>(void* device,
                                                         ParamKey key,
                                                         float* value) {
  ParamValue v;
  auto status = device_getparam(device, key, &v);
  if (status != SUCCESS) return status;
  *value = v.f32;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_getparam<double>(void* device,
                                                          ParamKey key,
                                                          double* value) {
  ParamValue v;
  auto status = device_getparam(device, key, &v);
  if (status != SUCCESS) return status;
  *value = v.f64;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status device_getparam<void*>(void* device,
                                                         ParamKey key,
                                                         void** value) {
  ParamValue v;
  auto status = device_getparam(device, key, &v);
  if (status != SUCCESS) return status;
  *value = v.ptr;
  return SUCCESS;
}

}  // namespace armdnnlibrary
