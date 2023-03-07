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

#include "adnn/runtime/context.h"
#include "runtime/context.h"
#include "utilities/dll_export.h"
#include "utilities/logging.h"

namespace adnn {

Context::Context(Device* device) : device_(device) {
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->context_create);
  context_ = device_->GetCallback()->context_create(device_->GetDevice());
}

Status Context::SetParam(ParamKey key, ParamValue value) {
  params_[key] = value;
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->context_setparam);
  return device_->GetCallback()->context_setparam(context_, key, value);
}

Status Context::GetParam(ParamKey key, ParamValue* value) {
  if (!params_.count(key)) {
    return INVALID_PARAMETER;
  }
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->context_getparam);
  return device_->GetCallback()->context_getparam(context_, key, value);
}

int32_t Context::GetWorkThreadNum() {
  if (!params_.count(CONTEXT_WORK_THREAD_NUM)) {
    return 1;
  }
  return params_[CONTEXT_WORK_THREAD_NUM].i32;
}

void* Context::MemoryAlloc(size_t size) {
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->memory_alloc);
  return device_->GetCallback()->memory_alloc(context_, size);
}

void Context::MemoryFree(void* ptr) {
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->memory_free);
  return device_->GetCallback()->memory_free(context_, ptr);
}

void* Context::MemoryAlignedAlloc(size_t alignment, size_t size) {
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->memory_aligned_alloc);
  return device_->GetCallback()->memory_aligned_alloc(
      context_, alignment, size);
}

void Context::MemoryAlignedFree(void* ptr) {
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->memory_aligned_free);
  return device_->GetCallback()->memory_aligned_free(context_, ptr);
}

Context::~Context() {
  ADNN_CHECK(device_);
  ADNN_CHECK(device_->GetCallback()->context_destroy);
  device_->GetCallback()->context_destroy(context_);
  device_ = nullptr;
  context_ = nullptr;
}

ADNN_DLL_EXPORT void* context_create(void* device) {
  if (!device) {
    return nullptr;
  }
  return reinterpret_cast<void*>(
      new Context(reinterpret_cast<Device*>(device)));
}

ADNN_DLL_EXPORT void context_destroy(void* context) {
  if (context) {
    delete reinterpret_cast<Context*>(context);
  }
}

template <>
ADNN_DLL_EXPORT Status context_setparam<ParamValue>(void* context,
                                                    ParamKey key,
                                                    ParamValue value) {
  if (!context) {
    return INVALID_PARAMETER;
  }
  return reinterpret_cast<Context*>(context)->SetParam(key, value);
}

template <>
ADNN_DLL_EXPORT Status context_getparam<ParamValue>(void* context,
                                                    ParamKey key,
                                                    ParamValue* value) {
  if (!context) {
    return INVALID_PARAMETER;
  }
  return reinterpret_cast<Context*>(context)->GetParam(key, value);
}

template <>
ADNN_DLL_EXPORT Status context_setparam<bool>(void* context,
                                              ParamKey key,
                                              bool value) {
  ParamValue v;
  v.b = value;
  return context_setparam(context, key, v);
}

template <>
ADNN_DLL_EXPORT Status context_setparam<int32_t>(void* context,
                                                 ParamKey key,
                                                 int32_t value) {
  ParamValue v;
  v.i32 = value;
  return context_setparam(context, key, v);
}

template <>
ADNN_DLL_EXPORT Status context_setparam<int64_t>(void* context,
                                                 ParamKey key,
                                                 int64_t value) {
  ParamValue v;
  v.i64 = value;
  return context_setparam(context, key, v);
}

template <>
ADNN_DLL_EXPORT Status context_setparam<float>(void* context,
                                               ParamKey key,
                                               float value) {
  ParamValue v;
  v.f32 = value;
  return context_setparam(context, key, v);
}

template <>
ADNN_DLL_EXPORT Status context_setparam<double>(void* context,
                                                ParamKey key,
                                                double value) {
  ParamValue v;
  v.f64 = value;
  return context_setparam(context, key, v);
}

template <>
ADNN_DLL_EXPORT Status context_setparam<void*>(void* context,
                                               ParamKey key,
                                               void* value) {
  ParamValue v;
  v.ptr = value;
  return context_setparam(context, key, v);
}

template <>
ADNN_DLL_EXPORT Status context_getparam<bool>(void* context,
                                              ParamKey key,
                                              bool* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.b;
  return SUCCESS;
}

template <>
ADNN_DLL_EXPORT Status context_getparam<int32_t>(void* context,
                                                 ParamKey key,
                                                 int32_t* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.i32;
  return SUCCESS;
}

template <>
ADNN_DLL_EXPORT Status context_getparam<int64_t>(void* context,
                                                 ParamKey key,
                                                 int64_t* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.i64;
  return SUCCESS;
}

template <>
ADNN_DLL_EXPORT Status context_getparam<float>(void* context,
                                               ParamKey key,
                                               float* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.f32;
  return SUCCESS;
}

template <>
ADNN_DLL_EXPORT Status context_getparam<double>(void* context,
                                                ParamKey key,
                                                double* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.f64;
  return SUCCESS;
}

template <>
ADNN_DLL_EXPORT Status context_getparam<void*>(void* context,
                                               ParamKey key,
                                               void** value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.ptr;
  return SUCCESS;
}

}  // namespace adnn
