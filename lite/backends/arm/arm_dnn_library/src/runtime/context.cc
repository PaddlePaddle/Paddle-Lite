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

#include "arm_dnn_library/runtime/context.h"
#include "runtime/context.h"
#include "utilities/logging.h"

namespace armdnnlibrary {

ARM_DNN_LIBRARY_THREAD_LOCAL void* Context::workspace_data_ = nullptr;
ARM_DNN_LIBRARY_THREAD_LOCAL size_t Context::workspace_size_ = 0;

Context::Context(Device* device) : device_(device) {
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->context_create);
  context_ = device_->callback()->context_create(device_->device());
  // Initialize context params from device params.
  ParamValue value;
  value.b = device_->support_arm_fp16();
  SetParam(CONTEXT_ENABLE_ARM_FP16, value);
  value.b = device_->support_arm_bf16();
  SetParam(CONTEXT_ENABLE_ARM_BF16, value);
  value.b = device_->support_arm_dotprod();
  SetParam(CONTEXT_ENABLE_ARM_DOTPROD, value);
  value.b = device_->support_arm_sve2();
  SetParam(CONTEXT_ENABLE_ARM_SVE2, value);
  value.b = device_->support_arm_sve2_i8mm();
  SetParam(CONTEXT_ENABLE_ARM_SVE2_I8MM, value);
  value.b = device_->support_arm_sve2_f32mm();
  SetParam(CONTEXT_ENABLE_ARM_SVE2_F32MM, value);
}

Status Context::SetParam(ParamKey key, ParamValue value, bool force) {
  switch (key) {
    case CONTEXT_WORK_PACE_DATA:
      ARM_DNN_LIBRARY_LOG(ERROR)
          << "Unsupported key(" << static_cast<int32_t>(key)
          << ") for context_setparam() beacause the param is readonly!";
      return INVALID_PARAMETER;
    case CONTEXT_WORK_PACE_SIZE:
      ARM_DNN_LIBRARY_CHECK_GT(value.szt, 0);
      SetWorkspaceSize(value.szt);
      return SUCCESS;
    default:
      break;
  }
  params_[key] = value;
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->context_setparam);
  return device_->callback()->context_setparam(context_, key, value);
}

Status Context::GetParam(ParamKey key, ParamValue* value) {
  switch (key) {
    case CONTEXT_WORK_PACE_DATA:
      value->ptr = workspace_data();
      return SUCCESS;
    case CONTEXT_WORK_PACE_SIZE:
      value->szt = workspace_size();
      return SUCCESS;
    default:
      break;
  }
  if (!params_.count(key)) {
    memset(value, 0, sizeof(ParamValue));
    return INVALID_PARAMETER;
  }
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->context_getparam);
  return device_->callback()->context_getparam(context_, key, value);
}

void* Context::MemoryAlloc(size_t size) {
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->memory_alloc);
  return device_->callback()->memory_alloc(context_, size);
}

void Context::MemoryFree(void* ptr) {
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->memory_free);
  return device_->callback()->memory_free(context_, ptr);
}

void* Context::MemoryAlignedAlloc(size_t size, size_t alignment) {
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->memory_aligned_alloc);
  return device_->callback()->memory_aligned_alloc(context_, size, alignment);
}

void Context::MemoryAlignedFree(void* ptr) {
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->memory_aligned_free);
  return device_->callback()->memory_aligned_free(context_, ptr);
}

Context::~Context() {
  ARM_DNN_LIBRARY_CHECK(device_);
  ARM_DNN_LIBRARY_CHECK(device_->callback()->context_destroy);
  device_->callback()->context_destroy(context_);
  device_ = nullptr;
  context_ = nullptr;
}

int32_t Context::work_thread_num() {
  if (!params_.count(CONTEXT_WORK_THREAD_NUM)) {
    return 1;
  }
  return params_[CONTEXT_WORK_THREAD_NUM].i32;
}

bool Context::enable_arm_fp16() {
  if (!params_.count(CONTEXT_ENABLE_ARM_FP16)) {
    return false;
  }
  return params_[CONTEXT_ENABLE_ARM_FP16].b;
}

bool Context::enable_arm_bf16() {
  if (!params_.count(CONTEXT_ENABLE_ARM_BF16)) {
    return false;
  }
  return params_[CONTEXT_ENABLE_ARM_BF16].b;
}

bool Context::enable_arm_dotprod() {
  if (!params_.count(CONTEXT_ENABLE_ARM_DOTPROD)) {
    return false;
  }
  return params_[CONTEXT_ENABLE_ARM_DOTPROD].b;
}

bool Context::enable_arm_sve2() {
  if (!params_.count(CONTEXT_ENABLE_ARM_SVE2)) {
    return false;
  }
  return params_[CONTEXT_ENABLE_ARM_SVE2].b;
}

bool Context::enable_arm_sve2_i8mm() {
  if (!params_.count(CONTEXT_ENABLE_ARM_SVE2_I8MM)) {
    return false;
  }
  return params_[CONTEXT_ENABLE_ARM_SVE2_I8MM].b;
}

bool Context::enable_arm_sve2_f32mm() {
  if (!params_.count(CONTEXT_ENABLE_ARM_SVE2_F32MM)) {
    return false;
  }
  return params_[CONTEXT_ENABLE_ARM_SVE2_F32MM].b;
}

void* Context::workspace_data() { return workspace_data_; }

size_t Context::workspace_size() { return workspace_size_; }

void* Context::SetWorkspaceSize(size_t size) {
  if (size > workspace_size_) {
    if (workspace_data_) {
      MemoryAlignedFree(workspace_data_);
    }
    workspace_data_ = MemoryAlignedAlloc(size);
    workspace_size_ = size;
  }
  return workspace_data_;
}

ARM_DNN_LIBRARY_DLL_EXPORT void* context_create(void* device) {
  if (!device) {
    return nullptr;
  }
  return reinterpret_cast<void*>(
      new Context(reinterpret_cast<Device*>(device)));
}

ARM_DNN_LIBRARY_DLL_EXPORT void context_destroy(void* context) {
  if (context) {
    delete reinterpret_cast<Context*>(context);
  }
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status
context_setparam<ParamValue>(void* context, ParamKey key, ParamValue value) {
  if (!context) {
    return INVALID_PARAMETER;
  }
  return reinterpret_cast<Context*>(context)->SetParam(key, value);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status
context_getparam<ParamValue>(void* context, ParamKey key, ParamValue* value) {
  if (!context) {
    return INVALID_PARAMETER;
  }
  return reinterpret_cast<Context*>(context)->GetParam(key, value);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<bool>(void* context,
                                                         ParamKey key,
                                                         bool value) {
  ParamValue v;
  v.b = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<int32_t>(void* context,
                                                            ParamKey key,
                                                            int32_t value) {
  ParamValue v;
  v.i32 = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<int64_t>(void* context,
                                                            ParamKey key,
                                                            int64_t value) {
  ParamValue v;
  v.i64 = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<size_t>(void* context,
                                                           ParamKey key,
                                                           size_t value) {
  ParamValue v;
  v.szt = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<float>(void* context,
                                                          ParamKey key,
                                                          float value) {
  ParamValue v;
  v.f32 = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<double>(void* context,
                                                           ParamKey key,
                                                           double value) {
  ParamValue v;
  v.f64 = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_setparam<void*>(void* context,
                                                          ParamKey key,
                                                          void* value) {
  ParamValue v;
  v.ptr = value;
  return context_setparam(context, key, v);
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_getparam<bool>(void* context,
                                                         ParamKey key,
                                                         bool* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.b;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_getparam<int32_t>(void* context,
                                                            ParamKey key,
                                                            int32_t* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.i32;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_getparam<int64_t>(void* context,
                                                            ParamKey key,
                                                            int64_t* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.i64;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_getparam<size_t>(void* context,
                                                           ParamKey key,
                                                           size_t* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.szt;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_getparam<float>(void* context,
                                                          ParamKey key,
                                                          float* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.f32;
  return SUCCESS;
}

template <>
ARM_DNN_LIBRARY_DLL_EXPORT Status context_getparam<double>(void* context,
                                                           ParamKey key,
                                                           double* value) {
  ParamValue v;
  auto status = context_getparam(context, key, &v);
  if (status != SUCCESS) return status;
  *value = v.f64;
  return SUCCESS;
}

}  // namespace armdnnlibrary
