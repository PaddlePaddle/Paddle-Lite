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

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

namespace adnn {

/**
 * Result code.
 */
typedef enum {
  SUCCESS = 0,
  OUT_OF_MEMORY,
  INVALID_PARAMETER,
  FEATURE_NOT_SUPPORTED
} Status;

/**
 * Result code.
 */
typedef enum {
  /**
   * Bind the worker threads to big clusters, use the max cores if the number of
   * the requested worker threads exceed the number of cores.
   */
  HIGH = 0,
  /**
   * Bind the worker threads to little clusters, use the max cores if the number
   * of the requested worker threads exceed the number of cores.
   */
  LOW,
  /**
   * Bind the worker threads to big and little clusters, use the max cores if
   * the number of the requested worker threads exceed the number of cores.
   */
  FULL,
  /**
   * No bind depends on the scheduling policy of the system.
   */
  NO_BIND,
  /**
   * Bind the worker threads to the big clusters in turn. Switch binding to the
   * next core after every 10 calls.
   */
  RAND_HIGH,
  /**
   * Bind the worker threads to the little clusters in turn. Switch binding to
   * the next core after every 10 calls.
   */
  RAND_LOW
} PowerMode;

/**
 * The key-value pair in device_setparam(...) and context_setparam(...) is used
 * for updating the parameters of the device and context.
 */
typedef enum {
  /**
   * Max number of threads of thread pool, dtype of value is int32_t.
   */
  DEVICE_MAX_THREAD_NUM = 0,
  /**
   * The power mode, dtype of value is int32_t.
   */
  DEVICE_POWER_MODE,
  /**
   * Does support Arm float16 instruction set on the current platform, dtype of
   * value is bool.
   */
  DEVICE_HAS_ARM_FP16,
  /**
   * Does Support Arm dotprod instruction set on the current platform, dtype of
   * value is bool.
   */
  DEVICE_HAS_ARM_DOTPROD,
  /**
   * Does support Arm sve2 instruction set on on the current platform, dtype of
   * value is bool.
   */
  DEVICE_HAS_ARM_SVE2,
  /**
   * The number of threads used in the current context, dtype of value is
   * int32_t.
   */
  CONTEXT_WORK_THREAD_NUM,
  /**
   * Enable/disable using Arm float16 instruction set, dtype of value is bool.
   */
  CONTEXT_ENABLE_ARM_FP16,
  /**
   * Enable/disable using Arm dotprod instruction set, dtype of value is bool.
   */
  CONTEXT_ENABLE_ARM_DOTPROD,
  /**
   * Enable/disable using Arm sve2 instruction set, dtype of value is bool.
   */
  CONTEXT_ENABLE_ARM_SVE2,
} ParamKey;

typedef union {
  bool b;
  int32_t i32;
  int64_t i64;
  float f32;
  double f64;
  void* ptr;
} ParamValue;

/**
 * Custom callback functions.
 */
typedef struct {
  void* (*device_open)();
  void (*device_close)(void* device);
  Status (*device_setparam)(void* device, ParamKey key, ParamValue value);
  Status (*device_getparam)(void* device, ParamKey key, ParamValue* value);
  void* (*context_create)(void* device);
  void (*context_destroy)(void* context);
  Status (*context_setparam)(void* device, ParamKey key, ParamValue value);
  Status (*context_getparam)(void* device, ParamKey key, ParamValue* value);
  void* (*memory_alloc)(void* context, size_t size);
  void (*memory_free)(void* context, void* ptr);
  void* (*memory_aligned_alloc)(void* context, size_t alignment, size_t size);
  void (*memory_aligned_free)(void* context, void* ptr);
} Callback;

/**
 * An opaque type for Device.
 */
typedef struct Device Device;

/**
 * An opaque type for Context.
 */
typedef struct Context Context;

}  // namespace adnn
