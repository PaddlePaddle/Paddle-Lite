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
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/utils/log/cp_logging.h"

#ifdef LITE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // LITE_WITH_CUDA

namespace paddle {
namespace lite {

using lite_api::TargetType;
using lite_api::PrecisionType;
using lite_api::DataLayoutType;
using lite_api::PrecisionTypeLength;
using lite_api::TargetToStr;
using lite_api::Place;
using lite_api::PrecisionToStr;
using lite_api::DataLayoutToStr;
using lite_api::TargetRepr;
using lite_api::PrecisionRepr;
using lite_api::DataLayoutRepr;

namespace host {
const int MALLOC_ALIGN = 64;

// Allocate the requested memory and return a pointer to it.
// Byte alignment and memory checking are performed.
inline void* malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(std::malloc(offset + size));
  // Memory checking
  CHECK(p) << "Error occurred in malloc period: available space is not enough "
              "for mallocing "
           << size << " bytes.";
  // Byte alignment
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}

// Deallocate the memory.
inline void free(void* ptr) {
  if (ptr) {
    std::free(static_cast<void**>(ptr)[-1]);
  }
}

// Copy size characters from memory area src to memory area dst.
// Memory checking is performed.
inline void memcpy(void* dst, const void* src, size_t size) {
  if (size > 0) {
    CHECK(dst) << "Error: the destination of memcpy can not be nullptr.";
    CHECK(src) << "Error: the source of memcpy can not be nullptr.";
    std::memcpy(dst, src, size);
  }
}

// Reinterprets the objects pointed to by lhs and rhs as arrays of
// unsigned char and compares the first count characters of these arrays.
inline int memcmp(const void* lhs, const void* rhs, std::size_t count) {
  if (count > 0) {
    CHECK(lhs) << "Error: the destination of memcpy can not be nullptr.";
    CHECK(rhs) << "Error: the source of memcpy can not be nullptr.";
    return std::memcmp(lhs, rhs, count);
  } else {
    return 0;
  }
}

}  // namespace host

// Memory copy directions.
enum class IoDirection {
  HtoH = 0,  // Host to host
  HtoD,      // Host to device
  DtoH,      // Device to host
  DtoD,      // Device to device
};

// This interface should be specified by each kind of target.
template <TargetType Target, typename StreamTy = int, typename EventTy = int>
class TargetWrapper {
 public:
  using stream_t = StreamTy;
  using event_t = EventTy;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size) {
    LOG(FATAL) << "Unimplemented malloc for " << TargetToStr(Target);
    return nullptr;
  }
  static void Free(void* ptr) { LOG(FATAL) << "Unimplemented"; }

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir) {
    LOG(FATAL) << "Unimplemented";
  }
  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream) {
    MemcpySync(dst, src, size, dir);
  }
};

// This interface should be specified by each kind of target.
using TargetWrapperHost = TargetWrapper<TARGET(kHost)>;
using TargetWrapperX86 = TargetWrapperHost;
template <>
class TargetWrapper<TARGET(kHost)> {
 public:
  using stream_t = int;
  using event_t = int;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir = lite::IoDirection::HtoH);
  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream) {
    MemcpySync(dst, src, size, dir);
  }
};

#ifdef LITE_WITH_FPGA
template <>
class TargetWrapper<TARGET(kFPGA)> {
 public:
  using stream_t = int;
  using event_t = int;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);
  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream) {
    MemcpySync(dst, src, size, dir);
  }
};
#endif

}  // namespace lite
}  // namespace paddle
