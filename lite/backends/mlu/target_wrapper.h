// Copyright (c) 2019 Cambricon Authors. All Rights Reserved.

#pragma once
#include "lite/backends/mlu/mlu_utils.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetWrapperMlu = TargetWrapper<TARGET(kMLU)>;

template <>
class TargetWrapper<TARGET(kMLU)> {
 public:
  using queue_t = cnrtQueue_t;

  static size_t num_devices();
  static size_t maxinum_queue() { return 0; }  // TODO(zhangshijin): fix out it.

  static size_t GetCurDevice() { return 0; }

  static void CreateQueue(queue_t* queue) {}
  static void DestroyQueue(const queue_t& queue) {}

  static void QueueSync(const queue_t& queue) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);
  // static void MemcpyAsync(void* dst,
  //                         const void* src,
  //                         size_t size,
  //                         IoDirection dir,
  //                         const queue_t& queue);
};

}  // namespace lite
}  // namespace paddle
