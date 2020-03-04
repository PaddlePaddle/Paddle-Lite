// Copyright (c) 2019 Cambricon Authors. All Rights Reserved.

#include "lite/backends/mlu/target_wrapper.h"
#include <memory>
#include "lite/backends/mlu/mlu_utils.h"

namespace paddle {
namespace lite {
namespace mlu {

void cnrtMemcpyHtoD(void* dst, const void* src, size_t size) {
  CNRT_CALL(cnrtMemcpy(
      dst, const_cast<void*>(src), size, CNRT_MEM_TRANS_DIR_HOST2DEV))
      << " cnrt memcpy htod failed";
}

void cnrtMemcpyDtoH(void* dst, const void* src, size_t size) {
  CNRT_CALL(cnrtMemcpy(
      dst, const_cast<void*>(src), size, CNRT_MEM_TRANS_DIR_DEV2HOST))
      << " cnrt memcpy dtoh failed";
}

}  // namespace mlu

size_t TargetWrapperMlu::num_devices() {
  uint32_t dev_count = 0;
  CNRT_CALL(cnrtGetDeviceCount(&dev_count)) << " cnrt get device count failed";
  LOG(INFO) << "Current MLU device count: " << dev_count;
  return dev_count;
}

void* TargetWrapperMlu::Malloc(size_t size) {
  void* ptr{};
  CNRT_CALL(cnrtMalloc(&ptr, size)) << " cnrt malloc failed";
  // LOG(INFO) << "Malloc mlu ptr: " << ptr << " with size: " << size;
  return ptr;
}

void TargetWrapperMlu::Free(void* ptr) {
  CNRT_CALL(cnrtFree(ptr)) << " cnrt free failed";
}

void TargetWrapperMlu::MemcpySync(void* dst,
                                  const void* src,
                                  size_t size,
                                  IoDirection dir) {
  // LOG(INFO) << "dst: " << dst << " src: " << src << " size: " << size
  //<< " dir: " << (int)dir;
  switch (dir) {
    case IoDirection::DtoD: {
      std::unique_ptr<char[]> cpu_tmp_ptr(new char[size]);
      mlu::cnrtMemcpyDtoH(cpu_tmp_ptr.get(), src, size);
      mlu::cnrtMemcpyHtoD(dst, cpu_tmp_ptr.get(), size);
      break;
    }
    case IoDirection::HtoD:
      mlu::cnrtMemcpyHtoD(dst, src, size);
      break;
    case IoDirection::DtoH:
      mlu::cnrtMemcpyDtoH(dst, src, size);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection" << static_cast<int>(dir);
  }
}

// void TargetWrapperMlu::MemcpyAsync(void* dst,
//                                    const void* src,
//                                    size_t size,
//                                    IoDirection dir,
//                                    const stream_t& stream) {
//   LOG(WARNING) << "Mlu unsupported MemcpyAsync now, use MemcpySync.";
//   MemcpySync(dst, src, size, dir);
// }

}  // namespace lite
}  // namespace paddle
