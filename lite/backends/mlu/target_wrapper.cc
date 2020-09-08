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

#include "lite/backends/mlu/target_wrapper.h"

#include <memory>
#include <utility>

#include "lite/backends/mlu/mlu_utils.h"
#include "lite/utils/macros.h"

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

LITE_THREAD_LOCAL cnmlCoreVersion_t TargetWrapperMlu::mlu_core_version_{
    CNML_MLU270};
LITE_THREAD_LOCAL int TargetWrapperMlu::mlu_core_number_{1};
LITE_THREAD_LOCAL bool TargetWrapperMlu::use_first_conv_{false};
LITE_THREAD_LOCAL std::vector<float> TargetWrapperMlu::mean_vec_;
LITE_THREAD_LOCAL std::vector<float> TargetWrapperMlu::std_vec_;
LITE_THREAD_LOCAL DataLayoutType TargetWrapperMlu::input_layout_{
    DATALAYOUT(kNCHW)};

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
void TargetWrapperMlu::SetMLURunMode(
    lite_api::MLUCoreVersion core_version,
    int core_number,
    DataLayoutType input_layout,
    std::pair<std::vector<float>, std::vector<float>> firstconv_param) {
  switch (core_version) {
    case (lite_api::MLUCoreVersion::MLU_220):
      mlu_core_version_ = CNML_MLU220;
      break;
    case (lite_api::MLUCoreVersion::MLU_270):
      mlu_core_version_ = CNML_MLU270;
      break;
    default:
      mlu_core_version_ = CNML_MLU270;
      break;
  }
  mlu_core_number_ = core_number;
  mean_vec_ = firstconv_param.first;
  std_vec_ = firstconv_param.second;
  use_first_conv_ = !(mean_vec_.empty() || std_vec_.empty());
  input_layout_ = input_layout;
}

cnmlCoreVersion_t TargetWrapperMlu::MLUCoreVersion() {
  return mlu_core_version_;
}

int TargetWrapperMlu::MLUCoreNumber() { return mlu_core_number_; }

bool TargetWrapperMlu::UseFirstConv() { return use_first_conv_; }

const std::vector<float>& TargetWrapperMlu::MeanVec() { return mean_vec_; }

const std::vector<float>& TargetWrapperMlu::StdVec() { return std_vec_; }

DataLayoutType TargetWrapperMlu::InputLayout() { return input_layout_; }

}  // namespace lite
}  // namespace paddle
