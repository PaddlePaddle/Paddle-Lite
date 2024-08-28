// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace lite {
/**
 * When LITE_WITH_OPENCL is enabled, Paddle-Lite will interact with the
 * OpenCL-related environment. Currently, when Paddle-Lite interacts with the
 * OpenCL runtime environment, it directly interacts through CLRuntime and
 * ClWrapper. CLContext actually serves as a part of the Kernel, carrying the
 * clKernel built for each runtime Kernel. In the process of using paddle_api or
 * program, the OpenCL environment has to be initialized. However, in practice,
 * sometimes the model is not an OpenCL model. Initializing the OpenCL
 * environment in such cases is a waste of memory, especially in environments
 * where there is a clear intention to avoid initializing the OpenCL
 * environment. Therefore, a method to isolate the OpenCL environment is
 * provided. When interacting with the framework, this proxy is uniformly
 * adopted.
 */
class ClGlobalDelegate {
 public:
  static ClGlobalDelegate& Global() {
    static ClGlobalDelegate x;
    return x;
  }
  /**
   * @brief set use opencl
   * @param use_opencl
   */
  void SetUseOpenCL(bool use_opencl) {
    use_opencl_ = use_opencl;
    VLOG(4) << "Set opencl softly , use_opencl: "
            << (use_opencl_ ? "enable" : "disable");
  }
  /**
   * @brief get use opencl
   * @return
   */
  bool UseOpenCL() const { return use_opencl_; }

  /**
   * @brief check opencl backend valid
   * @param check_fp16_valid
   * @return
   */
  bool IsOpenCLBackendValid(bool check_fp16_valid) {
    VLOG(3) << "Delegete opencl valid check, check_fp16_valid: "
            << check_fp16_valid << ", use_opencl_:" << use_opencl_;
    // use attempt to use opencl , enable it.
    SetUseOpenCL(true);
    bool opencl_valid = false;

#ifdef LITE_WITH_OPENCL
    bool opencl_lib_found = paddle::lite::CLWrapper::Global()->OpenclLibFound();
    LOG(INFO) << "Found opencl library: " << opencl_lib_found;
    if (!opencl_lib_found) return false;

    bool dlsym_success = paddle::lite::CLWrapper::Global()->DlsymSuccess();
    LOG(INFO) << "Dlsym Success: " << dlsym_success;
    if (!dlsym_success) return false;
    opencl_valid = paddle::lite::CLRuntime::Global()->OpenCLAvaliableForDevice(
        check_fp16_valid);
    LOG(INFO) << "Opencl Valid: " << opencl_valid;
#endif
    return opencl_valid;
  }

  /**
   * @brief get opencl device type
   * @return
   */
  int GetOpenCLDeviceType() {
    if (this->IsOpenCLBackendValid(false)) {
      return paddle::lite::CLRuntime::Global()->GetGpuType();
    }
    return -1;
  }

 private:
  ClGlobalDelegate() = default;
  // if user do not set this flag, as old ways.
  bool use_opencl_{true};
};
}  // namespace lite
}  // namespace paddle
