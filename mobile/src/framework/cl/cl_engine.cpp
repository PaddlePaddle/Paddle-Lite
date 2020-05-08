/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "framework/cl/cl_engine.h"
#include "CL/cl.h"
#include "framework/cl/cl_tool.h"

#include <cstdlib>
#include <cstring>

namespace paddle_mobile {
namespace framework {

bool CLEngine::Init() {
  LOG(paddle_mobile::kNO_LOG) << "CLEngine::Init()";
  if (initialized_) {
    return true;
  }
  LOG(paddle_mobile::kNO_LOG) << "CLEngine::Init() ...";
  cl_int status;
  bool is_setplatform_success = SetPlatform();
  bool is_setcldeviceid_success = SetClDeviceId();
  is_init_success_ = is_setplatform_success && is_setcldeviceid_success;
  initialized_ = true;
  return initialized_;
  //  setClCommandQueue();
  //  std::string filename = "./HelloWorld_Kernel.cl";
  //  loadKernelFromFile(filename.c_str());
  //  buildProgram();
}

CLEngine *CLEngine::Instance() {
  static CLEngine cl_engine_;
  cl_engine_.Init();
  return &cl_engine_;
}

bool CLEngine::isInitSuccess() { return is_init_success_; }
bool CLEngine::SetPlatform() {
  platform_ = NULL;      // the chosen platform
  cl_uint numPlatforms;  // the NO. of platforms
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS) {
    return false;
  }
  /**For clarity, choose the first available platform. */
  LOG(paddle_mobile::kNO_LOG) << "numPlatforms: " << numPlatforms;
  if (numPlatforms > 0) {
    cl_platform_id *platforms = reinterpret_cast<cl_platform_id *>(
        malloc(numPlatforms * sizeof(cl_platform_id)));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    platform_ = platforms[0];
    free(platforms);
    LOG(paddle_mobile::kNO_LOG) << "platform: " << platform_;
    return status == CL_SUCCESS;
  }

  return false;
}

bool CLEngine::SetClDeviceId() {
  cl_uint numDevices = 0;
  LOG(paddle_mobile::kNO_LOG) << "platform: " << platform_;
  cl_int status =
      clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (status != CL_SUCCESS) {
    return false;
  }
  LOG(paddle_mobile::kNO_LOG) << "numDevices: " << numDevices;

  if (numDevices > 0) {
    status = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, numDevices, devices_,
                            NULL);
    LOG(paddle_mobile::kNO_LOG) << "devices_[0]" << devices_[0];
    return status == CL_SUCCESS;
  }
  return false;
}
}  // namespace framework
}  // namespace paddle_mobile
