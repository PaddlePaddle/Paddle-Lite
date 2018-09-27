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

#include <CL/cl.h>
#include <cstdlib>
#include <cstring>

namespace paddle_mobile {
namespace framework {

bool CLEngine::Init() {
  cl_int status;
  setPlatform();
  setClDeviceId();
  setClContext();
  setClCommandQueue();
  std::string filename = "./HelloWorld_Kernel.cl";
  loadKernelFromFile(filename.c_str());
  buildProgram();
  initialized_ = true;
}

CLEngine *CLEngine::Instance() {
  static CLEngine cl_engine_;
  return &cl_engine_;
}

std::unique_ptr<_cl_kernel, clKernel_deleter> CLEngine::GSetKernel(
    const std::string &kernel_name) {
  std::unique_ptr<_cl_kernel, clKernel_deleter> kernel(
      clCreateKernel(program_.get(), kernel_name.c_str(), NULL));
  return std::move(kernel);
}

bool CLEngine::SetClCommandQueue() {
  cl_int status;
  command_queue_.reset(
      clCreateCommandQueue(context_.get(), devices_[0], 0, &status));
  return true;
}

bool CLEngine::SetPlatform() {
  platform_ = NULL;      // the chosen platform
  cl_uint numPlatforms;  // the NO. of platforms
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);

  /**For clarity, choose the first available platform. */
  if (numPlatforms > 0) {
    cl_platform_id *platforms = reinterpret_cast<cl_platform_id *>(
        malloc(numPlatforms * sizeof(cl_platform_id)));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    platform_ = platforms[0];
    free(platforms);
    return true;
  } else {
    return false;
  }
}

bool CLEngine::SetClContext() {
  context_.reset(clCreateContext(NULL, 1, devices_, NULL, NULL, NULL));
  return true;
}

bool CLEngine::SetClDeviceId() {
  cl_uint numDevices = 0;
  devices_ = NULL;
  cl_int status =
      clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

  if (numDevices > 0) {
    std::cout << numDevices << std::endl;
    devices_ = reinterpret_cast<cl_platform_id *>(
        malloc(numDevices * sizeof(cl_device_id)));
    status = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, numDevices, devices_,
                            NULL);
    return true;
  }
  return false;
}

bool CLEngine::LoadKernelFromFile(const char *kernel_file) {
  size_t size;
  char *str;
  std::fstream f(kernel_file, (std::fstream::in | std::fstream::binary));

  if (!f.is_open()) {
    return false;
  }

  size_t fileSize;
  f.seekg(0, std::fstream::end);
  size = fileSize = (size_t)f.tellg();
  f.seekg(0, std::fstream::beg);
  str = new char[size + 1];
  if (!str) {
    f.close();
    return 0;
  }

  f.read(str, fileSize);
  f.close();
  str[size] = '\0';
  const char *source = str;
  size_t sourceSize[] = {strlen(source)};
  program_.reset(
      clCreateProgramWithSource(context_.get(), 1, &source, sourceSize, NULL));
  return true;
}

bool CLEngine::BuildProgram() {
  cl_int status;
  status = clBuildProgram(program_.get(), 0, 0, "-cl-fast-relaxed-math", 0, 0);
  return true;
}

}  // namespace framework
}  // namespace paddle_mobile
