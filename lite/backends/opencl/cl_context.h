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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/backends/opencl/cl_image.h"
#include "lite/backends/opencl/cl_include.h"

namespace paddle {
namespace lite {

class CLContext {
 public:
  ~CLContext() {
    for (size_t kidx = 0; kidx < kernels_.size(); ++kidx) {
      clReleaseKernel(kernels_[kidx]->get());
      kernels_[kidx].reset();
    }
    kernels_.clear();
    kernel_offset_.clear();
    for (auto &p : programs_) {
      clReleaseProgram(p.second->get());
    }
    programs_.clear();
    LOG(INFO) << "release cl::Program, cl::Kernel finished.";
  }

  cl::CommandQueue &GetCommandQueue();

  cl::Context &GetContext();

  cl::Program &GetProgram(const std::string &file_name,
                          const std::string &options);

  void AddKernel(const std::string &kernel_name,
                 const std::string &file_name,
                 const std::string &options = "",
                 const std::string &time_stamp = "");

  cl::Kernel &GetKernel(const int index);

  cl::Kernel &GetKernel(const std::string &name);

  cl::NDRange DefaultWorkSize(const CLImage &image);

  cl::NDRange LocalWorkSize(cl::NDRange global_work_size, size_t max_work_size);

  cl::NDRange LocalWorkSizeTurn(cl::NDRange global_work_size,
                                size_t max_work_size,
                                int divitor = 2);
  //  cl::NDRange LocalWorkSizeConv1x1(cl::NDRange global_work_size,
  //                                   size_t max_work_size);
 private:
  std::unordered_map<std::string, std::unique_ptr<cl::Program>> programs_;
  std::vector<std::unique_ptr<cl::Kernel>> kernels_;
  std::map<std::string, int> kernel_offset_;
};

}  // namespace lite
}  // namespace paddle
