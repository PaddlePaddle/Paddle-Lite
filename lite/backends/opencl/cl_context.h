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
#include <set>
#include <string>
#include <vector>
#include "lite/backends/opencl/cl_image.h"
#include "lite/backends/opencl/cl_include.h"

namespace paddle {
namespace lite {

class CLContext {
 public:
  ~CLContext() {
    GetCommandQueue().finish();
    for (size_t kidx = 0; kidx < kernels_.size(); ++kidx) {
      // Note(ysh329): Don't need `clReleaseKernel`
      kernels_[kidx].reset();
    }
    kernels_.clear();
    kernel_offset_.clear();
    for (auto &p : CLRuntime::Global()->program_map()) {
      // Note(ysh329): Dont't need `clReleaseProgram`
      p.second.reset();
    }
    CLRuntime::Global()->program_map().clear();
    LOG(INFO) << "release cl::Program, cl::Kernel finished.";
  }

  cl::CommandQueue &GetCommandQueue();

  cl::Context &GetContext();

  void AddKernel(const std::string &kernel_name,
                 const std::string &file_name,
                 const std::string &options = "",
                 const std::string &time_stamp = "");

  cl::Kernel &GetKernel(const int index);

  cl::Kernel &GetKernel(const std::string &name);

  cl_int RunKernel(const cl::Kernel &kernel,
                   const cl::NDRange &global,
                   const cl::NDRange &local,
                   cl::Event *event = nullptr);

  cl::NDRange DefaultGlobalWorkSize(const CLImage &image);

  cl::NDRange DefaultLocalWorkSize(
      const cl::NDRange &global_work_size,
      register size_t max_work_size,
      const int &divitor = 2,
      const bool &tune_reverse = false,
      const size_t &user_defined_max_work_size = 0);

  std::set<cl::NDRange> GenerateLocalWorkSizes(cl::NDRange global_work_size,
                                               size_t max_work_size);
  bool IsArmMali();

 private:
  std::vector<std::shared_ptr<cl::Kernel>> kernels_;
  std::map<std::string, int> kernel_offset_;
  std::map<std::string, cl::NDRange> tuned_lwss_map_;
};

}  // namespace lite
}  // namespace paddle
