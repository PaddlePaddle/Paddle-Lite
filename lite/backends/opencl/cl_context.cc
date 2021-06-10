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

#include "lite/backends/opencl/cl_context.h"
#include <memory>
#include <string>
#include <utility>
#include "lite/api/paddle_place.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {

cl::CommandQueue &CLContext::GetCommandQueue() {
  return CLRuntime::Global()->command_queue();
}

cl::Context &CLContext::GetContext() { return CLRuntime::Global()->context(); }

void CLContext::AddKernel(const std::string &kernel_name,
                          const std::string &file_name,
                          const std::string &options,
                          const std::string &time_stamp) {
  cl_int status{CL_SUCCESS};
#ifdef LITE_WITH_LOG
  VLOG(3) << " --- to get program " << file_name << " --- ";
#endif
  auto program = CLRuntime::Global()->GetProgram(file_name, options);
#ifdef LITE_WITH_LOG
  VLOG(3) << " --- end get program --- ";
  VLOG(3) << " --- to create kernel: " << kernel_name << " --- ";
#endif
  std::shared_ptr<cl::Kernel> kernel(
      new cl::Kernel(program, kernel_name.c_str(), &status));
  CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
  VLOG(3) << " --- end create kernel --- ";
#endif
  kernels_.emplace_back(std::move(kernel));
  STL::stringstream kernel_key;
  kernel_key << kernel_name << options << time_stamp;
  kernel_offset_[kernel_key.str()] = kernels_.size() - 1;
}

cl::Kernel &CLContext::GetKernel(const int index) {
#ifdef LITE_WITH_LOG
  VLOG(3) << " --- kernel count: " << kernels_.size() << " --- ";
#endif
  CHECK(static_cast<size_t>(index) < kernels_.size())
      << "The index must be less than the size of kernels.";
  CHECK(kernels_[index] != nullptr)
      << "The target kernel pointer cannot be null.";
  return *(kernels_[index]);
}

cl::Kernel &CLContext::GetKernel(const std::string &name) {
  auto it = kernel_offset_.find(name);
  CHECK(it != kernel_offset_.end()) << "Cannot find the kernel function: "
                                    << name;
  return GetKernel(it->second);
}

cl_int CLContext::RunKernel(const cl::Kernel &kernel,
                            const cl::NDRange &global,
                            const cl::NDRange &local,
                            cl::Event *event) {
  cl_int ret = GetCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global, local, nullptr, event);
  CL_CHECK_FATAL(ret);

  static int cnt = 0;
  const int flush_period = 10;
  if (cnt % flush_period == 0) {
    ret = GetCommandQueue().flush();
    CL_CHECK_FATAL(ret);
  }
  cnt++;

  return ret;
}

cl::NDRange CLContext::DefaultGlobalWorkSize(const CLImage &image) {
  // n c h w
  auto image_dim = image.tensor_dims();
  if (image_dim.size() == 4) {
    auto n = image_dim[0];
    auto h = image_dim[2];
    auto w = image_dim[3];
    auto image_width = image.ImageWidth();
    auto work_size_0 = image_width / w;
    auto work_size_1 = w;
    auto work_size_2 = n * h;
    return cl::NDRange{static_cast<size_t>(work_size_0),
                       static_cast<size_t>(work_size_1),
                       static_cast<size_t>(work_size_2)};
  } else if (image_dim.size() == 2) {
    return cl::NDRange{static_cast<size_t>(1),
                       static_cast<size_t>(image.ImageWidth()),
                       static_cast<size_t>(image.ImageHeight())};
  } else if (image_dim.size() == 1) {
    return cl::NDRange{static_cast<size_t>(1),
                       static_cast<size_t>(image.ImageWidth()),
                       static_cast<size_t>(1)};
  } else if (image_dim.size() == 3) {
    auto c = image_dim[0];
    auto h = image_dim[1];
    auto w = image_dim[2];
    return cl::NDRange{static_cast<size_t>((c + 3) / 4),
                       static_cast<size_t>(w),
                       static_cast<size_t>(h)};
  } else {
    LOG(FATAL) << "Not support this dimension, need to be implemented!";
    return cl::NDRange{};
  }
}

std::set<cl::NDRange> CLContext::GenerateLocalWorkSizes(cl::NDRange gws,
                                                        size_t max_ws) {
  size_t tune_type = CLRuntime::Global()->auto_tune();

  cl::NDRange tmp_lws =
      DefaultLocalWorkSize(gws, max_ws, /*divisor=*/2, /*tune_reverse=*/false);
  cl::NDRange last_lws = cl::NDRange{
      static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>(0)};
  std::set<cl::NDRange> lwss{tmp_lws};

  auto gen_lws = [&](const std::set<bool> &tune_reverses,
                     const std::set<size_t> &divisors) {
    for (bool tune_reverse : tune_reverses) {
      for (size_t divisor : divisors) {
        tmp_lws = DefaultLocalWorkSize(gws, max_ws, divisor, tune_reverse);
        lwss.emplace(tmp_lws);
      }
    }
  };

  std::set<bool> tune_reverses{true, false};
  std::set<size_t> divisors;
  if (tune_type == lite_api::CL_TUNE_NONE) {
    // do nothing
  } else if (tune_type == lite_api::CL_TUNE_RAPID) {
    divisors = {1, 2, 4, 8};
  } else if (tune_type == lite_api::CL_TUNE_NORMAL) {
    divisors = {1, 3, 5, 7, 9, 11, 13};
  } else if (tune_type == lite_api::CL_TUNE_EXHAUSTIVE) {
    divisors = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
#undef GEN_MORE_LWS
#ifdef GEN_MORE_LWS
    auto genByGlobal = [&](size_t global_i) -> std::set<size_t> {
      std::set<size_t> locals;
      int idx = 1;
      while (idx <= global_i) {
        if (global_i % idx == 0) {
          locals.insert(idx);
        }
        idx = idx << 2;
      }
      for (size_t i = 1; i <= 16; i++) {
        if (global_i % i == 0) {
          locals.insert(i);
        }
      }
      return locals;
    };
    std::set<size_t> locals_x = genByGlobal(static_cast<size_t>(gws[0]));
    std::set<size_t> locals_y = gws.dimensions() > 1
                                    ? genByGlobal(static_cast<size_t>(gws[1]))
                                    : std::set<size_t>{1};
    std::set<size_t> locals_z = gws.dimensions() > 2
                                    ? genByGlobal(static_cast<size_t>(gws[2]))
                                    : std::set<size_t>{1};
    std::map<std::string, size_t> device_info_map =
        CLRuntime::Global()->GetDeviceInfo();
    std::vector<size_t> max_work_item_sizes{
        device_info_map["CL_DEVICE_MAX_WORK_ITEM_SIZES_0"],
        device_info_map["CL_DEVICE_MAX_WORK_ITEM_SIZES_1"],
        device_info_map["CL_DEVICE_MAX_WORK_ITEM_SIZES_2"]};
    for (auto x : locals_x) {
      if (x <= max_work_item_sizes[0]) {
        for (auto y : locals_y) {
          if (y <= max_work_item_sizes[1]) {
            for (auto z : locals_z) {
              auto group_size = x * y * z;
              if (z <= max_work_item_sizes[2] && group_size <= max_ws &&
                  group_size >= /*min_workgrop_size=*/8) {
                lwss.insert(cl::NDRange{x, y, z});
              }
            }
          }
        }
      }
    }
#endif  // GEN_MORE_LWS
  } else {
    LOG(FATAL) << "Unsupported opencl tune type:" << tune_type;
  }
  gen_lws(tune_reverses, divisors);
  return lwss;
}

cl::NDRange CLContext::DefaultLocalWorkSize(
    const cl::NDRange &gws,
    register size_t max_ws,
    const int &divisor /*=2*/,
    const bool &reverse /*=false*/,
    const size_t &user_def_max_ws /*=0*/) {
  register size_t lx = reverse ? gws[2] : gws[0];
  register size_t ly = gws[1];
  register size_t lz = reverse ? gws[0] : gws[2];

  max_ws = (user_def_max_ws > 0 && user_def_max_ws <= max_ws) ? user_def_max_ws
                                                              : max_ws;
  max_ws = divisor > 1 ? max_ws / divisor : max_ws;

  if (max_ws > 0) {
    while (ly > max_ws) {
      // replace mod with bit operate
      ly = (ly & 0x01) ? 1 : ly >> 1;
    }
    while (ly * lz > max_ws) {
      lz = (lz & 0x01) ? 1 : lz >> 1;
    }
    while (ly * lz * lx > max_ws) {
      lx = (lx & 0x01) ? 1 : lx >> 1;
    }
  }

  return reverse ? cl::NDRange{lz, ly, lx} : cl::NDRange{lx, ly, lz};
}

bool CLContext::IsArmMali() {
  return CLRuntime::Global()->GetGpuType() == GpuType::ARM_MALI;
}

}  // namespace lite
}  // namespace paddle
