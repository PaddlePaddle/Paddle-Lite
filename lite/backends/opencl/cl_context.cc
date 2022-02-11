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
#include <math.h>
#include <memory>
#include <string>
#include <utility>
#include "lite/api/paddle_place.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/utils/log/cp_logging.h"
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

std::set<cl::NDRange, CLContext::CompareByRange>
CLContext::GenerateLocalWorkSizes(cl::NDRange gws, size_t max_ws) {
  size_t tune_type = CLRuntime::Global()->auto_tune();
  auto first_lws = DefaultLocalWorkSize(gws, max_ws, tune_type, 3, false);
  std::set<cl::NDRange, CompareByRange> lwss;
  for (auto one_lws : first_lws) {
    lwss.insert(one_lws);
  }
  auto gen_lws = [&](const std::set<bool> &tune_reverses,
                     const std::set<size_t> &divisors) {
    for (bool tune_reverse : tune_reverses) {
      for (size_t divisor : divisors) {
        std::set<cl::NDRange, CompareByRange> tmp_lws =
            DefaultLocalWorkSize(gws, max_ws, tune_type, divisor, tune_reverse);
        for (cl::NDRange one_lws : tmp_lws) {
          lwss.insert(one_lws);
        }
      }
    }
  };
  std::set<bool> tune_reverses{true, false};
  std::set<size_t> divisors;
  if (tune_type == lite_api::CL_TUNE_NONE) {
    return lwss;
  } else if (tune_type == lite_api::CL_TUNE_RAPID) {
    divisors = {1, 2, 4, 8};
  } else if (tune_type == lite_api::CL_TUNE_NORMAL) {
    divisors = {1, 3, 5, 7, 9, 11, 13};
  } else if (tune_type == lite_api::CL_TUNE_EXHAUSTIVE) {
    divisors = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  } else {
    LOG(FATAL) << "Unsupported opencl tune type:" << tune_type;
  }

#undef GEN_LOCAL_LWS
#ifdef GEN_LOCAL_LWS
  gen_lws(tune_reverses, divisors);
#endif

#undef GEN_MUL_LWS
#ifdef GEN_MUL_LWS
  std::vector<uint32_t> lws = {1, 1, 1};
  for (lws[0] = 1; lws[0] < gws[0] * 2; lws[0] *= 2) {
    for (lws[1] = 1; lws[1] < gws[1] * 2; lws[1] *= 2) {
      for (lws[2] = 1; lws[2] < gws[2] * 2; lws[2] *= 2) {
        if (lws[0] * lws[1] * lws[2] <= max_ws &&
            lws[0] * lws[1] * lws[2] >= 32) {
          lwss.insert(cl::NDRange{lws[0], lws[1], lws[2]});
        }
      }
    }
  }
#endif  // GEN_DIV_LWS

#define GEN_DIV_LWS
#ifdef GEN_DIV_LWS
  auto GetDivisors = [&](int number) {
    const int max_divisor = static_cast<int>(sqrt(number));
    std::vector<int> divisors;
    divisors.reserve(max_divisor / 3 + 1);
    for (int i = 1; i <= max_divisor; ++i) {
      const int d = number / i;
      if (i * d == number) {
        divisors.push_back(i);
        if (d != i) {
          divisors.push_back(d);
        }
      }
    }
    return divisors;
  };
  std::vector<int> locals_x = GetDivisors(static_cast<int>(gws[0]));
  std::vector<int> locals_y = GetDivisors(static_cast<int>(gws[1]));
  std::vector<int> locals_z = GetDivisors(static_cast<int>(gws[2]));

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
                group_size >= 32) {
              lwss.insert(cl::NDRange{static_cast<size_t>(x),
                                      static_cast<size_t>(y),
                                      static_cast<size_t>(z)});
            }
          }
        }
      }
    }
  }
#endif  // GEN_DIV_LWS
  return lwss;
}
std::set<cl::NDRange, CLContext::CompareByRange>
CLContext::DefaultLocalWorkSize(const cl::NDRange &gws,
                                register size_t max_ws,
                                size_t tune_type /*=0*/,
                                const int &divisor /*=2*/,
                                const bool &reverse /*=false*/,
                                const size_t &user_def_max_ws /*=0*/) {
  register size_t lx = reverse ? gws[2] : gws[0];
  register size_t ly = gws[1];
  register size_t lz = reverse ? gws[0] : gws[2];

  max_ws = (user_def_max_ws > 0 && user_def_max_ws <= max_ws) ? user_def_max_ws
                                                              : max_ws;
  std::set<cl::NDRange, CompareByRange> lws_set;
  int ly_src = ly;
  int lx_src = lx;
  int lz_src = lz;
  max_ws = divisor > 1 ? max_ws / divisor : max_ws;
  do {
    ly = ly_src;
    lx = lx_src;
    lz = lz_src;
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
    if (lx * ly * lz >= 32) {
      lws_set.insert(
          (reverse ? cl::NDRange{lz, ly, lx} : cl::NDRange{lx, ly, lz}));
    }
    ly_src = (ly_src & 0x01) ? 1 : ly_src >> 1;
  } while (ly_src > 1);
  if (tune_type == lite_api::CL_TUNE_NONE && lws_set.empty()) {
    lws_set.insert(
        (reverse ? cl::NDRange{lz, ly, lx} : cl::NDRange{lx, ly, lz}));
  }
  return lws_set;
}

bool CLContext::IsArmMali() {
  return CLRuntime::Global()->GetGpuType() == GpuType::ARM_MALI;
}

bool CLContext::IsAppleM1() {
  return CLRuntime::Global()->GetGpuType() == GpuType::APPLE_M1;
}

}  // namespace lite
}  // namespace paddle
