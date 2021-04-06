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

#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/profile/timer.h"
#include "lite/core/tensor.h"
#include "lite/utils/replace_stl/stream.h"

#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_include.h"
#endif

namespace paddle {
namespace lite {
namespace profile {

enum class Type {
  kUnk = 0,
  kCreate,
  kDispatch,
};

extern std::map<Type, std::string> TypeStr;

struct TimeInfo {
  float avg;
  float min;
  float max;
#ifdef LITE_WITH_OPENCL
  float cl_avg;
  float cl_min;
  float cl_max;
#endif
};

struct OpCharacter {
  TargetType target;
  void* op_lite{nullptr};
  std::string op_type{std::string("N/A")};
  std::string kernel_name{std::string("N/A")};
  std::string kernel_attr{std::string("N/A")};
  std::string kernel_func_name{std::string("N/A")};
  std::string remark{std::string("N/A")};

  std::string input_shape{"N/A"};
  std::string output_shape{"N/A"};
  std::string filter_shape{"N/A"};

  float macs{0};
  float macs_ps{0};

  float io_duration{0};

#ifdef LITE_WITH_OPENCL
  cl::Event cl_event{};
  std::string global_work_size{"N/A"};
  std::string local_work_size{"N/A"};

  std::string NDRangeToStr(const cl::NDRange& range) {
    std::string range_str{""};
    const size_t range_dims = range.dimensions();
    if (range_dims == 0) return "NullRange";
    for (size_t i = 0; i < range_dims; ++i) {
      range_str += std::to_string(range[i]);
      if (i != range_dims - 1) {
        range_str += ",";
      }
    }
    return range_str;
  }
#else
  void* cl_event{nullptr};
#endif

  std::string DimToStr(const paddle::lite::DDimLite& dim) {
    if (!dim.size()) return "NotImpl";
    std::string dim_str{""};
    for (size_t i = 0; i < dim.size(); ++i) {
      dim_str += std::to_string(dim[i]);
      if (i != dim.size() - 1) {
        dim_str += "x";
      }
    }
    return dim_str;
  }

  std::string str() {
    std::string str{""};
    str += kernel_name + "/" + kernel_func_name + "/" + remark + "/" +
           input_shape + "/" + filter_shape + "/" + output_shape;
    return str;
  }
};

class StatisUnit final {
 public:
  explicit StatisUnit(const OpCharacter& ch);
  lite::profile::Timer* Timer(Type type);
  OpCharacter& Character() { return character; }

  OpCharacter character;

 protected:
  std::unique_ptr<lite::profile::Timer> create_t;
  std::unique_ptr<lite::profile::Timer> dispatch_t;
};

class Profiler final {
 public:
  Profiler() = default;
  explicit Profiler(const std::string& name) : name_(name) {}
  int NewTimer(const OpCharacter& ch);
  void StartTiming(Type type, const int index, KernelContext* ctx);
  void StopTiming(Type type, const int index, KernelContext* ctx);
  std::string Summary(Type type, bool concise = true, size_t warm_up = 10);
  int GetKernelFuncCalledTimes(const std::string& op_type,
                               const std::string& kernel_attr,
                               const std::string& kernel_func_name);
  float GetKernelFuncSummaryGOPs(const std::string& op_type,
                                 const std::string& kernel_attr,
                                 const std::string& kernel_func_name);
  OpCharacter* GetOpCharacter(const size_t index);

 private:
  std::string name_{std::string("N/A")};
  std::vector<StatisUnit> units_;
};

}  // namespace profile
}  // namespace lite
}  // namespace paddle
