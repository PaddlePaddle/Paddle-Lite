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

#include "lite/kernels/host/print_compute.h"
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";
const char kBoth[] = "BOTH";

class LogGuard {
 public:
  inline LogGuard() { LogMutex().lock(); }

  inline ~LogGuard() { LogMutex().unlock(); }

 private:
  static std::mutex &LogMutex() {
    static std::mutex mtx;
    return mtx;
  }
};

struct Formater {
  std::string message;
  std::string name;
  std::vector<int64_t> dims;
  PrecisionType dtype{PRECISION(kUnk)};
  LoD lod;
  int summarize;
  const void *data{nullptr};
  std::stringstream logs;

  void operator()(size_t size) {
    PrintMessage();
    PrintName();
    PrintDims();
    PrintDtype();
    PrintLod();
    PrintData(size);
    LogGuard guard;
    std::cout << logs.str();
  }

 private:
  void PrintMessage() { logs << std::time(nullptr) << "\t" << message << "\t"; }
  void PrintName() {
    if (!name.empty()) {
      logs << "Tensor[" << name << "]" << std::endl;
    }
  }
  void PrintDims() {
    if (!dims.empty()) {
      logs << "\tshape: [";
      for (auto i : dims) {
        logs << i << ",";
      }
      logs << "]" << std::endl;
    }
  }
  void PrintDtype() {
    logs << "\tdtype: " << PrecisionToStr(dtype) << std::endl;
  }
  void PrintLod() {
    if (!lod.empty()) {
      logs << "\tLoD: [";
      for (auto level : lod) {
        logs << "[ ";
        for (auto i : level) {
          logs << i << ",";
        }
        logs << " ]";
      }
      logs << "]" << std::endl;
    }
  }

  void PrintData(size_t size) {
    CHECK(data);
    if (dtype == PRECISION(kBool)) {
      Display<bool>(size);
    } else if (dtype == PRECISION(kInt8)) {
      Display<int8_t>(size);
    } else if (dtype == PRECISION(kInt16)) {
      Display<int16_t>(size);
    } else if (dtype == PRECISION(kInt32)) {
      Display<int32_t>(size);
    } else if (dtype == PRECISION(kInt64)) {
      Display<int64_t>(size);
    } else if (dtype == PRECISION(kFloat)) {
      Display<float>(size);
    } else {
      logs << "\tdata: unprintable type: " << PrecisionToStr(dtype)
           << std::endl;
    }
  }

  template <typename T>
  void Display(size_t size) {
    const auto *d = reinterpret_cast<const T *>(data);
    logs << "\tdata: ";
    if (summarize != -1) {
      summarize = std::min(size, (size_t)summarize);
      for (int i = 0; i < summarize; i++) {
        logs << d[i] << ",";
      }
    } else {
      for (size_t i = 0; i < size; i++) {
        logs << d[i] << ",";
      }
    }
    logs << std::endl;
  }
};

void PrintCompute::Run() {
  auto &param = Param<param_t>();
  param.out->CopyDataFrom(*param.in);
  // Print value
  if ((param.is_forward && param.print_phase == kBackward) ||
      (!param.is_forward && param.print_phase == kForward)) {
    return;
  }
  Formater formater;
  formater.message = param.message;
  if (param.print_tensor_name) {
    formater.name = param.name;
  }
  if (param.print_tensor_type) {
    formater.dtype = param.in->precision();
  }
  if (param.print_tensor_shape) {
    auto &dims = param.in->dims();
    formater.dims.resize(dims.size());
    for (int i = 0; i < dims.size(); ++i) formater.dims[i] = dims[i];
  }
  if (param.print_tensor_lod) {
    formater.lod = param.in->lod();
  }
  formater.summarize = param.summarize;
  formater.data = reinterpret_cast<const void *>(param.in->raw_data());
  formater(param.in->dims().production());
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    print, kHost, kAny, kAny, paddle::lite::kernels::host::PrintCompute, def)
    .BindInput("In",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
