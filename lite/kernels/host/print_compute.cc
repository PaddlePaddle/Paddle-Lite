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

#include <mutex>  // NOLINT
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";
const char kBoth[] = "BOTH";

class TensorFormatter {
 public:
  TensorFormatter() {}

  std::string Format(const Tensor& print_tensor,
                     const std::string& tensor_name = "",
                     const std::string& message = "") {
    std::stringstream log_stream;
    if (!tensor_name.empty()) {
      log_stream << "Variable: " << tensor_name << std::endl;
    }

    if (!message.empty()) {
      log_stream << "  - message: " << message << std::endl;
    }

    if (print_tensor_lod_) {
      log_stream << "  - lod: {";
      const LoD& lod = print_tensor.lod();
      for (auto level : lod) {
        log_stream << "{";
        bool is_first = true;
        for (auto i : level) {
          if (is_first) {
            log_stream << i;
            is_first = false;
          } else {
            log_stream << ", " << i;
          }
        }
        log_stream << "}";
      }
      log_stream << "}" << std::endl;
    }

    log_stream << "  - place: " << TargetToStr(print_tensor.target())
               << std::endl;  // TODO(hong19860320) always kHost

    if (print_tensor_shape_) {
      log_stream << "  - shape: " << print_tensor.dims().repr() << std::endl;
    }

    if (print_tensor_layout_) {
      log_stream << "  - layout: "
                 << DataLayoutToStr(
                        DATALAYOUT(kNCHW))  // TODO(hong19860320) Query the data
                                            // layout from target tensor
                 << std::endl;
    }

    auto dtype = print_tensor.precision();
    if (print_tensor_type_) {
      log_stream << "  - dtype: " << PrecisionToStr(dtype) << std::endl;
    }

    if (dtype == PRECISION(kBool)) {
      FormatData<bool>(print_tensor, log_stream);
    } else if (dtype == PRECISION(kInt8)) {
      FormatData<int8_t>(print_tensor, log_stream);
    } else if (dtype == PRECISION(kInt16)) {
      FormatData<int16_t>(print_tensor, log_stream);
    } else if (dtype == PRECISION(kInt32)) {
      FormatData<int32_t>(print_tensor, log_stream);
    } else if (dtype == PRECISION(kInt64)) {
      FormatData<int64_t>(print_tensor, log_stream);
    } else if (dtype == PRECISION(kFloat)) {
      FormatData<float>(print_tensor, log_stream);
    } else {
      log_stream << "\tdata: unprintable type: " << PrecisionToStr(dtype)
                 << std::endl;
    }
    return log_stream.str();
  }

  void Print(const Tensor& print_tensor,
             const std::string& tensor_name = "",
             const std::string& message = "") {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << Format(print_tensor, tensor_name, message);
  }

  void SetPrintTensorType(bool print_tensor_type) {
    print_tensor_type_ = print_tensor_type;
  }
  void SetPrintTensorShape(bool print_tensor_shape) {
    print_tensor_shape_ = print_tensor_shape;
  }
  void SetPrintTensorLod(bool print_tensor_lod) {
    print_tensor_lod_ = print_tensor_lod;
  }
  void SetPrintTensorLayout(bool print_tensor_layout) {
    print_tensor_layout_ = print_tensor_layout;
  }
  void SetSummarize(int64_t summarize) { summarize_ = summarize; }

 private:
  template <typename T>
  void FormatData(const Tensor& print_tensor, std::stringstream& log_stream) {
    int64_t print_size = summarize_ == -1
                             ? print_tensor.numel()
                             : (std::min)(summarize_, print_tensor.numel());
    const T* data = print_tensor.data<T>();  // Always kHost, so unnessary to
                                             // copy the data from device
    log_stream << "  - data: [";
    if (print_size > 0) {
      log_stream << data[0];
      for (int64_t i = 1; i < print_size; ++i) {
        log_stream << " " << data[i];
      }
    }
    log_stream << "]" << std::endl;
  }

  int64_t summarize_ = -1;
  bool print_tensor_type_ = true;
  bool print_tensor_shape_ = true;
  bool print_tensor_lod_ = true;
  bool print_tensor_layout_ = true;
};

void PrintCompute::Run() {
  auto& param = Param<param_t>();
  param.out->CopyDataFrom(*param.in);

  if ((param.is_forward && param.print_phase == kBackward) ||
      (!param.is_forward && param.print_phase == kForward)) {
    return;
  }

  int first_n = param.first_n;
  if (first_n > 0 && ++times_ > first_n) return;

  TensorFormatter formatter;
  const std::string& name = param.print_tensor_name ? param.name : "";
  formatter.SetPrintTensorType(param.print_tensor_type);
  formatter.SetPrintTensorShape(param.print_tensor_shape);
  formatter.SetPrintTensorLod(param.print_tensor_lod);
  formatter.SetPrintTensorLayout(param.print_tensor_layout);
  formatter.SetSummarize(static_cast<int64_t>(param.summarize));
  formatter.Print(*param.in, name, param.message);
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
