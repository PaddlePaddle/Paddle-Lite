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

#include "lite/backends/fpga/KD/pes/input_pe.hpp"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

class FeedCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::FeedParam;

  std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() override {
    std::unique_ptr<type_infer_handler_t> res(new type_infer_handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      // std::cout << "inputs: " << inputs << std::endl;
      auto* type = inputs.at("Input");
      std::cout << "type: " << type << std::endl;
      exit(-1);
      CHECK(type->target() == TARGET(kHost));

      auto in_place = type->place();
      auto target = TARGET(kFPGA);
      auto precision = in_place.precision;
      auto layout = in_place.layout;

      if (in_place.precision == PRECISION(kFloat)) {
        precision = PRECISION(kFP16);
        layout = DATALAYOUT(kNHWC);
      }

      auto* out_type =
          Type::Get(type->id(), target, precision, layout, in_place.device);
      return out_type;
    };
    return res;
  }

  void PrepareForRun() override;

  void Run() override;

 private:
  zynqmp::InputPE pe_;
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
