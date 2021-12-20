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
#include "driver/kunlunxin_xtcl/utility.h"
#include "utility/debug.h"
#include "utility/string.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

class Converter {
 public:
  explicit Converter(xtcl::network::xNetworkBuilder* builder,
                     xtcl::network::xTensorCompiler::ParamNDArrayMap* params,
                     std::map<hal::Operand*, std::vector<xtcl::xExpr>>* exprs)
      : builder_(builder), params_(params), exprs_(exprs) {}
  ~Converter() {}

  // Convert a NNAdapter model to XTC network and exprs
  int Apply(hal::Model* model);
  xtcl::network::xNetworkBuilder* builder() { return builder_; }
  // Mapping a XTCL expr to a NNAdapter operand
  xtcl::xExpr GetMappedExpr(hal::Operand* operand);
  xtcl::xExpr UpdateExprMap(hal::Operand* operand, xtcl::xExpr expr);
  template <typename T>
  std::shared_ptr<T> AddExpr(hal::Operand* operand = nullptr,
                             const std::string& custom_name = "") {
    std::string operand_id = OperandIdToString(operand);
    std::string expr_name = string_format("op_%d_%s_%s_%s",
                                          expr_index_++,
                                          typeid(T).name(),
                                          operand_id.c_str(),
                                          custom_name.c_str());
    return std::make_shared<T>(expr_name);
  }
  xtcl::xExpr AddConstantExpr(const void* values,
                              NNAdapterOperandPrecisionCode precision,
                              const std::vector<int32_t>& dimensions = {});
  xtcl::xExpr AddInt32ConstantExpr(const int32_t* values,
                                   const std::vector<int32_t>& dimensions);
  xtcl::xExpr AddInt32ConstantExpr(const std::vector<int32_t>& values,
                                   const std::vector<int32_t>& dimensions = {});
  xtcl::xExpr AddFloat32ConstantExpr(const float* values,
                                     const std::vector<int32_t>& dimensions);
  xtcl::xExpr AddFloat32ConstantExpr(
      const std::vector<float>& values,
      const std::vector<int32_t>& dimensions = {});
  // Convert a constant and model input operand and map to a XTCL expr
  xtcl::xExpr ConvertOperand(hal::Operand* operand,
                             std::vector<int32_t> dimensions = {});

 private:
  xtcl::network::xNetworkBuilder* builder_{nullptr};
  xtcl::network::xTensorCompiler::ParamNDArrayMap* params_{nullptr};
  std::map<hal::Operand*, std::vector<xtcl::xExpr>>* exprs_{nullptr};
  // Only for generating the unique name for XTCL expr
  uint32_t expr_index_{0};
};

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
