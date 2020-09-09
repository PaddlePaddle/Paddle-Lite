// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/rknpu/bridges/utility.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "rknpu/rknpu_pub.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

rk::nn::PrecisionType ToRknpuPrecisionType(PrecisionType precision) {
  rk::nn::PrecisionType t = rk::nn::PrecisionType::UNKNOWN;

  switch (precision) {
    case PrecisionType::kFloat:
      t = rk::nn::PrecisionType::FLOAT32;
      break;
    case PrecisionType::kFP16:
      t = rk::nn::PrecisionType::FLOAT16;
      break;
    case PrecisionType::kInt16:
      t = rk::nn::PrecisionType::INT16;
      break;
    case PrecisionType::kInt32:
      t = rk::nn::PrecisionType::INT32;
      break;
    case PrecisionType::kInt64:
      t = rk::nn::PrecisionType::INT64;
      break;
    case PrecisionType::kInt8:
      t = rk::nn::PrecisionType::INT8;
      break;
    case PrecisionType::kBool:
      t = rk::nn::PrecisionType::BOOL8;
      break;
    default:
      break;
  }

  return t;
}

rk::nn::DataLayoutType ToRknpuDataLayoutType(DataLayoutType layout) {
  rk::nn::DataLayoutType t = rk::nn::DataLayoutType::UNKNOWN;

  switch (layout) {
    case DataLayoutType::kNCHW:
      t = rk::nn::DataLayoutType::NCHW;
      break;
    case DataLayoutType::kNHWC:
      t = rk::nn::DataLayoutType::NHWC;
      break;
    default:
      break;
  }

  return t;
}

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname) {
  auto iarg_names = op_info->input_argnames();
  if (std::find(iarg_names.begin(), iarg_names.end(), argname) !=
      iarg_names.end()) {
    auto inputs = op_info->Input(argname);
    if (inputs.empty()) {
      return false;
    }
    auto var_name = inputs.front();
    auto var = scope->FindVar(var_name);
    return var != nullptr;
  } else {
    return false;
  }
}
}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
