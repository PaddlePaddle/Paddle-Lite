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
namespace lite_metal {
namespace subgraph {
namespace rknpu {

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

rk::nn::PrecisionType CvtPrecisionType(PrecisionType itype) {
  rk::nn::PrecisionType otype = rk::nn::PrecisionType::UNKNOWN;
  switch (itype) {
    case PrecisionType::kFloat:
      otype = rk::nn::PrecisionType::FLOAT32;
      break;
    case PrecisionType::kFP16:
      otype = rk::nn::PrecisionType::FLOAT16;
      break;
    case PrecisionType::kInt16:
      otype = rk::nn::PrecisionType::INT16;
      break;
    case PrecisionType::kInt32:
      otype = rk::nn::PrecisionType::INT32;
      break;
    case PrecisionType::kInt64:
      otype = rk::nn::PrecisionType::INT64;
      break;
    case PrecisionType::kInt8:
      otype = rk::nn::PrecisionType::INT8;
      break;
    case PrecisionType::kBool:
      otype = rk::nn::PrecisionType::BOOL8;
      break;
    default:
      LOG(FATAL) << "[Rockchip NPU] Can not convert precision type("
                 << PrecisionToStr(itype) << ") from Lite to Rockchip NPU";
      break;
  }
  return otype;
}

rk::nn::DataLayoutType CvtDataLayoutType(DataLayoutType itype) {
  rk::nn::DataLayoutType otype = rk::nn::DataLayoutType::UNKNOWN;
  switch (itype) {
    case DataLayoutType::kNCHW:
      otype = rk::nn::DataLayoutType::NCHW;
      break;
    case DataLayoutType::kNHWC:
      otype = rk::nn::DataLayoutType::NHWC;
      break;
    default:
      LOG(WARNING) << "[Rockchip NPU] Can not convert data layout type("
                   << DataLayoutToStr(itype) << ") from Lite to Rockchip NPU";
      break;
  }
  return otype;
}

std::vector<int32_t> CvtShape(const std::vector<int64_t>& in_shape) {
  std::vector<int32_t> out_shape;
  for (size_t i = 0; i < in_shape.size(); i++) {
    out_shape.push_back(static_cast<int32_t>(in_shape[i]));
  }
  return out_shape;
}

std::vector<int32_t> CvtShape(const DDim& in_dims) {
  return CvtShape(in_dims.Vectorize());
}

std::shared_ptr<rk::nn::Tensor> CvtTensor(rk::nn::Graph* graph,
                                          const std::string& name,
                                          const std::vector<int64_t>& shape,
                                          const std::vector<float>& scales,
                                          void* data,
                                          PrecisionType precision,
                                          DataLayoutType layout) {
  CHECK(!name.empty()) << "[Rockchip NPU] The name of RKNPU tensor is empty!";
  auto attr = std::make_shared<rk::nn::TensorAttr>();
  attr->name = name;
  attr->role =
      data == nullptr ? rk::nn::TensorRole::VAR : rk::nn::TensorRole::CONST;
  attr->dims = CvtShape(shape);
  attr->precision = CvtPrecisionType(precision);
  attr->layout = CvtDataLayoutType(layout);
  switch (precision) {
    case PrecisionType::kInt8:
      attr->qntBits = 8;
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale = scales;
      break;
    case PrecisionType::kInt32:
      attr->qntBits = 32;
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale = scales;
      break;
    default:
      // in RKNPU some const node such as batchnorm's scale, mean is float32
      // const node such as conv's bias is int8
      LOG(WARNING) << "[Rockchip NPU] Can not convert precision type("
                   << PrecisionToStr(precision)
                   << ") from Lite to Rockchip NPU";
      break;
  }
  auto tensor = graph->CreateTensor(attr, data);
  CHECK(tensor != nullptr);
  return tensor;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
