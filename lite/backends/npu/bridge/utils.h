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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/core/mir/node.h"
#include "lite/core/op_lite.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

std::string UniqueName(const std::string& prefix);

ge::DataType PrecisionConverter(PrecisionType itype);

ge::Format DataLayoutConverter(DataLayoutType itype);

ge::TensorPtr CvtFromLiteTensor(Tensor* in_tensor,
                                std::vector<int64_t> out_shape = {},
                                PrecisionType in_ptype = PRECISION(kFloat),
                                DataLayoutType in_ltype = DATALAYOUT(kNCHW));

template <typename T>
ge::TensorPtr CreateTensorAndFillData(std::vector<T> data,
                                      std::vector<int64_t> shape = {},
                                      ge::Format format = ge::FORMAT_NCHW) {
  const std::type_info& info = typeid(T);
  ge::DataType type = ge::DT_FLOAT;
  if (info == typeid(float)) {
    type = ge::DT_FLOAT;
  } else if (info == typeid(int8_t)) {
    type = ge::DT_INT8;
  } else if (info == typeid(int32_t)) {
    type = ge::DT_INT32;
  } else {
    LOG(FATAL) << "Unknow value type " << info.name();
  }
  if (shape.empty()) {
    shape = {static_cast<int64_t>(data.size())};
  } else {
    int size = 1;
    for (auto i : shape) {
      size *= i;
    }
    CHECK_EQ(data.size(), size);
  }
  ge::TensorDesc desc(ge::Shape(shape), format, type);
  ge::TensorPtr tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(desc);
  tensor->SetData(reinterpret_cast<uint8_t*>(data.data()),
                  data.size() * sizeof(T));
  return tensor;
}

template <typename T>
ge::TensorPtr CreateTensorAndFillData(T value,
                                      std::vector<int64_t> shape = {1},
                                      ge::Format format = ge::FORMAT_NCHW) {
  int64_t size = 1;
  for (auto i : shape) {
    size *= i;
  }
  std::vector<T> data(size, value);
  return CreateTensorAndFillData(data, shape, format);
}

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle
