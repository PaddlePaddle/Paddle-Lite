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
#include <utility>
#include <vector>
#include "driver/imagination_nna/imgdnn_manager.h"
#include "driver/imagination_nna/utility.h"

namespace nnadapter {
namespace imagination_nna {

class Converter {
 public:
  explicit Converter(
      ImgdnnManager* imgdnn_mgr,
      std::map<core::Operand*, std::vector<imgdnn_tensor>>* tensors)
      : imgdnn_mgr_(imgdnn_mgr), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to imgdnn graph and tensors
  int Apply(core::Model* model);
  // Mapping a imgdnn tensor to a NNAdapter operand
  ImgdnnManager* GetImgdnnMgr() { return imgdnn_mgr_; }
  imgdnn_tensor GetMappedTensor(core::Operand* operand);
  imgdnn_tensor UpdateTensorMap(core::Operand* operand, imgdnn_tensor tensor);
  imgdnn_tensor AddTensor(int32_t* dimensions_data,
                          uint32_t dimensions_count,
                          imgdnn_type type,
                          const float* quant_scales,
                          const int32_t* zero_point,
                          uint32_t quant_scale_count,
                          uint32_t quant_channel_dim,
                          void* buffer);
  imgdnn_tensor AddTensor(const NNAdapterOperandType* type,
                          void* buffer,
                          std::vector<int32_t> dimensions);
  // Quant8 constant operand with asymmetric per-layer quantizion
  imgdnn_tensor AddQuant8ConstantTensor(uint8_t* values,
                                        int32_t* dimensions_data,
                                        uint32_t dimensions_count,
                                        float quant_scale,
                                        int32_t zero_point);
  // Quant8 constant operand with symmetric per-channel quantizion
  imgdnn_tensor AddQuant8ConstantTensor(int8_t* values,
                                        int32_t* dimensions_data,
                                        uint32_t dimensions_count,
                                        float* quant_scales,
                                        uint32_t quant_scale_count,
                                        uint32_t quant_channel_dim);
  // Quant32 constant operand with symmetric per-layer quantizion
  imgdnn_tensor AddQuant32ConstantTensor(int32_t* values,
                                         int32_t* dimensions_data,
                                         uint32_t dimensions_count,
                                         float quant_scale);
  imgdnn_tensor ConvertOperand(core::Operand* operand,
                               std::vector<int32_t> dimensions = {});

 private:
  ImgdnnManager* imgdnn_mgr_{nullptr};
  std::map<core::Operand*, std::vector<imgdnn_tensor>>* tensors_;
};

#define ADD_OPERATOR(__operator__, ...) \
  converter->GetImgdnnMgr()->__operator__(__VA_ARGS__)

}  // namespace imagination_nna
}  // namespace nnadapter
