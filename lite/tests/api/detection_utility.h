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

#pragma once
#include <gflags/gflags.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

template <class T = float>
void SetDetectionInput(
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
    std::vector<int> input_shape,
    std::vector<T> raw_data,
    int input_size) {
  auto input_names = predictor->GetInputNames();
  int batch_size = input_shape[0];
  int rh = input_shape[2];
  int rw = input_shape[3];
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor->GetInputByName(tensor_name);
    if (tensor_name == "image") {
      in_tensor->Resize(
          std::vector<int64_t>(input_shape.begin(), input_shape.end()));
      auto* input_data = in_tensor->mutable_data<T>();
      if (raw_data.empty()) {
        for (int i = 0; i < input_size; i++) {
          input_data[i] = 0.f;
        }
      } else {
        memcpy(input_data, raw_data.data(), sizeof(T) * input_size);
      }
    } else if (tensor_name == "im_shape" || tensor_name == "im_size") {
      in_tensor->Resize({batch_size, 2});
      auto* im_shape_data = in_tensor->mutable_data<T>();
      for (int i = 0; i < batch_size * 2; i += 2) {
        im_shape_data[i] = rh;
        im_shape_data[i + 1] = rw;
      }
    } else if (tensor_name == "scale_factor") {
      in_tensor->Resize({batch_size, 2});
      auto* scale_factor_data = in_tensor->mutable_data<T>();
      for (int i = 0; i < batch_size * 2; i++) {
        scale_factor_data[i] = 1;
      }
    } else {
      LOG(FATAL) << "Unsupported the input: " << tensor_name;
    }
  }
}

}  // namespace lite
}  // namespace paddle
