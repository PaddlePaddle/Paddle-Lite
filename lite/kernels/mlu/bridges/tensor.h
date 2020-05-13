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

#include <fstream>
#include <string>
#include <vector>
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

class MLUTensor {
 public:
  MLUTensor()
      : mlu_tensor_(nullptr),
        tensor_type_(CNML_TENSOR),
        mlu_dtype_(CNML_DATA_FLOAT32) {}

  void set_mlu_ptr(void* mlu_data) { mlu_ptr_ = mlu_data; }

  MLUTensor(const std::vector<int64_t>& shape,
            cnmlTensorType_t tensor_type = CNML_TENSOR,
            cnmlDataOrder_t shape_order = CNML_NCHW,
            cnmlDataType_t mlu_dtype = CNML_DATA_FLOAT32,
            cnmlDataOrder_t data_order = CNML_NHWC);

  void remember(const std::vector<int>& shape,
                cnmlTensorType_t tensor_type,
                cnmlDataType_t mlu_dtype,
                cnmlDataOrder_t shape_order,
                cnmlDataOrder_t data_order);
  void Create();
  cnmlTensor_t mlu_tensor();
  void* mlu_data() {
    CHECK(mlu_ptr_ != nullptr);
    return mlu_ptr_;
  }

  cnmlDataType_t dtype() { return mlu_dtype_; }
  void set_mlu_dtype(cnmlDataType_t type) { mlu_dtype_ = type; }

  const std::vector<int64_t>& get_origin_shape() const { return origin_shape_; }

  ~MLUTensor();

  void ToFile(std::string file_name);
  cnmlDataOrder_t dorder() { return data_order_; }

 private:
  cnmlTensor_t mlu_tensor_;

  std::vector<int> shape_;
  std::vector<int64_t> origin_shape_;
  cnmlTensorType_t tensor_type_;
  cnmlDataType_t mlu_dtype_;
  int dim_{0};
  cnmlDataOrder_t data_order_;
  void* mlu_ptr_;
};

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
