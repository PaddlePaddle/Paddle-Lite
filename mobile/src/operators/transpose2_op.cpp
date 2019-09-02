/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef TRANSPOSE2_OP

#include <vector>

#include "common/enforce.h"
#include "operators/transpose2_op.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void Transpose2Op<Dtype, T>::InferShape() const {
  auto input_x_dims = this->param_.InputX()->dims();
  auto axis = this->param_.Axis();

  size_t x_dims_size = input_x_dims.size();
  size_t axis_size = axis.size();

  if (std::is_same<DeviceType<kGPU_CL>, Dtype>::value) {
    bool shouldResize = true;
    int diff_dim = 0;
    if (axis_size > 4) {
      for (int i = 0; i < axis_size - 4; ++i) {
        if (axis[i] != i) {
          shouldResize = false;
          break;
        } else {
          diff_dim++;
        }
      }
      if (shouldResize) {
        std::vector<int> temp_axis_dims;
        temp_axis_dims.reserve(static_cast<size_t>(4));
        for (int i = axis_size - 4; i < axis_size; ++i) {
          temp_axis_dims.push_back(axis[i] - diff_dim);
        }
        axis.resize(4);
        axis.clear();
        axis.insert(axis.begin(), temp_axis_dims.begin(), temp_axis_dims.end());
      }
    }

    auto input_dim_size = input_x_dims.size();
    shouldResize = true;
    if (input_dim_size > 4) {
      for (int i = 0; i < input_dim_size - 4; ++i) {
        if (input_x_dims[i] != 0 && input_x_dims[i] != 1) {
          shouldResize = false;
          break;
        }
      }
      if (shouldResize) {
        std::vector<int64_t> temp_intput_dims;
        temp_intput_dims.reserve(static_cast<size_t>(4));
        for (int i = input_dim_size - 4; i < input_dim_size; ++i) {
          temp_intput_dims.push_back(input_x_dims[i]);
        }
        framework::DDim temp_ddim = framework::make_ddim(temp_intput_dims);
        this->param_.InputX()->Resize(temp_ddim);
      }
    }

    axis_size = axis.size();
    input_x_dims = this->param_.InputX()->dims();
    x_dims_size = input_x_dims.size();
  }

  PADDLE_MOBILE_ENFORCE((x_dims_size == axis_size),
                        "input_dims must "
                        "be equal to the axis_size. ")

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    PADDLE_MOBILE_ENFORCE(
        axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
        "Each element of Attribute axis should be a unique value "
        "range from 0 to (dims - 1), "
        "where the dims is the axis's size");
  }
  framework::DDim out_dims(input_x_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = input_x_dims[axis[i]];
  }
  this->param_.Out()->Resize(out_dims);
  std::vector<int64_t> xshape_dims(input_x_dims.size() + 1, 0);
  for (int i = 0; i < input_x_dims.size(); ++i) {
    xshape_dims[i + 1] = input_x_dims[i];
  }
  this->param_.OutputXShape()->Resize(framework::make_ddim(xshape_dims));
  if (std::is_same<DeviceType<kGPU_CL>, Dtype>::value) {
    this->param_.OutputXShape()->Resize(input_x_dims);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(transpose2, ops::Transpose2Op);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(transpose2, ops::Transpose2Op);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(transpose2, ops::Transpose2Op);
#endif
#endif  // TRANSPOSE_OP
