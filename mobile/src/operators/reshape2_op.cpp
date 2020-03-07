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

#ifdef RESHAPE2_OP

#include "operators/reshape2_op.h"
#include <vector>
#include "operators/kernel/reshape_kernel.h"
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void Reshape2Op<Dtype, T>::InferShape() const {
  if (this->param_.InputShape() != nullptr) {
    return;
  }
  auto &shape = this->param_.Shape();
  auto input_x_dims = this->param_.InputX()->dims();
  bool shouldResize = true;
  if (std::is_same<DeviceType<kGPU_CL>, Dtype>::value) {
    auto input_dim_size = input_x_dims.size();
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
        input_x_dims = this->param_.InputX()->dims();
      }
    }
  }

  auto out_dims = ValidateShape(shape, input_x_dims);
  this->param_.Out()->Resize(out_dims);
  if (std::is_same<DeviceType<kGPU_CL>, Dtype>::value) {
    input_x_dims = this->param_.InputX()->dims();
    shouldResize = true;
    if (out_dims.size() > 4) {
      for (int i = 0; i < out_dims.size() - 4; ++i) {
        if (out_dims[i] != 0 && out_dims[i] != 1) {
          shouldResize = false;
          break;
        }
      }
      if (shouldResize) {
        std::vector<int64_t> temp_output_dims;
        temp_output_dims.reserve(static_cast<size_t>(4));
        for (int i = out_dims.size() - 4; i < out_dims.size(); ++i) {
          temp_output_dims.push_back(out_dims[i]);
        }
        framework::DDim temp_ddim = framework::make_ddim(temp_output_dims);
        this->param_.Out()->Resize(temp_ddim);
      }
    }
  }
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
REGISTER_OPERATOR_CPU(reshape2, ops::Reshape2Op);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(reshape2, ops::Reshape2Op);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(reshape2, ops::Reshape2Op);
#endif

#endif
