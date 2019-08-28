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

#ifdef SLICE_OP

#include "operators/slice_op.h"
#include <algorithm>
#include <vector>

namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void SliceOp<Dtype, T>::InferShape() const {
  auto axes = this->param_.axes_;
  auto input = this->param_.input_;
  auto output = this->param_.output_;
  if (std::is_same<DeviceType<kGPU_CL>, Dtype>::value) {
    auto output_dims = output->dims();
    auto output_dims_size = output_dims.size();
    bool should_resize = true;
    if (output_dims_size > 4) {
      for (int i = 0; i < output_dims_size - 4; ++i) {
        if (output_dims[i] != 0 && output_dims[i] != 1) {
          should_resize = false;
          break;
        }
      }
      if (should_resize) {
        std::vector<int64_t> temp_output_dims;
        temp_output_dims.reserve(static_cast<size_t>(4));
        for (int i = output_dims_size - 4; i < output_dims_size; ++i) {
          temp_output_dims.push_back(output_dims[i]);
        }
        framework::DDim temp_ddim = framework::make_ddim(temp_output_dims);
        this->param_.output_->Resize(temp_ddim);
      }
    }
  }
  PADDLE_MOBILE_ENFORCE(axes.size() == 1, "axes size should equals 1");
  PADDLE_MOBILE_ENFORCE(input->dims().size() == output->dims().size(),
                        "input dim size should equals output dim size");
  if (std::is_same<DeviceType<kGPU_CL>, Dtype>::value) {
    PADDLE_MOBILE_ENFORCE(
        output->dims().size() -
                (axes[0] - (this->param_.original_output_dims_size_ -
                            this->param_.output_->dims().size())) ==
            3,
        "op only support slice channel now");
  }
  auto starts = this->param_.starts_;
  auto ends = this->param_.ends_;
  framework::DDim out_dims(input->dims());
  PADDLE_MOBILE_ENFORCE(starts.size() == ends.size(),
                        "starts.size should equal ends.size");
  PADDLE_MOBILE_ENFORCE(axes.size() == starts.size(),
                        "axes.size should equal starts.size");
  int dim_value, start, end;
  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = axes[i] - (this->param_.original_output_dims_size_ -
                          this->param_.output_->dims().size());
    dim_value = out_dims[axis];
    if (dim_value > 0) {
      start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      // start = std::min(start, dim_value);
      end = std::min(end, dim_value);
      // start = std::min(start, end);
      PADDLE_MOBILE_ENFORCE(end > start, "end should greater than start");
      out_dims[axis] = end - start;
    }
  }
  output->Resize(out_dims);
  if (std::is_same<DeviceType<kCPU>, Dtype>::value) {
    LoDTensor *output_lod = reinterpret_cast<LoDTensor *>(output);
    LoDTensor *input_lod = reinterpret_cast<LoDTensor *>(input);
    if (axes[0] != 0) {
      output_lod->set_lod(input_lod->lod());
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(slice, ops::SliceOp);
#endif
#ifdef PADDLE_MOBILE_FPGA
REGISTER_OPERATOR_FPGA(slice, ops::SliceOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(slice, ops::SliceOp);
#endif
#endif  // SLICE_OP
