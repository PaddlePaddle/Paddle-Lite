/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "operators/box_coder_op.h"
#include <vector>
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void BoxCoderOp<Dtype, T>::InferShape() const {
  auto input_priorbox_dims = param_.InputPriorBox()->dims();
  auto input_priorboxvar_dims = param_.InputPriorBoxVar()->dims();
  auto input_targetbox_dims = param_.InputTargetBox()->dims();

  auto code_type = param_.CodeType();

  if (code_type == "encode_center_size") {
    if (input_targetbox_dims.size() != 2) {
      LOG(kLOG_ERROR) << " The rank of Input of TargetBox must be 2";
    }
    if (input_targetbox_dims[1] != 4) {
      LOG(kLOG_ERROR) << " The shape of TargetBox is [M, 4]";
    }
  }
  if (code_type == "decode_center_size") {
    if (input_targetbox_dims.size() != 3) {
      LOG(kLOG_ERROR) << "The rank of Input of TargetBox must be 3";
    }
    if (input_targetbox_dims[1] != input_priorbox_dims[0] ||
        input_targetbox_dims[2] != input_priorbox_dims[1]) {
      LOG(kLOG_ERROR) << " dimension not match";
    }
  }
  param_.OutputBox()->Resize(framework::make_ddim(
      {input_targetbox_dims[0], input_priorbox_dims[0], 4}));
}
template class BoxCoderOp<CPU, float>;
}  // namespace operators
}  // namespace paddle_mobile
