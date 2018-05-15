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

#include "conv_op.h"
#include "framework/data_type.h"
#include "framework/op_proto_maker.h"
#include "framework/operator.h"

namespace paddle_mobile {
namespace operators {

int ConvOutputSize(int input_size, int filter_size, int dilation, int padding,
                   int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

template <typename Dtype, typename T>
void ConvOp<Dtype, T>::InferShape() const {
//  std::cout << " begin get dims: " << std::endl;

  auto in_dims = param_.Input()->dims();

//  std::cout << " end get in dims: " << std::endl;

  //  std::cout << " in_dims: " << in_dims << std::endl;

//  std::cout << " begin get Filter " << std::endl;

  auto filter_dims = param_.Filter()->dims();

//  std::cout << " end get Filter " << std::endl;

//  std::cout << " begin get Attrs " << std::endl;

  const std::vector<int> &strides = param_.Strides();

//  std::cout << " end get Attrs " << strides[0] << std::endl;

  std::vector<int> paddings = param_.Paddings();

  int groups = param_.Groups();

  std::vector<int> dilations = param_.Dilations();

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2], filter_dims[i + 2],
                                          dilations[i], paddings[i],
                                          strides[i]));
  }

  framework::DDim ddim = framework::make_ddim(output_shape);
  param_.Output()->Resize(ddim);
}

template class ConvOp<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile
