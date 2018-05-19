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

#include "pool_op.h"

namespace paddle_mobile {
namespace operators {

int PoolOutputSize(int input_size, int filter_size, int padding, int stride,
                   bool ceil_mode) {
    int output_size;
    if (!ceil_mode) {
        output_size = (input_size - filter_size + 2 * padding) / stride + 1;
    } else {
        output_size =
            (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
    }
    return output_size;
}
template <typename DeviceType, typename T>
void PoolOp<DeviceType, T>::InferShape() const {
    auto in_x_dims = param_.Input()->dims();
    std::vector<int> ksize = param_.Ksize();
    std::vector<int> paddings = param_.Paddings();
    std::vector<int> strides = param_.Strides();
    bool ceil_mode = param_.isCeilMode();

    if (param_.isGlobalPooling()) {
        ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
        for (size_t i = 0; i < ksize.size(); ++i) {
            paddings[i] = 0;
            ksize[i] = static_cast<int>(in_x_dims[i + 2]);
        }
    }
    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
        output_shape.push_back(PoolOutputSize(
            in_x_dims[i + 2], ksize[i], paddings[i], strides[i], ceil_mode));
    }
    param_.Output()->Resize(framework::make_ddim(output_shape));
    DLOG << "infer shape out size =" << param_.Output()->numel();
}
template class PoolOp<CPU, float>;
} // namespace operators
} // namespace paddle_mobile
