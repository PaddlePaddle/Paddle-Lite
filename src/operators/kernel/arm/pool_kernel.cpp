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
#include "common/log.h"
#include <operators/kernel/pool_kernel.h>

namespace paddle_mobile {
namespace operators {

inline void PoolBasic(std::string pooling_type, std::vector<int> ksize,
                      std::vector<int> strides, std::vector<int> paddings,
                      const Tensor *in_x, Tensor *out) {
    if (pooling_type == "max") {
        math::PoolFunctor<CPU, math::MaxPool<float>, float> pool2d_forward;
        math::MaxPool<float> pool_process;
        pool2d_forward(*in_x, ksize, strides, paddings, pool_process, out);

    } else if (pooling_type == "avg") {
        math::PoolFunctor<CPU, math::AvgPool<float>, float> pool2d_forward;
        math::AvgPool<float> pool_process;
        pool2d_forward(*in_x, ksize, strides, paddings, pool_process, out);
    }
}

template <> void PoolKernel<CPU, float>::Compute(const PoolParam &param) const {
    const Tensor *in_x = param.Input();
    Tensor *out = param.Output();
    std::string pooling_type = param.PoolingType();

    std::vector<int> ksize = param.Ksize();

    std::vector<int> strides = param.Strides();

    std::vector<int> paddings = param.Paddings();
    if (ksize.size() != 2) {
        LOG(paddle_mobile::LogLevel::kLOG_ERROR)
            << "Pool op only supports 2D and 3D input.";
    }

    if (param.isGlobalPooling()) {
        for (size_t i = 0; i < ksize.size(); ++i) {
            paddings[i] = 0;
            ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
        }
    }

    PoolBasic(pooling_type, ksize, strides, paddings, in_x, out);

    //    if (param.isGlobalPooling() || ksize[0] != ksize[1] ||
    //        strides[0] != strides[1] || strides[1] != 2 ||
    //        paddings[0] != paddings[1] || paddings[1] > 1) {
    //        PoolBasic(pooling_type, ksize, strides, paddings, in_x, out);
    //
    //    } else if (ksize[0] == 2) {
    //
    //    } else if (ksize[0] == 3) {
    //
    //    } else {
    //        PoolBasic(pooling_type, ksize, strides, paddings, in_x, out);
    //    }
}
} // namespace operators
} // namespace paddle_mobile
