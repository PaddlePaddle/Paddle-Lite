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
#ifndef MOBILE_DEEP_LEARNING_BATCH_NORMAL_LAYER_H
#define MOBILE_DEEP_LEARNING_BATCH_NORMAL_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"
namespace mdl {
    class BatchNormalLayer: public Layer {
    public:
        BatchNormalLayer(const Json &config);
        ~BatchNormalLayer();
        void forward(int thread_num);

        // means of each channels
        Matrix *_mean;

        // variance of each channels
        Matrix *_variance;

        // temp data matrix
        Matrix *_temp;

        int _channels;

        float _eps = 0.000010;

        // assist matrix for gemm
        Matrix *_batch_sum_multiplier;
        Matrix *_num_by_chans;
        Matrix *_spatial_sum_mutiplier;




    };
}



#endif //MOBILE_DEEP_LEARNING_BATCH_NORMAL_LAYER_H
