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
#include "layer/sigmoid_layer.h"

namespace mdl {
    SigmoidLayer::SigmoidLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::SIGMOID;

    }

    SigmoidLayer::~SigmoidLayer() {

    }

    inline float sigmoid(float x) {
        return 1. / (1. + exp(-x));
    }


    void SigmoidLayer::forward(int thread_num) {
        float *inputdata = _input[0]->get_data();
        float *outputdata = _output[0]->get_data();
        int count = _output[0]->count();

        for (int i = 0; i < count; ++i) {
            outputdata[i] = sigmoid(inputdata[i]);

        }


    }
}
