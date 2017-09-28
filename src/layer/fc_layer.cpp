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
#include "layer/fc_layer.h"

#include "math/gemm.h"

namespace mdl {
    FCLayer::FCLayer(const Json &config): Layer(config) {
        assure_memory();
        auto &param = config["param"];
        _layer_type = LayerType::FULLCONNECT;
        _output_num = param["output_num"].int_value();
        _bias_buffer = new Matrix();
        _bias_buffer->resize({1, _input[0]->count(0, 1)});
        _bias_buffer->reallocate(1.0);
    }

    FCLayer::~FCLayer() {
        if (_bias_buffer != nullptr) {
            delete _bias_buffer;
            _bias_buffer = nullptr;
        }
    }

    void FCLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        float *weight_data = _weight[0]->get_data();
        float *bias_data = _weight[1]->get_data();
        int m = _input[0]->count(0, 1);
        int n = _output_num;
        int k = _input[0]->count(1);

        Gemmer::gemmers[0]->sgemm(m, n, k, input_data, weight_data, output_data);
        Gemmer::gemmers[0]->sgemm(m, n, 1, _bias_buffer->get_data(), bias_data, output_data, 1.0, 1.0);

        descript();
    }
};
