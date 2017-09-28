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
#include "layer/bias_layer.h"
#include "math/gemm.h"

namespace mdl {
    BiasLayer::BiasLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::BIAS;
        int axis = 1;
        int num_axis = 1;
        auto begin = _input[0]->get_dimensions().begin();
        vector<int> bias_shape(begin + 1, begin + 2);
        _bias = _weight[1];
        _outer_dim = _input[0]->count(0, axis);
        _bias_dim = _bias->count();
        _inner_dim = _input[0]->count(axis + _bias->get_dimensions().size());
        _dim = _bias_dim * _inner_dim;
        _bias_multiplier = new Matrix();
        _bias_multiplier->resize(vector<int>(1, _inner_dim));
        _bias_multiplier->reallocate(1);

    }

    BiasLayer::~BiasLayer() {
        if (_bias_multiplier != nullptr) {
            delete _bias_multiplier;
            _bias_multiplier = nullptr;
        }

    }

    void BiasLayer::forward() {
        float *output_data = _output[0]->get_data();
        float *bias_data = _bias->get_data();
        for (int i = 0; i < _outer_dim; ++i) {
            Gemmer::gemmers[0]->sgemm(_bias_dim, _inner_dim, 1, bias_data, _bias_multiplier->get_data(),
                                      output_data, 1, 1);
            output_data += _dim;

        }

    }


}

