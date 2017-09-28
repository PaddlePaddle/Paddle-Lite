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
#include "layer/scale_layer.h"

namespace mdl {
    ScaleLayer::ScaleLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::SCALE;
        auto &param = config["param"];
        _bias_term = param["bias_term"].int_value();
        if (_bias_term > 0) {
            _bias_layer = new BiasLayer(config);

        }
        _scale = _weight[0];

        _outer_dim = _input[0]->count(0, 1);
        _scale_dim = _scale->count();
        _inner_dim = _input[0]->count(1 + _scale->get_dimensions().size());

    }

    ScaleLayer::~ScaleLayer() {
        if (_bias_layer != nullptr) {
            delete _bias_layer;
            _bias_layer = nullptr;
        }
    }

    void ScaleLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        for (int n = 0; n < _outer_dim; ++n) {
            for (int d = 0; d < _scale_dim; ++d) {
                float factor = _scale->get_data()[d];
                Math::v_scale(_inner_dim, factor, input_data, output_data);
                input_data += _inner_dim;
                output_data += _inner_dim;

            }

        }

        if (_bias_layer) {
            _bias_layer->forward();

        }
    }
}
