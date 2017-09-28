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
#include "layer/lrn_layer.h"

namespace mdl {
    LrnLayer::LrnLayer(const Json &config): Layer(config) {
        assure_memory();
        auto &param = config["param"];
        _layer_type = LayerType::LRN;
        _alpha = param["alpha"].number_value();
        _beta = param["beta"].number_value();
        _local_size = param["local_size"].int_value();



        _scale_buffer = new Matrix();
        _scale_buffer->resize(_input[0]->get_dimensions());
        _scale_buffer->reallocate();
        _sqr_buffer = new Matrix();
        auto sqr_size = _input[0]->get_dimensions();
        sqr_size[0] = 1;
        sqr_size[1] += _local_size - 1;
        _sqr_buffer->resize(sqr_size);
        _sqr_buffer->reallocate();
        std::fill(_sqr_buffer->get_data(), _sqr_buffer->get_data() + _sqr_buffer->count(), 0.0);
    }

    LrnLayer::~LrnLayer() {
        if (_scale_buffer != nullptr) {
            delete _scale_buffer;
            _scale_buffer = nullptr;
        }
        if (_sqr_buffer != nullptr) {
            delete _sqr_buffer;
            _sqr_buffer = nullptr;
        }
    }

    void LrnLayer::forward(int thread_num) {
        const float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        float *scale_data = _scale_buffer->get_data();
        std::fill(scale_data, scale_data + _scale_buffer->count(), 1.0);
        float *sqr_data = _sqr_buffer->get_data();
        float alpha_by_size = _alpha / _local_size;
        int input_num = _input[0]->dimension(0);
        int input_channel = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);
        int pre_pad = (_local_size - 1) / 2;
        for (int n = 0; n < input_num; n++) {
            Math::v_sqr(input_channel * input_height * input_width, input_data + _input[0]->offset({n}), sqr_data + _sqr_buffer->offset({0, pre_pad}));
            for (int i = 0; i < _local_size; i++) {
                Math::axpy(input_height * input_width, alpha_by_size, sqr_data + _sqr_buffer->offset({0, i}), scale_data + _scale_buffer->offset({n, 0}));
            }
            for (int i = 1; i < input_channel; i++) {
                float *scale_buffer_data_start = scale_data + _scale_buffer->offset({n, i - 1});
                std::copy(scale_buffer_data_start, scale_buffer_data_start + input_height * input_width, scale_data + _scale_buffer->offset({n, i}));
                Math::axpy(input_height * input_width, alpha_by_size, sqr_data + _sqr_buffer->offset({0, _local_size + i - 1}), scale_data + _scale_buffer->offset({n, i}));
                Math::axpy(input_height * input_width, -alpha_by_size, sqr_data + _sqr_buffer->offset({0, i - 1}), scale_data + _scale_buffer->offset({n, i}));
            }
        }
        Math::v_pow(_scale_buffer->count(), scale_data, -_beta, output_data);
        Math::v_mul(_scale_buffer->count(), output_data, input_data, output_data);
        descript();
    }
};

