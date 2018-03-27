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

#include "layer/elt_wise_layer.h"

namespace mdl {
    EltWiseLayer::EltWiseLayer(const Json &config):Layer(config) {
        assure_memory();
        _layer_type = LayerType::ELTWISE;
        _coeffs = vector<float>(_input[0]->count(), 1);
        auto &param = config["param"];
        _type = param["type"].string_value();
        const vector<Json> &coeffs= param["coeffs"].array_items();
        for (int i = 0; i < coeffs.size(); ++i) {
            _coeffs[i] = coeffs[i].number_value();
        }
    }

    EltWiseLayer::~EltWiseLayer() {}

    void EltWiseLayer::forward(int thread_num) {
        if (_type == "max") {
            forward_max();
        } else if (_type == "product") {
            forward_product();
        } else if (_type == "sum") {
            forward_sum();
        }
        descript();
    }

    void EltWiseLayer::forward_max() {
        float *output_data = _output[0]->get_data();
        int input_count = _input.size();
        int matrix_size = _input[0]->count();
        float *input_data_0 = _input[0]->get_data();
        float *input_data_1 = _input[1]->get_data();

        for (int i = 0; i < matrix_size; ++i) {
            output_data[i] = std::max(input_data_0[i], input_data_1[i]);
        }

        for (int j = 2; j < input_count; ++j) {
            float *tmp = _input[j]->get_data();
            for (int k = 0; k < matrix_size; ++k) {
                output_data[k] = std::max(output_data[k], tmp[k]);
            }

        }


    }

    void EltWiseLayer::forward_product() {
        float *output_data = _output[0]->get_data();
        int input_count = _input.size();
        int matrix_size = _input[0]->count();
        float *input_data_0 = _input[0]->get_data();
        float *input_data_1 = _input[1]->get_data();

        Math::v_mul(input_count, input_data_0, input_data_1, output_data);

        for (int i = 2; i < input_count; ++i) {

            Math::v_mul(input_count, output_data, _input[i]->get_data(), output_data);

        }
    }

    void EltWiseLayer::forward_sum() {
        _output[0]->set(0);
        float *output_data = _output[0]->get_data();
        int input_count = _input.size();
        int matrix_size = _input[0]->count();
        for (int i = 0; i < input_count; ++i) {
            Math::axpy(matrix_size,_coeffs[i],_input[i]->get_data(), output_data);

        }


    }
}


