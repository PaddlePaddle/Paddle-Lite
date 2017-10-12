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

#include "layer/flatten_layer.h"

namespace mdl {
    FlattenLayer::FlattenLayer(const Json &config) : Layer(config) {
        auto &param = config["param"];
        flatten_start = param["start"].int_value();
        int input_dimensions = _input[0]->get_dimensions().size();
        flatten_end = param["end"].int_value();
        if (flatten_end < 0) {
        flatten_end +=input_dimensions;
        }
        vector<int> output_shape;
        for (int i = 0; i < flatten_start; ++i) {
            output_shape.push_back(_input[0]->dimension(i));
        }
        int flatten_size = _input[0]->count(flatten_start, flatten_end);
        for (int j = flatten_end + 1; j < input_dimensions; ++j) {
            output_shape.push_back(_input[0]->dimension(j));

        }
        _output[0]->resize(output_shape);

    }

    FlattenLayer::~FlattenLayer() {}

    void FlattenLayer::forward(int thread_num) {
        _output[0]->set_data(_input[0]->get_data());

    }
}