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
#include "base/matrix.h"

namespace mdl {
    Matrix::Matrix(): 
        _data(nullptr), _data_size(0), _dimensions_size(0), _is_external_memory(false) {
    }

    Matrix::Matrix(const Json &config):
        _data(nullptr), _data_size(0), _dimensions_size(0), _is_external_memory(false) {
        vector<int> dimensions;

        for (int index = 0; index < config.array_items().size(); index++) {
            dimensions.push_back(config[index].int_value());
        }

        resize(dimensions);
    }
    
    Matrix::~Matrix() {
        clear_data();
    }

    void Matrix::resize(const vector<int> &dimensions) {
        _dimensions = dimensions;
        _dimensions_size = dimensions.size();
        _data_size = count(0);
    }

    void Matrix::reallocate(float value) {
        clear_data();
        _is_external_memory = false;
        _data = new float[_data_size];
        std::fill(_data, _data + _data_size, value);
    }

    void Matrix::clear_data() {
        if (_data != nullptr) {
            if (!_is_external_memory) {
                delete[] _data;
            }
            _data = nullptr;
        }
    }
};
