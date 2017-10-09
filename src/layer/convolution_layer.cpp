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
#include "layer/convolution_layer.h"
#include "math/gemm.h"
#include <fstream>
#include <thread>

namespace mdl {
    ConvolutionLayer::ConvolutionLayer(const Json &config) : Layer(config), _col_buffer(nullptr),
                                                             _bias_buffer(nullptr) {
        assure_memory();
        _layer_type = LayerType::CONVOLUTION;
        _pid = config["pid"].int_value();
        auto &param = config["param"];
        _output_num = param["output_num"].int_value();
        _kernel_size = param["kernel_size"].int_value();
        _pad = param["pad"].int_value();
        _stride = param["stride"].int_value();
        // _bias_term default = 1
        if (param.object_items().count("bias_term")) {
            _bias_term = param["bias_term"].int_value();
        } else {
            _bias_term = 1;
        }
        // _group default = 1
        if (param.object_items().count("group")) {
            _group = param["group"].int_value();
        } else {
            _group = 1;
        }
        _need_im2col = !((_kernel_size == 1) && (_pad == 0) && (_stride == 1));
        if (_need_im2col) {
            _col_buffer = new Matrix();
            _col_buffer->resize({_weight[0]->count(1) * _group, _output[0]->dimension(2), _output[0]->dimension(3)});
            _col_buffer->reallocate();
        }
        _bias_buffer = new Matrix();
        _bias_buffer->resize({_output[0]->count(2)});
        _bias_buffer->reallocate();
        std::fill(_bias_buffer->get_data(), _bias_buffer->get_data() + _bias_buffer->count(), 1.0);
    }

    ConvolutionLayer::~ConvolutionLayer() {
        if (_col_buffer != nullptr) {
            delete _col_buffer;
            _col_buffer = nullptr;
        }
        if (_bias_buffer != nullptr) {
            delete _bias_buffer;
            _bias_buffer = nullptr;
        }
    }

    void run1(int id, int m, int n, int k, const float *A, const float *B, float *C) {
        Gemmer::gemmers[id]->sgemm(m, n, k, A, B, C);
    }

    void run2(int id, int m, int n, int k, const float *A, const float *B, float *C, float alpha, float beta) {
        Gemmer::gemmers[id]->sgemm(m, n, k, A, B, C, alpha, beta);
    }

    void ConvolutionLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        float *weight_data = _weight[0]->get_data();

        int input_offset = _input[0]->count(1);
        int output_offset = _output[0]->count(1);

        int input_num = _input[0]->count(0, 1);

        for (int i = 0; i < input_num; i++) {
            forward_gemm(input_data + i * input_offset, weight_data, output_data + i * output_offset, thread_num);
            if (_bias_term == 1) {
                float *bias_data = _weight[1]->get_data();
                forward_bias(output_data + i * output_offset, bias_data, thread_num);
            }

        }
        descript();
    }

    void ConvolutionLayer::forward_gemm(float *input_data, float *weight_data, float *output_data, int thread_num) {
        int input_channel = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);
        // try im2col
        float *col_data = input_data;
        if (_need_im2col) {
            col_data = _col_buffer->get_data();
            im2col(input_data, input_channel, input_height, input_width, _kernel_size, _pad, _stride, col_data);
        }
        int m = _output[0]->dimension(1);
        int n = _output[0]->count(2);
        int k = _weight[0]->count(1);
        int weight_offset_ = m * k / _group;
        int col_offset_ = k * n;
        int output_offset_ = m * n / _group;

#ifdef MULTI_THREAD
        if (thread_num > 1) {
            std::thread *ths = new std::thread[thread_num];
            int m1 = m / thread_num;
            int m2 = m % thread_num;
            for (int i = 0; i < thread_num; i++) {
                int row_count = m1;
                if (i == thread_num - 1) {
                    row_count = m1 + m2;
                }
                ths[i] = std::thread(run1, i, row_count, n, k, weight_data + i * m1 * k, col_data,
                                     output_data + i * m1 * n);
            }
            for (int j = 0; j < thread_num; ++j) {
                ths[j].join();

            }
            delete []ths;
            return;
        }

        int pid = this->pid();
        if (pid < 1 || pid > Gemmer::gemmers.size()) {
            pid = 1;
        }
#else
        int pid = 1;
#endif

        for (int i = 0; i < _group; ++i) {
            Gemmer::gemmers[pid - 1]->sgemm(m / _group, n, k, weight_data + weight_offset_ * i,
                                            col_data + col_offset_ * i, output_data + output_offset_ * i);

        }

    }


    void ConvolutionLayer::forward_bias(float *output_data, float *bias_data, int thread_num) {
        int m = _output_num;
        int n = _output[0]->count(2);
        int k = 1;

#ifdef MULTI_THREAD
        if (thread_num > 1) {
            std::thread *ths = new std::thread[thread_num];
            int m1 = m / thread_num;
            int m2 = m % thread_num;
            for (int i = 0; i < thread_num; i++) {
                int row_count = m1;
                if (i == thread_num - 1) {
                    row_count = m1 + m2;
                }
                ths[i] = std::thread(run2, i, row_count, n, k, bias_data + i * m1 * k, _bias_buffer->get_data(),
                                     output_data + i * m1 * n, 1.0, 1.0);
            }
            for (int j = 0; j < thread_num; ++j) {
                ths[j].join();

            }
            delete []ths;
            return;
        }

        int pid = this->pid();
        if (pid < 1 || pid > Gemmer::gemmers.size()) {
            pid = 1;
        }
#else
        int pid = 1;
#endif

        Gemmer::gemmers[pid - 1]->sgemm(m, n, k, bias_data, _bias_buffer->get_data(), output_data, 1.0, 1.0);
    }

};

