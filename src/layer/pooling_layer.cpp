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
#include "layer/pooling_layer.h"

namespace mdl {
    PoolingLayer::PoolingLayer(const Json &config): Layer(config) {
        auto &param = config["param"];
        _layer_type = LayerType::POOL;
        _pid = config["pid"].int_value();
        _type = param["type"].string_value();
        _global_pooling = param["global_pooling"].bool_value();

        if (_global_pooling) {
            _pad = 0;
            _stride = 1;
            _kernel_size = _input[0]->dimension(2);
        } else {
            _kernel_size = param["kernel_size"].int_value();
            _pad = param["pad"].int_value();
            _stride = param["stride"].int_value();

        }
        assure_memory();
    }

    void PoolingLayer::forward(int thread_num) {
        if (_type == "max") {
            forward_max();
        } else if (_type == "ave") {
            forward_ave();
        }
        descript();
    }

    void PoolingLayer::forward_max() {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        int channel = _input[0]->dimension(1);
        int height = _input[0]->dimension(2);
        int width = _input[0]->dimension(3);
        int pool_height = (int)ceil((float)(height + 2 * _pad - _kernel_size) / _stride) + 1;
        int pool_width = (int)ceil((float)(width + 2 * _pad - _kernel_size) / _stride) + 1;
        for (int c = 0; c < channel; c++) {
            for (int ph = 0; ph < pool_height; ph++) {
                for (int pw = 0; pw < pool_width; pw++) {
                    int hstart = ph * _stride - _pad;
                    int wstart = pw * _stride - _pad;
                    int hend = min(hstart + _kernel_size, height + _pad);
                    int wend = min(wstart + _kernel_size, width + _pad);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    hend = min(hend, height);
                    wend = min(wend, width);
#ifdef ANDROID
                    if (hend - hstart != 3 || wend - wstart != 3) {
                        float max_value = -INT_MAX;
                        for (int h = hstart; h < hend; h++) {
                            for (int w = wstart; w < wend; w++) {
                                float value = input_data[h * width + w];
                                if (value > max_value) {
                                    max_value = value;
                                }
                            }
                        }
                        output_data[ph * pool_width + pw] = max_value;
                    } else {
                        const float *pos1 = input_data + hstart * width + wstart;
                        const float *pos2 = input_data + (hstart + 1) * width + wstart;
                        const float *pos3 = input_data + (hstart + 2) * width + wstart;
                        const float32x4_t data1 = vld1q_f32(pos1);
                        const float32x4_t data2 = vld1q_f32(pos2);
                        const float32x4_t data3 = vld1q_f32(pos3);
                        const float32x4_t max_data = vmaxq_f32(vmaxq_f32(data1, data3), data2);
                        float32x2_t res = vpmax_f32(vget_high_f32(vsetq_lane_f32(-INT_MAX, max_data, 3)), vget_low_f32(max_data));
                        res = vpmax_f32(res, res);
                        output_data[ph * pool_width + pw] = vget_lane_f32(res, 0);
                    }
#else
                    float max_value = -INT_MAX;
                    for (int h = hstart; h < hend; h++) {
                        for (int w = wstart; w < wend; w++) {
                            float value = input_data[h * width + w];
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                    output_data[ph * pool_width + pw] = max_value;
#endif
                }
            }
            input_data += _input[0]->offset({0, 1});
            output_data += _output[0]->offset({0, 1});
        }
    }

    void PoolingLayer::forward_ave() {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        int channel = _input[0]->dimension(1);
        int height = _input[0]->dimension(2);
        int width = _input[0]->dimension(3);
        int pool_height = (int)ceil((float)(height + 2 * _pad - _kernel_size) / _stride) + 1;
        int pool_width = (int)ceil((float)(width + 2 * _pad - _kernel_size) / _stride) + 1;
        for (int c = 0; c < channel; c++) {
            for (int ph = 0; ph < pool_height; ph++) {
                for (int pw = 0; pw < pool_width; pw++) {
                    int hstart = ph * _stride - _pad;
                    int wstart = pw * _stride - _pad;
                    int hend = min(hstart + _kernel_size, height + _pad);
                    int wend = min(wstart + _kernel_size, width + _pad);
                    int pool_size = (hend - hstart) * (wend - wstart);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    hend = min(hend, height);
                    wend = min(wend, width);
                    float sum = 0;
                    for (int h = hstart; h < hend; h++) {
                        for (int w = wstart; w < wend; w++) {
                            sum += input_data[h * width + w];
                        }
                    }
                    sum /= pool_size;
                    output_data[ph * pool_width + pw] = sum;
                }
            }
            input_data += _input[0]->offset({0, 1});
            output_data += _output[0]->offset({0, 1});
        }
    }
};
