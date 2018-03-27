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
#include "layer/batch_normal_layer.h"

#include "math/gemm.h"

namespace mdl {

    BatchNormalLayer::BatchNormalLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::BATCHNORMAL;
        _channels = _input[0]->dimension(1);
        vector<int> sz;
        sz.push_back(_channels);

        _mean = new Matrix();
        _mean->resize(sz);
        _mean->reallocate(0);

        _variance = new Matrix();
        _variance->resize(sz);
        _variance->reallocate(0);

        _temp = new Matrix();
        _temp->resize(_input[0]->get_dimensions());
        _temp->reallocate(0);

        _batch_sum_multiplier = new Matrix();
        sz[0] = _input[0]->dimension(0);
        _batch_sum_multiplier->resize(sz);
        _batch_sum_multiplier->reallocate(1.0);

        int spatial_dim = _input[0]->count() / (_channels * _input[0]->dimension(0));
        _spatial_sum_mutiplier = new Matrix();
        sz[0] = spatial_dim;
        _spatial_sum_mutiplier->resize(sz);
        _spatial_sum_mutiplier->reallocate(1.0);

        int numbychans = _channels * _input[0]->dimension(0);
        sz[0] = numbychans;
        _num_by_chans = new Matrix();
        _num_by_chans->resize(sz);
        _num_by_chans->reallocate(1.0);
    }

    BatchNormalLayer::~BatchNormalLayer() {
        if (_mean != nullptr) {
            delete _mean;
            _mean = nullptr;
        }

        if (_variance != nullptr) {
            delete _variance;
            _variance = nullptr;
        }
        if (_temp != nullptr) {
            delete _temp;
            _temp = nullptr;
        }

        if (_batch_sum_multiplier != nullptr) {
            delete _batch_sum_multiplier;
            _batch_sum_multiplier = nullptr;
        }
        if (_spatial_sum_mutiplier != nullptr) {
            delete _spatial_sum_mutiplier;
            _spatial_sum_mutiplier = nullptr;
        }
        if (_num_by_chans != nullptr) {
            delete _num_by_chans;
            _num_by_chans = nullptr;
        }

    }

    void BatchNormalLayer::forward(int thread_num) {
        
        if(_output[0]->get_data() != _input[0]->get_data())
        {
            _output[0]->set_data(_input[0]->get_data());
        }
        float scale_factor = _weight[2]->get_data()[0] == 0 ? 0 : 1 / _weight[2]->get_data()[0];

        int num = _input[0]->dimension(0);

        int spatial_dim = _input[0]->count() / (_input[0]->dimension(0) * _channels);

        Math::v_scale(_variance->count(),
                      scale_factor,
                      _weight[0]->get_data(),
                      _mean->get_data());

        Math::v_scale(_variance->count(),
                      scale_factor,
                      _weight[1]->get_data(),
                      _variance->get_data());

        // replicate mean to input size
        Gemmer::gemmers[0]->sgemm(num, _channels, 1,
                                  _batch_sum_multiplier->get_data(),
                                  _mean->get_data(),
                                  _num_by_chans->get_data(),
                                  1, 0.);

        // substact mean for data in output matrix
        Gemmer::gemmers[0]->sgemm(_channels * num, spatial_dim,
                                  1,
                                  _num_by_chans->get_data(),
                                  _spatial_sum_mutiplier->get_data(),
                                  _output[0]->get_data(),
                                  -1, 1.);
        // normalize variance in case divide by zero
        Math::v_add(_variance->count(),
                    _eps, _variance->get_data());

        Math::v_pow(_variance->count(),
                    _variance->get_data(),
                    0.5f, _variance->get_data());

        // replicate variance to input size
        Gemmer::gemmers[0]->sgemm(num, _channels, 1,
                                  _batch_sum_multiplier->get_data(),
                                  _variance->get_data(),
                                  _num_by_chans->get_data(),
                                  1, 0.);

        Gemmer::gemmers[0]->sgemm(_channels * num, spatial_dim,
                                  1, _num_by_chans->get_data(),
                                  _spatial_sum_mutiplier->get_data(),
                                  _temp->get_data(), 1.0, 0.);

        Math::v_div(_temp->count(), _output[0]->get_data(),
                    _temp->get_data(),
                    _output[0]->get_data());
    }
}
