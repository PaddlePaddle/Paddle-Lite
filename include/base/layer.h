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
#ifndef MDL_LAYER_H
#define MDL_LAYER_H

#include "base/matrix.h"

namespace mdl {
    /**
     * abstract layer type
     *
     */
    enum class LayerType {CONCAT, CONVOLUTION, FULLCONNECT, LRN, POOL,
        RELU, SPLIT, SCALE, BATCHNORMAL, SOFTMAX, SIGMOID, BIAS,
    ELTWISE};

    class Layer {
    public:

        /**
         * init layer with a json object which specifies the name & weight matix & shape of input & output matrix
         * @param config
         * @return
         */
        Layer(const Json &config);

        virtual ~Layer() {
        }

        virtual void forward(int thread_num = 1) {
        }

        /**
         * get output matrices of current layer
         * @return vector of matrices
         */
        vector<Matrix *> output() {
            return _output;
        }

#ifdef NEED_DUMP
        vector<Matrix *> weight() {
            return _weight;
        }
#endif

        /**
         * get input matrices of current layer
         * @return  vector of matrices
         */
        vector<Matrix *> input() {
            return _input;
        }

        void assure_memory() {
            // LOGI("%s : %p", _name.c_str(), _output[0]);
        }

        void descript() {
            // LOGI("%s : %s", _name.c_str(), _output[0]->descript().c_str());
        }


        /**
         * get layer name
         * @return -name
         */
        string name() {
            return _name;
        }

        /**
         * get layer_type
         * @return _layer_type
         */
        LayerType layer_type() {
            return _layer_type;
        }

        /**
         * get pid of the layer
         * @return _pid
         */
        int pid() {
            return _pid;
        }

    protected:
        int _pid;

        string _name;

        enum LayerType _layer_type;

        vector<Matrix *> _input;

        vector<Matrix *> _output;

        vector<Matrix *> _weight;
    };
};

#endif
