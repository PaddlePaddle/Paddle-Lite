//
// Created by Wang,Liu(MMS) on 2017/10/11.
//

#ifndef MOBILE_DEEP_LEARNING_PERMUTE_H
#define MOBILE_DEEP_LEARNING_PERMUTE_H

#include "base/layer.h"
#include "commons/commons.h"
namespace mdl {
    class PermuteLayer : Layer {
    public:
        PermuteLayer(const Json &config);
        ~PermuteLayer();
        void forward(int thread_num);

    };
}



#endif //MOBILE_DEEP_LEARNING_PERMUTE_H
