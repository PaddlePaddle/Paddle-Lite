

#ifdef FLATTEN2_OP

#include "operators/kernel/flatten2_kernel.h"

namespace paddle_mobile{
namespace operators{

template <>
bool Flatten2Kernel<GPU_CL, float >::Init(paddle_mobile::operators::FlattenParam<paddle_mobile::GPU_CL> *param) {
    this->cl_helper_.AddKernel("flatten2", "flatten2_kernel.cl");
    return true;
}

template <>
void Flatten2Kernel<GPU_CL, float >::Compute(
        const paddle_mobile::operators::FlattenParam<paddle_mobile::GPU_CL> &param) {

}

}
}

#endif