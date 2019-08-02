



#ifdef DENSITY_PRIORBOX_OP

#include <operators/kernel/prior_box_kernel.h>

namespace paddle_mobile{
namespace operators{

template <>
bool DensityPriorBoxKernel<GPU_CL, float >::Init(
        paddle_mobile::operators::DensityPriorBoxParam<paddle_mobile::GPU_CL> *param) {
    this->cl_helper_.AddKernel("density_prior_box","density_prior_box_kernel.cl");
    return true;
}

template <>
void DensityPriorBoxKernel<GPU_CL, float >::Compute(
        const paddle_mobile::operators::DensityPriorBoxParam<paddle_mobile::GPU_CL> &param) {

}

}
}


#endif