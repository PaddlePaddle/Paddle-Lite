

#ifdef FLATTEN2_OP

#include "operators/flatten2_op.h"
namespace paddle_mobile{
namespace operators{
template <typename DeviceType, typename T>
void Flatten2Op<DeviceType, T>::InferShape() const {

}

}
}

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(flatten2, ops::Flatten2Op);
#endif



#endif